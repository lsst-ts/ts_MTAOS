# This file is part of ts_MTAOS.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["MTAOS"]

import asyncio
import inspect
import json
import logging
import time
import typing

import eups
import numpy as np
import yaml
from astropy import units as u
from lsst.ts import salobj
from lsst.ts.idl.enums.MTAOS import ClosedLoopState, FilterType
from lsst.ts.observatory.control.maintel.comcam import ComCam
from lsst.ts.observatory.control.maintel.lsstcam import LsstCam
from lsst.ts.ofc import OFCData
from lsst.ts.utils import astropy_time_from_tai_unix, make_done_future

from . import CONFIG_SCHEMA, TELESCOPE_DOF_SCHEMA, Config, Model, __version__, utility

try:
    from lsst.ts.ofc import __version__ as __ofc_version__
except ImportError:
    __ofc_version__ = "unknown"

try:
    from lsst.ts.wep import __version__ as __wep_version__
except ImportError:
    __wep_version__ = "unknown"


class MTAOS(salobj.ConfigurableCsc):
    # Class attribute comes from the upstream BaseCsc class
    valid_simulation_modes = (0,)
    version = __version__

    DEFAULT_TIMEOUT = 10.0
    LONG_TIMEOUT = 60.0
    LOG_FILE_NAME = "MTAOS.log"
    MAX_TIME_SAMPLE = 100
    CMD_TIMEOUT = 60.0

    def __init__(
        self, config_dir=None, log_to_file=False, log_level=None, simulation_mode=0
    ):
        """Initialize the MTAOS CSC class.

        MTAOS: Main telescope active optical system.
        CSC: Commandable SAL component.
        SAL: Service abstraction layer.

        Parameters
        ----------
        config_dir : str or None, optional
            Directory of configuration files, or None for the standard
            configuration directory (obtained from get_default_config_dir()).
            This is provided for unit testing. (the default is None.)
        log_to_file : bool, optional
            Output the log to files. The files will be in logs directory. (the
            default is False.)
        log_level : int or str, optional
            Logging level of file handler. It can be "DEBUG" (10), "INFO" (20),
            "WARNING" (30), "ERROR" (40), or "CRITICAL" (50). (the default is
            None.)
        simulation_mode : int, optional
            Simulation mode. (the default is 0: do not simulate.)

        Attributes
        ----------
        log : `logging.Logger`
            A logger.
        state0DofValidator : `salobj.DefaultingValidator`
            Validator for the telescopedof configuration file.
        visit_id_offset: `int`
            Offset applied to visit id. TODO (DM-31365): Remove workaround to
            visitId being of type long in MTAOS runWEP command.
        remotes : `dict`
            A dictionary with `salobj.Remote` for each component the MTAOS
            communicates with.
        issue_correction_to : `set`
            Set with the name of the component in self.remote that also makes
            the name of the method to issue the correction, e.g.,
            `m2hex` -> `issue_m2hex_correction`.
        model : `None` or `lsst.ts.mtaos.Model`
            MTAOS Model class. This attribute is initialized during
            configuration.
        issue_correction_lock : `asyncio.Lock`
            A lock used to synchronize sending corrections to the components.
        wep_config : `dict`
            Default configuration for the wep. This is used in `do_runWEP()`,
            when the user does not provide an override configuration.
        execution_times : `dict`
            Dictionary to store critical execution times.
        ocps: `salobj.Remote`
            Remote for the OCPS component.
        mtcs: `lsst.ts.mtaos.MTCS`
            MTCS CSC.
        camera: `lsst.ts.observatory.control.maintel.lsstcam.LsstCam` or
            `lsst.ts.observatory.control.maintel.comcam.ComCam`
            Camera object.
        stress_scale_approach: `str`
            Approach to scale or truncate the bending modes to keep the total
            stress within limits.
        stress_scale_factor: `float`
            Factor to scale the bending modes to keep the total stress within
            limits.
        m1m3_stress_limit: `float`
            Maximum allowable stress on the M1M3 mirror.
        m2_stress_limit: `float`
            Maximum allowable stress on the M2 mirror.
        use_ocps: `bool`
            Flag to use OCPS for the WEP process.
        DEFAULT_TIMEOUT : `float`
            Default timeout (in seconds). Used on normal operations, e.g.
            issuing corrections to the CSCs.
        LONG_TIMEOUT : `float`
            Long timeout (in seconds). Used in operations that will take longer
            than normal operations, e.g. running OFC.
        LOG_FILE_NAME : `str`
            Name of the log file.
        MAX_TIME_SAMPLE : `int`
            Maximum samples of execution times to store.
        """

        cscName = utility.getCscName()

        super().__init__(
            cscName,
            index=0,
            config_schema=CONFIG_SCHEMA,
            config_dir=config_dir,
            initial_state=salobj.State.STANDBY,
            simulation_mode=int(simulation_mode),
        )

        self.evt_softwareVersions.set(subsystemVersions=self.get_subsystems_versions())

        # Logger attribute comes from the upstream Controller class
        if log_to_file:
            log_dir = utility.getLogDir()
            log_path = log_dir.joinpath(self.LOG_FILE_NAME)
            utility.addRotFileHandler(
                self.log, log_path, logging.DEBUG if log_level is None else log_level
            )

        self.log.info("Prepare MTAOS CSC.")

        self.state0DofValidator = salobj.DefaultingValidator(
            schema=TELESCOPE_DOF_SCHEMA
        )

        # TODO (DM-31365): Remove workaround to visitId being of type long in
        # MTAOS runWEP command.
        #
        # Offset applied to the visit ids when getting images from butler. This
        # is mainly to work around an issue with xml < 9.2 where the visit id
        # is defined as a a long which allows values from -2147483648 to
        # 2147483647
        self.visit_id_offset = 0

        # Dictionary with remotes for M2 Hexapod, Camera Hexapod, M1M3 and M2
        # components. Note the use of include=[] in all remotes. This prevents
        # the remote from subscribing to events and telemetry from those
        # systems that we do not need, helping to solve resources.
        self.remotes = {
            "m2hex": salobj.Remote(
                self.domain,
                "MTHexapod",
                index=utility.MTHexapodIndex.M2.value,
                include=[],
            ),
            "camhex": salobj.Remote(
                self.domain,
                "MTHexapod",
                index=utility.MTHexapodIndex.Camera.value,
                include=[],
            ),
            "m1m3": salobj.Remote(
                self.domain, "MTM1M3", include=["appliedActiveOpticForces"]
            ),
            "m2": salobj.Remote(self.domain, "MTM2", include=["axialForce"]),
        }

        self.execution_times = {}

        # Set with the name of the component in self.remote that also makes the
        # name of the method to issue the correction, e.g.
        # m2hex -> issue_m2hex_correction
        self.issue_correction_to = {
            "m2hex",
            "camhex",
            "m1m3",
            "m2",
        }
        # Minimum forces to apply for m1m3.
        # If no force is larger than this value, in the
        # figure, forces won't be applied.
        self.m1m3_min_forces_to_apply = 1e-3
        # Minimum forces to apply for m2.
        # If no force is larger than this value, in the
        # figure, forces won't be applied.
        self.m2_min_forces_to_apply = 1e-3

        self.ocps = salobj.Remote(self.domain, "OCPS", 101)
        self.mtcs = salobj.Remote(self.domain, "MTCS", 101)

        self._camera = None
        self.closed_loop_task = make_done_future()

        # Model class to do the real data processing
        self.model = None

        # Lock to prevent multiple issueCorrection commands to execute at the
        # same time.
        self.issue_correction_lock = asyncio.Lock()

        self.wep_config = dict()

        self.log.info("MTAOS CSC is ready.")

    async def configure(self, config: typing.Any) -> None:
        """Configure the CSC.

        Parameters
        ----------
        config : `object`
            The configuration as described by the schema at ``schema_path``,
            as a struct-like object.

        Raises
        ------
        `salobj.ExpectedError`
            If fails to parse/validate `wep_config`.

        Notes
        -----
        Called when running the ``start`` command, just before changing
        summary state from `State.STANDBY` to `State.DISABLED`.
        """

        self._logExecFunc()
        self.log.debug("MTAOS configuration started.")

        # TODO (DM-31365): Remove workaround to visitId being of type long in
        # MTAOS runWEP command.
        self.visit_id_offset = config.visit_id_offset

        config_obj = Config(config)
        state0_dof_file = config_obj.getState0DofFile()
        dof_state0 = None

        if state0_dof_file is not None:
            with open(state0_dof_file) as fp:
                dof_state0 = self.state0DofValidator.validate(yaml.safe_load(fp))

        ofc_config_dir = self.config_dir / "ofc"
        if ofc_config_dir.exists():
            self.log.debug(f"Setting OFC configuration directory to {ofc_config_dir!r}")
        else:
            self.log.debug("Using default OFC configuration directory.")
            ofc_config_dir = None

        ofc_data = OFCData(
            name=config.instrument,
            config_dir=ofc_config_dir,
            log=self.log,
        )

        self.log.debug("ofc data ready. Creating model")

        self.model = Model(
            instrument=config.instrument,
            data_path=config.data_path,
            ofc_data=ofc_data,
            log=self.log,
            run_name=config.run_name,
            collections=config.collections,
            pipeline_instrument=(
                config.pipeline_instrument
                if hasattr(config, "pipeline_instrument")
                else None
            ),
            pipeline_n_processes=config.pipeline_n_processes,
            zernike_table_name=config.zernike_table_name,
            data_instrument_name=(
                config.data_instrument_name
                if hasattr(config, "data_instrument_name")
                else None
            ),
        )

        if dof_state0 is not None:
            self.model.ofc_data.dof_state0 = dof_state0

        if hasattr(config, "wep_config"):
            with open(self.config_dir / config.wep_config) as fp:
                self.wep_config = yaml.safe_load(fp)
                try:
                    self.model.wep_configuration_validation[config.instrument].validate(
                        self.wep_config
                    )
                except Exception as e:
                    self.log.exception("Failed to validate WEP configuration.")
                    raise salobj.ExpectedError(
                        f"Failed to validate WEP configuration with {e}. "
                        "Check CSC logs for more information."
                    )
        else:
            self.wep_config = dict()

        if config.camera == "comcam":
            self.camera = ComCam(
                self.domain,
                log=self.log,
                tcs_ready_to_take_data=self.mtcs.ready_to_take_data,
            )
        elif config.camera == "lsstCam":
            self.camera = LsstCam(
                self.domain,
                log=self.log,
                tcs_ready_to_take_data=self.mtcs.ready_to_take_data,
            )
        self.oods = self.camera.rem.ccoods
        self.use_ocps = config.use_ocps
        self.used_dofs = config.used_dofs

        # Set the stress scale approach, factor, and limits
        self.stress_scale_approach = config.stress_scale_approach
        self.stress_scale_factor = config.stress_scale_factor
        self.m1m3_stress_limit = config.m1m3_stress_limit
        self.m2_stress_limit = config.m2_stress_limit

        self.log.debug("MTAOS configuration completed.")

    async def end_enable(self, data):
        """Runs after CSC goes into enable.

        Parameters
        ----------
        data : `DataType`
            Command data
        """

        await self.pubEvent_degreeOfFreedom()

    def _logExecFunc(self):
        """Log the executed function."""

        funcName = inspect.stack()[1].function
        self.log.info(f"Execute {funcName}().")

    @staticmethod
    def get_config_pkg():
        return "ts_config_mttcs"

    async def start(self):
        self._logExecFunc()

        await super().start()

    async def begin_disable(self, data):
        """Begin do_disable; called before state changes.

        Parameters
        ----------
        data : `DataType`
            Command data
        """
        try:
            await self.model.interrupt_wep_process()
        except Exception:
            self.log.exception("Error trying to interrupt wep process.")

    async def do_resetCorrection(self, data):
        """Command to reset the current wavefront error calculations.

        When resetting the wavefront corrections it is recommended that the
        issueWavefrontCorrection command be sent to push the cleared wavefront
        corrections to the AOS (active optical system) subsystems.

        Parameters
        ----------
        data : object
            Data for the command being executed.
        """

        self._logExecFunc()
        self.assert_enabled()

        # If resetting wavefront error fails (e.g. raise an exception) command
        # will be rejected. Events will not be published.
        self.model.reset_wfe_correction()

        await self.pubEvent_degreeOfFreedom()
        await self.pubEvent_m2HexapodCorrection()
        await self.pubEvent_cameraHexapodCorrection()
        await self.pubEvent_m1m3Correction()
        await self.pubEvent_m2Correction()

    async def do_issueCorrection(self, data):
        """Command to issue the wavefront corrections to the M2 hexapod, camera
        hexapod, M1M3, and M2 using the most recently measured wavefront error.

        Parameters
        ----------
        data : object
            Data for the command being executed.
        """

        self._logExecFunc()
        self.assert_enabled()

        # This command may take some time to execute, so will send
        # ack_in_progress with estimated timeout.
        await self.cmd_issueCorrection.ack_in_progress(
            data,
            timeout=self.DEFAULT_TIMEOUT,
            result="issueCorrection started.",
        )

        # We don't want multiple commands to be executed at the same time.
        # This lock will block any subsequent command from being executed until
        # this one is done.
        async with self.issue_correction_lock:
            # This is where the bulk of the work is done. If any correction
            # fail this method will take care of unsetting the ones that
            # succedded and generate a report at the end. Also, if it fails,
            # it raises an exception and the command is rejected.
            await self.handle_corrections()
            await self.pubEvent_mirrorStresses()

    async def do_rejectCorrection(self, data):
        """Reject the most recent wavefront correction.

        Parameters
        ----------
        data : object
            Data for the command being executed.
        """

        self._logExecFunc()
        self.assert_enabled()

        await self.pubEvent_rejectedDegreeOfFreedom()
        self.model.reject_correction()

        await self.pubEvent_degreeOfFreedom()
        await self.pubEvent_m2HexapodCorrection()
        await self.pubEvent_cameraHexapodCorrection()
        await self.pubEvent_m1m3Correction()
        await self.pubEvent_m2Correction()

    async def do_selectSources(self, data):
        """Run source selection algorithm for a specific field and visit
        configuration.

        Parameters
        ----------
        data : object
            Data for the command being executed.

        """
        self.assert_enabled()

        # This command may take some time to execute, so will send
        # ack_in_progress with estimated timeout.
        await self.cmd_selectSources.ack_in_progress(
            data,
            timeout=self.LONG_TIMEOUT,
            result="selectSources started.",
        )

        await self.model.select_sources(
            ra=data.ra * u.hourangle.to(u.deg),
            dec=data.decl,
            sky_angle=data.pa,
            obs_filter=FilterType(data.filter),
            mode=data.mode,
        )

    async def do_preProcess(self, data):
        """Pre-process image for WEP.

        Parameters
        ----------
        data : object
            Data for the command being executed.
        """
        self.assert_enabled()

        # This command may take some time to execute, so will send
        # ack_in_progress with estimated timeout.
        await self.cmd_preProcess.ack_in_progress(
            data,
            timeout=self.LONG_TIMEOUT,
            result="preProcess started.",
        )

        if data.useOCPS:
            raise NotImplementedError("Use OCPS not implemented.")
        else:
            # TODO (DM-31365): Remove workaround to visitId being of type long
            # in MTAOS runWEP command.
            await self.model.pre_process(
                visit_id=self.visit_id_offset + data.visitId,
                config=yaml.safe_load(data.config),
            )

    async def do_runWEP(self, data):
        """Process wavefront data.

        Parameters
        ----------
        data : object
            Data for the command being executed.
        """
        self.assert_enabled()

        # This command may take some time to execute, so will send
        # ack_in_progress with estimated timeout.
        await self.cmd_runWEP.ack_in_progress(
            data,
            timeout=self.LONG_TIMEOUT,
            result="runWEP started.",
        )

        intra_visit_id = self.visit_id_offset + data.visitId
        extra_visit_id = (
            self.visit_id_offset + data.extraId if data.extraId > 0 else None
        )

        if data.useOCPS:
            try:
                self.log.debug("Check if visit was already processed.")
                await self.model.query_ocps_results(
                    self.model.instrument,
                    intra_visit_id,
                    extra_visit_id,
                    timeout=1,
                )
            except asyncio.TimeoutError:
                self.log.debug("Image not processed yet.")

                if extra_visit_id is None:
                    config = {"LSSTCam-FROM-OCS_CWFSIMAGE": f"{intra_visit_id}"}
                else:
                    config = {
                        "LSSTComCam-FROM-OCS_DONUTPAIR": f"{intra_visit_id},{extra_visit_id}"
                    }

                start_time = time.time()
                await self.ocps.cmd_execute.set_start(
                    config=json.dumps(config),
                    timeout=self.DEFAULT_TIMEOUT,
                )

                if "RUN_WEP" not in self.execution_times:
                    self.execution_times["RUN_WEP"] = []
                self.execution_times["RUN_WEP"].append(time.time() - start_time)

                await self.model.query_ocps_results(
                    self.model.instrument, intra_visit_id, extra_visit_id
                )
        else:
            # timestamp command was sent in ISO 8601 compliant date-time format
            # (YYYY-MM-DDTHH:MM:SS.sss), removing invalid characters.
            timestamp_sent_isot = (
                astropy_time_from_tai_unix(data.private_sndStamp)
                .isot.replace("-", "")
                .replace(":", "")
                .replace(".", "")
            )
            private_identity = data.private_identity.replace("@", "_").replace("-", "_")

            run_name_extention = f"_{private_identity}_{timestamp_sent_isot}"

            # TODO (DM-31365): Remove workaround to visitId being of type long
            # in MTAOS runWEP command.
            await self.model.run_wep(
                visit_id=intra_visit_id,
                extra_id=extra_visit_id,
                config=(
                    yaml.safe_load(data.config)
                    if len(data.config) > 0
                    else self.wep_config
                ),
                run_name_extention=run_name_extention,
                log_time=self.execution_times,
            )

            while len(self.execution_times["RUN_WEP"]) > self.MAX_TIME_SAMPLE:
                self.execution_times["RUN_WEP"].pop(0)

        await self.pubEvent_wavefrontError()
        await self.pubEvent_rejectedWavefrontError()
        await self.pubEvent_wepDuration()

    async def do_runOFC(self, data):
        """Run OFC on the latest wavefront errors data. Before running this
        command, you must have ran runWEP at least once.

        This command will run OFC to compute corrections but won't apply them.
        Use `issueCorrection` to apply the corrections. This allow users to
        evaluate whether the corrections are sensible before applying them.

        Parameters
        ----------
        data : object
            Data for the command being executed.
        """

        self.assert_enabled()

        # This command may take some time to execute, so will send
        # ack_in_progress with estimated timeout.
        await self.cmd_runOFC.ack_in_progress(
            data,
            timeout=self.LONG_TIMEOUT,
            result="runOFC started.",
        )

        async with self.issue_correction_lock:
            kp = self.model.ofc.controller.kp
            if data.userGain != 0.0:
                self.model.ofc.controller.kp = data.userGain

            config = yaml.safe_load(data.config) if len(data.config) > 0 else dict()

            # Set the ofc_data values based on configuration
            # This is needed to set what degrees of freedom will be used,
            # how many zernikes, etc.
            self.log.debug("Customizing OFC parameters.")
            original_ofc_data_values = await self.model.set_ofc_data_values(**config)

            try:
                # If this call fails (raise an exeception), command will be
                # rejected.
                # This is not a coroutine so it will block the event loop. Need
                # to think about how to fix it, maybe run in executor?
                self.model.calculate_corrections(
                    log_time=self.execution_times, **config
                )
                self.model.wavefront_errors.clear()
            finally:
                self.log.info("Restore ofc data values.")
                await self.model.set_ofc_data_values(**original_ofc_data_values)
                self.model.ofc.controller.kp = kp

            while (
                len(self.execution_times["CALCULATE_CORRECTIONS"])
                > self.MAX_TIME_SAMPLE
            ):
                self.execution_times["CALCULATE_CORRECTIONS"].pop(0)

            self.log.debug("Calculate the subsystem correction successfully.")

            await self.pubEvent_degreeOfFreedom()
            await self.pubEvent_mirrorStresses()
            await self.pubEvent_m2HexapodCorrection()
            await self.pubEvent_cameraHexapodCorrection()
            await self.pubEvent_m1m3Correction()
            await self.pubEvent_m2Correction()
            await self.pubEvent_ofcDuration()

    async def do_addAberration(self, data):
        """Utility command to add aberration to the system based on user
        provided wavefront errors. The command assume uniform aberration on all
        sensors.

        Parameters
        ----------
        data : object
            Data for the command being executed.
        """
        self.assert_enabled()

        # This command may take some time to execute, so will send
        # ack_in_progress with estimated timeout.
        await self.cmd_addAberration.ack_in_progress(
            data,
            timeout=self.LONG_TIMEOUT,
            result="addAberration started.",
        )

        # We don't want multiple commands to be executed at the same time.
        # This lock will block any subsequent command from being executed until
        # this one is done.
        async with self.issue_correction_lock:
            config = yaml.safe_load(data.config)

            if config is not None:
                self.log.debug("Customizing OFC parameters.")
                original_ofc_data_values = await self.model.set_ofc_data_values(
                    **config
                )
            else:
                original_ofc_data_values = dict()

            try:
                self.model.add_correction(wavefront_errors=data.wf, config=config)
            finally:
                if len(original_ofc_data_values) > 0:
                    self.log.debug("Restoring ofc_data values.")
                    await self.model.set_ofc_data_values(**original_ofc_data_values)

            await self.pubEvent_degreeOfFreedom()
            await self.pubEvent_m2HexapodCorrection()
            await self.pubEvent_cameraHexapodCorrection()
            await self.pubEvent_m1m3Correction()
            await self.pubEvent_m2Correction()

    async def do_interruptWEP(self, data: salobj.type_hints.BaseDdsDataType) -> None:
        """Interrupt a running wep process.

        Parameters
        ----------
        data : ``cmd_interrupWEP.DataType``
        """
        self.assert_enabled()

        await self.model.interrupt_wep_process()

    async def do_offsetDOF(self, data: salobj.type_hints.BaseDdsDataType) -> None:
        """Implement command offsetDOF.

        Parameters
        ----------
        data : salobj.type_hints.BaseDdsDataType
            Data for the command.

        Raises
        ------
        NotImplementedError
            Command not implemented yet.
        """
        self.assert_enabled()

        await self.cmd_offsetDOF.ack_in_progress(
            data,
            timeout=self.LONG_TIMEOUT,
            result="offsetDOF started.",
        )

        async with self.issue_correction_lock:
            self.model.offset_dof(offset=np.array(data.value))

            # if the corrections fails it will republish the dof event
            # after undoing the offsets.
            await self.handle_corrections()
            await self.pubEvent_degreeOfFreedom()
            await self.pubEvent_mirrorStresses()
            await self.pubEvent_m2HexapodCorrection()
            await self.pubEvent_cameraHexapodCorrection()
            await self.pubEvent_m1m3Correction()
            await self.pubEvent_m2Correction()

    async def do_resetOffsetDOF(self, data: salobj.type_hints.BaseDdsDataType) -> None:
        """Implement command reset offset dof.

        Parameters
        ----------
        data : salobj.type_hints.BaseDdsDataType
            Data for the command.

        Raises
        ------
        NotImplementedError
            Command not implemented yet.
        """
        self.assert_enabled()

        await self.cmd_resetOffsetDOF.ack_in_progress(
            data,
            timeout=self.LONG_TIMEOUT,
            result="Reset dof offset started.",
        )

        async with self.issue_correction_lock:
            self.model.reset_wfe_correction()

            # if the corrections fails it will republish the dof event
            # after undoing the offsets.
            await self.handle_corrections()
            await self.pubEvent_degreeOfFreedom()
            await self.pubEvent_mirrorStresses()

    def do_stopClosedLoop(self):
        """Stop the closed loop operation."""
        self._logExecFunc()

        self.closed_loop_task.cancel()
        await self.evt_closedLoopState.set_write(state=ClosedLoopState.IDLE)

    def do_startClosedLoop(self):
        """Start the closed loop operation."""
        self.closed_loop_task = asyncio.create_task(self.closed_loop())

    async def closed_loop(self):
        """Closed loop operation."""
        self._logExecFunc()

        prev_elevation = None
        while self.summary_state == salobj.State.ENABLED:
            try:
                await self.evt_closedLoopState.set_write(
                    state=ClosedLoopState.WAITING_IMAGE
                )

                self.oods.evt_imageInOODS.flush()
                image_in_oods = await self.oods.evt_imageInOODS.next(
                    flush=False, timeout=self.exposure_time
                )
                self.log.info(
                    f"Image {image_in_oods.obsid} {image_in_oods.raft} {image_in_oods.sensor} ingested."
                )

                await self.evt_closedLoopState.set_write(
                    state=ClosedLoopState.PROCESSING
                )

                await self.do_runWEP.set_start(
                    visitId=image_in_oods.obsid,
                    extraId=None,
                    useOCPS=self.use_ocps,
                    config=self.wep_config,
                    timeout=self.CMD_TIMEOUT,
                )

                filter, rotation_angle, elevation = await self.model.get_image_info(
                    image_in_oods.obsid
                )

                if prev_elevation is not None and abs(elevation - prev_elevation) > 9:
                    self.log.warning(
                        "Elevation delta is greater than 9 deg. Not applying corrections."
                    )
                    prev_elevation = elevation
                    continue

                config = {
                    "filter_name": filter,
                    "rotation_angle": rotation_angle,
                    "comp_dof_idx": {
                        "m2HexPos": [float(val) for val in self.used_dofs[:5]],
                        "camHexPos": [float(val) for val in self.used_dofs[5:10]],
                        "M1M3Bend": [float(val) for val in self.used_dofs[10:30]],
                        "M2Bend": [float(val) for val in self.used_dofs[30:]],
                    },
                }
                config_yaml = yaml.safe_dump(config)

                await self.do_runOFC.set_start(
                    userGain=0.0,
                    config=config_yaml,
                    timeout=self.CMD_TIMEOUT,
                )

                await self.evt_closedLoopState.set_write(
                    state=ClosedLoopState.WAITING_APPLY
                )
                await self.do_issueCorrection.set_start(timeout=self.CMD_TIMEOUT)

            except Exception:
                self.log.exception("Closed loop failed; turning off closed loop mode")
                await self.evt_closedLoopState.set_write(state=ClosedLoopState.ERROR)
                return

    def apply_stress_correction(
        self,
        stresses: np.ndarray[float],
        stress_limit: float,
        dof_aggr: np.ndarray[float],
        start_idx: int,
        end_idx: int,
    ) -> np.ndarray[float]:
        """
        Apply the stress correction by either scaling or
        truncating bending modes to keep the total stress within limits.

        Parameters
        ----------
        stresses : np.ndarray
            The individual bending mode stresses on the mirror.
        stress_limit : float
            The maximum allowable stress on the mirror.
        dof_aggr : np.ndarray
            The aggregated degrees of freedom.
        start_idx : int
            The starting index of the bending modes.
        end_idx : int
            The ending index of the bending modes.

        Returns
        -------
        np.ndarray
            The updated degrees of freedom with the stress correction applied.
        """

        # Get the bending modes within the specified range
        bending_modes = dof_aggr[start_idx:end_idx].copy()
        stress = self.stress_scale_factor * np.sqrt(np.sum(np.square(stresses)))

        # Check if the stress is over the limit
        if stress > stress_limit:
            self.log.warning(
                f"Stress {stress:.2f} psi is above the limit {stress_limit:.2f} psi. Applying correction."
            )

            if self.stress_scale_approach == "scale":
                self.log.warning(
                    "Using scale approach. Applying the same correction but with a lower amplitude."
                )

                scale = stress_limit / stress
                bending_modes *= scale

            elif self.stress_scale_approach == "truncate":
                self.log.warning(
                    "Using truncate approach. Truncating the correction"
                    " to only apply lower-order bending modes."
                )

                for i in reversed(range(len(bending_modes))):
                    if stress <= stress_limit:
                        break  # RSS is within limits, stop truncating

                    # Set the highest remaining bending mode to zero
                    stresses[i] = 0
                    bending_modes[i] = 0

                    # Recalculate RSS with the truncated modes
                    stress = self.stress_scale_factor * np.sqrt(
                        np.sum(np.square(stresses))
                    )

                self.log.warning(
                    f"After truncating, the new total stress is {stress:.2f} psi, "
                    f"which is {'within' if stress <= stress_limit else 'above'} the limit."
                )

            # Update the dof_aggr with the modified bending modes
            dof_aggr[start_idx:end_idx] = bending_modes.copy()

        else:
            self.log.info(
                f"Stress {stress:.2f} psi is within the limit {stress_limit:.2f} psi. Applying correction."
            )

        return dof_aggr

    async def handle_corrections(self):
        """Handle applying the corrections to all components.

        If one or more correction fail to apply to method will try to undo the
        successful corrections. If any of those fails to undo, it will
        continue and generate a report at the end.

        Raises
        ------
        RuntimeError:
            If one or more correction failed.
        """

        aggr_dof = self.model.get_dof_aggr()

        # Ensure the bending modes are within stress limits,
        # otherwise modify them to be within the limits.
        m1m3_stresses = self.model.get_m1m3_bending_mode_stresses()
        m2_stresses = self.model.get_m2_bending_mode_stresses()

        # Apply the stress correction to the M1M3 mirror
        dof_aggr_m1m3_stress_corrected = self.apply_stress_correction(
            m1m3_stresses, self.m1m3_stress_limit, aggr_dof, 10, 30
        )

        # Apply the stress correction to the M2 mirror
        dof_aggr_stress_corrected = self.apply_stress_correction(
            m2_stresses, self.m2_stress_limit, dof_aggr_m1m3_stress_corrected, 30, 50
        )

        # Update the model with the corrected degrees of freedom
        self.model.set_dof_aggr(dof_aggr_stress_corrected)
        self.model.get_updated_corrections()

        # Issue all corrections concurrently. If any of them fails, undo
        # corrections and reject command.
        issue_corrections_tasks = dict(
            [
                (comp, asyncio.create_task(getattr(self, f"issue_{comp}_correction")()))
                for comp in self.issue_correction_to
            ]
        )

        # Wait for all corrections to complete
        await asyncio.gather(
            *[task for task in issue_corrections_tasks.values()],
            return_exceptions=True,
        )

        # Check if there was any exception. If so, undo all successfull
        # corrections and reject command.
        if any(
            [task.exception() is not None for task in issue_corrections_tasks.values()]
        ):
            await self.pubEvent_rejectedDegreeOfFreedom()
            self.model.reject_correction()
            await self.pubEvent_degreeOfFreedom()

            # Undo corrections that completed.
            error_repor = await self.handle_undo_corrections(issue_corrections_tasks)

            raise RuntimeError(error_repor)

    async def handle_undo_corrections(self, issued_corrections):
        """Handle undoing corrections.

        The method will inspect the `issued_corrections` list of tasks, will
        undo all the successful corrections and log the unsuccessful. If any
        successful correction fail to be undone, it will log the issue and skip
        the error.

        At the end return a report with the activities performed.

        Parameters
        ----------
        issued_corrections : `list` of `asyncio.Task`
            List with all the tasks that executed corrections.

        Returns
        -------
        error_report : `str`
            String with error report.
        """

        failed_to_do = []
        failed_to_undo = []

        # Loop through all the components
        for comp in self.issue_correction_to:
            if issued_corrections[comp].exception() is None:
                # If the task exception is None it means the task completed
                # successfully and the correction needs to be undone. If it
                # fails to undo the exception log the error and continue.
                self.log.warning(f"Undoing {comp} correction.")
                try:
                    await getattr(self, f"issue_{comp}_correction")()
                except Exception:
                    self.log.exception(
                        f"Failed to undo successful correction in {comp}."
                    )
                    failed_to_undo.append(comp)
            else:
                # Correction failed, store it as failed and continue.
                failed_to_do.append(comp)
        # Generate a report about the issue.
        error_report = f"Failed to apply correction to: {failed_to_do}. "
        if len(failed_to_undo) > 0:
            error_report += f"Failed to undo correction to: {failed_to_undo}"

        return error_report

    async def issue_m2hex_correction(self, undo=False):
        """Issue the correction of M2 hexapod.

        Parameters
        ----------
        undo : bool
            If `True` apply the negative value of each correction.
        """

        x, y, z, u, v, w = self.model.m2_hexapod_correction()

        if undo:
            x, y, z, u, v, w = -x, -y, -z, -u, -v, -w

        try:
            await self.remotes["m2hex"].cmd_move.set_start(
                timeout=self.DEFAULT_TIMEOUT, x=x, y=y, z=z, u=u, v=v, w=w, sync=True
            )

            self.log.debug("Issue the M2 hexapod correction successfully.")

        except Exception:
            self.log.exception("M2 hexapod correction command failed.")
            await self.pubEvent_rejectedM2HexapodCorrection()
            raise

    async def issue_camhex_correction(self, undo=False):
        """Issue the correction of camera hexapod.

        Parameters
        ----------
        undo : bool
            If `True` apply the negative value of each correction.
        """

        x, y, z, u, v, w = self.model.cam_hexapod_correction()

        if undo:
            x, y, z, u, v, w = -x, -y, -z, -u, -v, -w

        try:
            await self.remotes["camhex"].cmd_move.set_start(
                timeout=self.DEFAULT_TIMEOUT, x=x, y=y, z=z, u=u, v=v, w=w, sync=True
            )

            self.log.debug("Issue the camera hexapod correction successfully.")

        except Exception:
            self.log.exception("Camera hexapod correction command failed.")
            await self.pubEvent_rejectedCameraHexapodCorrection()
            raise

    async def issue_m1m3_correction(self, undo=False):
        """Issue the correction of M1M3.

        Parameters
        ----------
        undo : bool
            If `True` apply the negative value of each correction.

        """
        z_forces = self.model.m1m3_correction()

        if undo:
            z_forces = np.negative(z_forces)

        try:
            should_apply = True
            try:
                applied_active_optics_forces = await self.remotes[
                    "m1m3"
                ].evt_appliedActiveOpticForces.aget(timeout=self.DEFAULT_TIMEOUT)
                delta_z_forces = z_forces - applied_active_optics_forces.zForces
                should_apply = np.any(
                    np.abs(delta_z_forces) > self.m1m3_min_forces_to_apply
                )
            except asyncio.TimeoutError:
                self.log.warning(
                    "Could not determine currently applied AOS forces for M1M3. "
                    "Applying full figure."
                )
                should_apply = True
            if should_apply:
                await self.remotes["m1m3"].cmd_applyActiveOpticForces.set_start(
                    timeout=self.DEFAULT_TIMEOUT, zForces=z_forces
                )
                self.log.debug("Issue the M1M3 correction successfully.")
            else:
                self.log.info(
                    "Skipping applying m1m3 forces. "
                    f"No values above threshold of {self.m1m3_min_forces_to_apply}N."
                )

        except Exception:
            self.log.exception("M1M3 correction command failed.")
            await self.pubEvent_rejectedM1M3Correction()
            raise

    async def issue_m2_correction(self, undo=False):
        """Issue the correction of M2.

        Parameters
        ----------
        undo : bool
            If `True` apply the negative value of each correction.

        """

        z_forces = self.model.m2_correction()

        if undo:
            z_forces = np.negative(z_forces)

        try:
            try:
                axial_forces = await self.remotes["m2"].tel_axialForce.aget(
                    timeout=self.DEFAULT_TIMEOUT
                )
                delta_forces = z_forces - axial_forces.applied
                if np.all(np.abs(delta_forces) < self.m2_min_forces_to_apply):
                    self.log.info(
                        f"Delta forces for M2 all below threshold ({self.m2_min_forces_to_apply}N). Skipping."
                    )
                    return
            except asyncio.TimeoutError:
                self.log.info(
                    "Could not determine the current M2 axial forces. Applying full figure."
                )

            await self.remotes["m2"].cmd_applyForces.set_start(
                timeout=self.DEFAULT_TIMEOUT, axial=z_forces
            )

            self.log.debug("Issue the M2 correction successfully.")

        except Exception:
            self.log.exception("M2 correction command failed.")
            await self.pubEvent_rejectedM2Correction()
            raise

    async def pubEvent_wavefrontError(self):
        """Publish the calculated wavefront error calculated by WEP.

        WEP: Wavefront estimation pipeline.
        """

        self._logExecFunc()
        self.model.get_wfe()

        for sensor_id, zernike_indices, zernike_values in zip(
            *self.model.get_wavefront_errors()
        ):
            zernike_indices_extended = np.zeros(100, dtype=int)
            zernike_values_extended = np.full(100, np.nan)
            zernike_indices_extended[: zernike_indices.size] = zernike_indices
            zernike_values_extended[: zernike_values.size] = zernike_values

            await self.evt_wavefrontError.set_write(
                sensorId=sensor_id,
                nollZernikeIndices=zernike_indices_extended,
                nollZernikeValues=zernike_values_extended,
                force_output=True,
            )
            await asyncio.sleep(0.1)

    async def pubEvent_rejectedWavefrontError(self):
        """Publish the rejected calculated wavefront error calculated by WEP.

        WEP: Wavefront estimation pipeline.
        """

        self._logExecFunc()
        self.model.get_rejected_wfe()

        for sensor_id, zernike_indices, zernike_values in zip(
            *self.model.get_rejected_wavefront_errors()
        ):
            zernike_indices_extended = np.zeros(100, dtype=int)
            zernike_values_extended = np.full(100, np.nan)
            zernike_indices_extended[: zernike_indices.size] = zernike_indices
            zernike_values_extended[: zernike_values.size] = zernike_values

            await self.evt_rejectedWavefrontError.set_write(
                sensorId=sensor_id,
                nollZernikeIndices=zernike_indices_extended,
                nollZernikeValues=zernike_values_extended,
                force_output=True,
            )
            await asyncio.sleep(0.1)

    async def pubEvent_degreeOfFreedom(self):
        """Publish the degree of freedom generated by the OFC calculation.

        OFC: Optical feedback control.
        """

        self._logExecFunc()

        dofAggr = self.model.get_dof_aggr()
        dofVisit = self.model.get_dof_lv()
        await self.evt_degreeOfFreedom.set_write(
            aggregatedDoF=dofAggr,
            visitDoF=dofVisit,
            force_output=True,
        )

    async def pubEvent_mirrorStresses(self):
        """Publish the calculated mirror stresses
        from the applied degrees of freedom.

        OFC: Optical feedback control.
        """

        self._logExecFunc()

        m1m3_stresses = self.model.get_m1m3_bending_mode_stresses()
        m2_stresses = self.model.get_m2_bending_mode_stresses()

        # Calculate the total stress on the mirror
        m1m3_total_stress = self.stress_scale_factor * np.sqrt(
            np.sum(np.square(m1m3_stresses))
        )
        m2_total_stress = self.stress_scale_factor * np.sqrt(
            np.sum(np.square(m2_stresses))
        )

        await self.evt_mirrorStresses.set_write(
            stressM2=m2_total_stress,
            stressM1M3=m1m3_total_stress,
        )

    async def pubEvent_rejectedDegreeOfFreedom(self):
        """Publish the rejected degree of freedom generated by the OFC
        calculation.

        OFC: Optical feedback control.
        """

        self._logExecFunc()

        dofAggr = self.model.get_dof_aggr()
        dofVisit = self.model.get_dof_lv()
        await self.evt_rejectedDegreeOfFreedom.set_write(
            aggregatedDoF=dofAggr,
            visitDoF=dofVisit,
            force_output=True,
        )

    async def pubEvent_m2HexapodCorrection(self):
        """Publish the M2 hexapod correction that would be commanded if the
        issueWavefrontCorrection command was sent.
        """

        self._logExecFunc()

        x, y, z, u, v, w = self.model.m2_hexapod_correction()
        await self.evt_m2HexapodCorrection.set_write(
            x=x, y=y, z=z, u=u, v=v, w=w, force_output=True
        )

    async def pubEvent_rejectedM2HexapodCorrection(self):
        """Publish the rejected M2 hexapod correction that would be commanded
        if the issueWavefrontCorrection command was sent.
        """

        self._logExecFunc()

        x, y, z, u, v, w = self.model.m2_hexapod_correction()
        await self.evt_rejectedM2HexapodCorrection.set_write(
            x=x, y=y, z=z, u=u, v=v, w=w, force_output=True
        )

    async def pubEvent_cameraHexapodCorrection(self):
        """Publish the camera hexapod correction that would be commanded if the
        issueWavefrontCorrection command was sent.
        """

        self._logExecFunc()

        x, y, z, u, v, w = self.model.cam_hexapod_correction()
        await self.evt_cameraHexapodCorrection.set_write(
            x=x, y=y, z=z, u=u, v=v, w=w, force_output=True
        )

    async def pubEvent_rejectedCameraHexapodCorrection(self):
        """Publish the rejected camera hexapod correction that would be
        commanded if the issueWavefrontCorrection command was sent.
        """

        self._logExecFunc()

        x, y, z, u, v, w = self.model.cam_hexapod_correction()
        await self.evt_rejectedCameraHexapodCorrection.set_write(
            x=x, y=y, z=z, u=u, v=v, w=w, force_output=True
        )

    async def pubEvent_m1m3Correction(self):
        """Publish the M1M3 correction that would be commanded if the
        issueWavefrontCorrection command was sent.
        """

        self._logExecFunc()

        zForces = self.model.m1m3_correction()
        await self.evt_m1m3Correction.set_write(zForces=zForces, force_output=True)

    async def pubEvent_rejectedM1M3Correction(self):
        """Publish the rejected M1M3 correction that would be commanded if the
        issueWavefrontCorrection command was sent.
        """

        self._logExecFunc()

        zForces = self.model.m1m3_correction()
        await self.evt_rejectedM1M3Correction.set_write(
            zForces=zForces, force_output=True
        )

    async def pubEvent_m2Correction(self):
        """Publish the M2 correction that would be commanded if the
        issueWavefrontCorrection command was sent.
        """

        self._logExecFunc()

        zForces = self.model.m2_correction()
        await self.evt_m2Correction.set_write(zForces=zForces, force_output=True)

    async def pubEvent_rejectedM2Correction(self):
        """Publish the rejected M2 correction that would be commanded if the
        issueWavefrontCorrection command was sent.
        """

        self._logExecFunc()

        zForces = self.model.m2_correction()
        await self.evt_rejectedM2Correction.set_write(
            zForces=zForces, force_output=True
        )

    async def pubEvent_wepDuration(self):
        """Publish the duration of the WEP calculation."""

        self._logExecFunc()

        duration = (
            np.mean(self.execution_times["RUN_WEP"])
            if "RUN_WEP" in self.execution_times
            and len(self.execution_times["RUN_WEP"]) > 0
            else 0.0
        )
        await self.evt_wepDuration.set_write(calcTime=duration)

    async def pubEvent_ofcDuration(self):
        """Publish the duration of the OFC calculation."""

        self._logExecFunc()

        duration = (
            np.mean(self.execution_times["CALCULATE_CORRECTIONS"])
            if "CALCULATE_CORRECTIONS" in self.execution_times
            and len(self.execution_times["CALCULATE_CORRECTIONS"]) > 0
            else 0.0
        )
        await self.evt_ofcDuration.set_write(calcTime=duration)

    def get_subsystems_versions(self) -> str:
        """Get subsystems versions string.

        Returns
        -------
        subsystems_versions : `str`
            A comma delimited list of key=value pairs relating subsystem name
            (key) to its version number (value).
        """

        lsst_distrib_version = ":".join(
            eups.Eups().findSetupProduct("lsst_distrib").tags
        )

        return f"ts_ofc={__ofc_version__},ts_wep={__wep_version__},lsst_distrib={lsst_distrib_version}"

    @classmethod
    def add_arguments(cls, parser):
        super(MTAOS, cls).add_arguments(parser)
        parser.add_argument(
            "--log-to-file",
            action="store_true",
            default=False,
            help="Output the log to files. The files will be in logs directory. "
            "The default log level is DEBUG.",
        )
        parser.add_argument(
            "--log-level",
            type=int,
            help="Debug level of log files. It can be DEBUG (10), INFO (20), "
            "WARNING (30), ERROR (40), or CRITICAL (50).",
        )

    @classmethod
    def add_kwargs_from_args(cls, args, kwargs):
        super(MTAOS, cls).add_kwargs_from_args(args, kwargs)
        kwargs["log_to_file"] = args.log_to_file
        kwargs["log_level"] = args.log_level
