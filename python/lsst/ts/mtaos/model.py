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

__all__ = ["Model"]

import asyncio
import concurrent.futures
import contextlib
import copy
import functools
import logging
import os
import shutil
import tempfile
import time
from typing import Optional

import numpy as np
import yaml
from lsst.afw.image import VisitInfo
from lsst.daf import butler as dafButler
from lsst.ts.ofc import OFC, BendModeToForce
from lsst.ts.ofc.utils.ofc_data_helpers import get_intrinsic_zernikes, get_sensor_names
from lsst.ts.salobj import DefaultingValidator
from lsst.ts.utils import make_done_future
from lsst.ts.wep.utils import writePipetaskCmd

from .config_schema import (
    CWFS_PIPELINE_CONFIG,
    GENERATE_DONUT_CATALOG_CONFIG,
    ISR_CONFIG,
    SCIENCE_SENSOR_PIPELINE_CONFIG,
    WEP_HEADER_CONFIG,
)
from .utility import define_visit, get_formatted_corner_wavefront_sensors_ids, timeit
from .wavefront_collection import WavefrontCollection


class Model:
    # Maximum length of queue for wavefront error
    MAX_LEN_QUEUE = 10

    def __init__(
        self,
        instrument,
        data_path,
        ofc_data,
        log=None,
        run_name="mtaos_wep",
        collections="LSSTComCam/raw/all,LSSTComCam/calib",
        pipeline_instrument=None,
        pipeline_n_processes=9,
        data_instrument_name=None,
        reference_detector=0,
        zernike_table_name="zernikes",
    ):
        """MTAOS model class.

        This class implements a model for the MTAOS operations. It encapsulates
        all business logic to isolate the CSC from the operation.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument.
        data_path : `str`
            Path to the data butler.
        ofc_data : `OFCData`
            OFC data container class.
        log : `logging.Logger` or `None`, optional
            Optional logging class to be used for logging operations. If
            `None`, creates a new logger.

        Other Parameters
        ----------------
        run_name : `str`, optional
            Which name to use when running the pipeline task. This defines
            the location where the data is written in the butler.
            Default is "mtaos_wep".
        collections : `str`, optional
            String with the data collections to add to the pipeline task.
            Default is "LSSTComCam/raw/all,LSSTComCam/calib".
        pipeline_instrument : `dict` or `None`, optional
            A dictionary that maps the name of the instrument to the name used
            in the pipeline task. If None, use default dictionary mapping.
        pipeline_n_processes : `int`, optional
            Number of processes to use when running pipeline. Default is 9.
        data_instrument_name : `dict` or `None`, optional
            A dictionary that maps the name of the instrument to the name used
            by the pipeline task to store the data products. If None, use
            default dictionary mapping.
        reference_detector : `int`, optional
            Which detector to use as a referece to construct the WCS.
        zernike_table_name : `str`, optional
            Name of the table in the butler with zernike coeffients.
            Default is "zernikes".

        Attributes
        ----------
        log : `Logger`
            Log facility.
        instrument : `str`
            Name of the instrument.
        data_path : `str`
            Path to the data butler.
        run_name : `str`, optional
            Which name to use when running the pipeline task. This defines
            the location where the data is written in the butler.
            Default is "mtaos_wep".
        collections : `str`, optional
            String with the data collections to add to the pipeline task.
            Default is "LSSTComCam/raw/all,LSSTComCam/calib".
        pipeline_instrument : `dict` or `None`, optional
            A dictionary that maps the name of the instrument to the name used
            in the pipeline task. If None, use default dictionary mapping.
        pipeline_n_processes : `int`, optional
            Number of processes to use when running pipeline. Default is 9.
        data_instrument_name : `dict` or `None`, optional
            A dictionary that maps the name of the instrument to the name used
            by the pipeline task to store the data products. If None, use
            default dictionary mapping.
        reference_detector : `int`, optional
            Which detector to use as a referece to construct the WCS.
        zernike_table_name : `str`, optional
            Name of the table in the butler with zernike coeffients.
            Default is "zernikes".
        wep_configuration_validation : `dict`
            Dictionary to store schema validations for wavefront estimation
            pipeline tasks.
        wavefront_errors : `WavefrontCollection`
            Object to manage list of wavefront errors.
        rejected_wavefront_errors : `WavefrontCollection`
            Object to manage list of rejected wavefront errors.
        fwhm_data : `list` of `FWHMSensorData`
            List of FWHM (full width at half maximum) sensor data.
        ofc : `lsst.ts.ofc.OFC`
            Optical feedback control object.
        m2_hexapod_correction : `M2HexapodCorrection`
            M2 hexapod correction.
        cam_hexapod_correction : `CameraHexapodCorrection`
            Camera hexapod correction.
        m1m3_correction : `M1M3Correction`
            M1M3 correction.
        m2_correction : `M2Correction`
            M2 correction.
        camera : `lsst.ts.wep.bsc.CameraData.CameraData`
            Current camera instance.
        wep_process : `Coroutine`, optional
            Task for the wep process.
        wep_process_started_task : `asyncio.Future`
            A future that is reset before a wep process is started and is set
            to done when it starts.
        """

        if log is None:
            self.log = logging.getLogger(type(self).__name__)
        else:
            self.log = log.getChild(type(self).__name__)

        self.instrument = instrument
        self.data_path = data_path
        self.run_name = run_name

        self.collections = collections
        self.pipeline_instrument = (
            pipeline_instrument
            if pipeline_instrument is not None
            else dict(
                comcam="lsst.obs.lsst.LsstComCam",
                lsstCam="lsst.obs.lsst.LsstCam",
                lsstFamCam="lsst.obs.lsst.LsstCam",
            )
        )
        self.pipeline_n_processes = pipeline_n_processes
        self.data_instrument_name = (
            data_instrument_name
            if data_instrument_name is not None
            else dict(
                comcam="LSSTComCam",
                lsstCam="LSSTCam",
                lsstFamCam="LSSTCam",
            )
        )
        self.zernike_table_name = zernike_table_name
        self.reference_detector = reference_detector

        science_sensor_config_schema = copy.deepcopy(WEP_HEADER_CONFIG)
        science_sensor_config_schema["properties"]["tasks"]["properties"] = dict()
        science_sensor_config_schema["properties"]["tasks"]["properties"].update(
            ISR_CONFIG
        )
        science_sensor_config_schema["properties"]["tasks"]["properties"].update(
            GENERATE_DONUT_CATALOG_CONFIG
        )
        science_sensor_config_schema["properties"]["tasks"]["properties"].update(
            SCIENCE_SENSOR_PIPELINE_CONFIG
        )

        cwfs_config_schema = copy.deepcopy(WEP_HEADER_CONFIG)
        cwfs_config_schema["properties"]["tasks"]["properties"] = dict()
        cwfs_config_schema["properties"]["tasks"]["properties"].update(ISR_CONFIG)
        cwfs_config_schema["properties"]["tasks"]["properties"].update(
            GENERATE_DONUT_CATALOG_CONFIG
        )
        cwfs_config_schema["properties"]["tasks"]["properties"].update(
            CWFS_PIPELINE_CONFIG
        )

        self.wep_configuration_validation = dict(
            comcam=DefaultingValidator(science_sensor_config_schema),
            lsstCam=DefaultingValidator(cwfs_config_schema),
            lsstFamCam=DefaultingValidator(science_sensor_config_schema),
        )

        # Collection of calculated list of wavefront error
        self.wavefront_errors = WavefrontCollection(self.MAX_LEN_QUEUE)

        # Collection of calculated list of rejected wavefront error
        self.rejected_wavefront_errors = WavefrontCollection(self.MAX_LEN_QUEUE)

        # Dictionary of FWHM (full width at half maximum) sensor data
        self._fwhm_data = dict()

        # Optical feedback control
        self.ofc = OFC(ofc_data)

        # M2 hexapod correction
        self.m2_hexapod_correction = None

        # Camera hexapod correction
        self.cam_hexapod_correction = None

        # M1M3 actuator correction
        self.m1m3_correction = None

        # M2 actuator correction
        self.m2_correction = None

        self.wep_process = None
        self.wep_process_started_task = make_done_future()

        # This asyncio.Lock is used to synchronize the initialization of a new
        # wep pipeline task process. The idea is that we want to limit the
        # number of executing processes to 1. If more than one call to
        # `run_wep` are made, we lock the resources before starting the
        # background process and unlock once the process has started. Any
        # additional cal to `run_wep` will then raise an exception while the
        # first one executes. This way we guarantee that only 1 process is
        # running at any time.
        self._wep_process_start_lock = asyncio.Lock()

        self.reset_wfe_correction()

    def get_fwhm_sensors(self):
        """Get list of fwhm sensor ids.

        Returns
        -------
        `list`
            List with the fwhm sensors ids.
        """
        return list(self._fwhm_data.keys())

    def get_fwhm_data(self):
        """Get an ndarray with the FWHM data for all the sensors.

        FWHM: Full width at half maximum.

        Returns
        -------
        `np.ndarray`
            2-D array with the fwhm data. Each element of the array contains an
            array with the fwhm data (in arcsec).
        """

        # Note that the array dtype bellow is object instead of float. The
        # reason is that we need to be able to support vectors with different
        # sizes. For instance, say you have 5 measurements for sensor 1, 7 for
        # sensor 2 and so on. Numpy does not support arrays of arrays with
        # different sizes of type float.
        return np.array(
            [self._fwhm_data[sensor_id] for sensor_id in self._fwhm_data],
            ndmin=1,
            dtype=object,
        )

    def set_fwhm_data(self, sensor_id, fwhm_data):
        """Set the FWHM sensor data.

        FWHM: Full width at half maximum.

        Parameters
        ----------
        sensor_id : int
            Sensor Id.
        fwhm_data : numpy.ndarray
            FWHM values for this sensor.
        """
        self._fwhm_data[sensor_id] = np.array(fwhm_data)

    def reset_fwhm_data(self):
        """Reset fhwm data."""
        self._fwhm_data = dict()

    def get_wfe(self):
        """Get the list of wavefront error from the collection.

        This is to let MtaosCsc to publish the latest calculated wavefront
        error.

        Returns
        -------
        list[lsst.ts.wep.ctrlIntf.SensorWavefrontData]
            List of wavefront error data.
        """

        return self.wavefront_errors.pop()

    def get_rejected_wfe(self):
        """Get the list of rejected wavefront error from the collection.

        This is to let MtaosCsc to publish the latest rejected wavefront
        error.

        Returns
        -------
        list[lsst.ts.wep.ctrlIntf.SensorWavefrontData]
            List of rejected wavefront error data.
        """
        return self.rejected_wavefront_errors.pop()

    def get_dof_aggr(self):
        """Get the aggregated DOF.

        DOF: Degree of freedom.

        Returns
        -------
        numpy.ndarray
            Aggregated DOF.
        """

        return self.ofc.controller.aggregated_state

    def set_dof_aggr(self, dof_aggr):
        """Set the aggregated DOF.

        DOF: Degree of freedom.

        Parameters
        ----------
        dof_aggr : `numpy.ndarray`
            Aggregated DOF.
        """

        self.ofc.controller.set_aggregated_state(dof_aggr)

    def get_dof_lv(self):
        """Get the DOF correction from the last visit.

        DOF: Degree of freedom.

        Returns
        -------
        numpy.ndarray
            DOF correction from the last visit.
        """

        return self.ofc.lv_dof

    def get_m1m3_bending_mode_stresses(self) -> np.ndarray:
        """Get the total M1M3 mirror stresses per bending mode.

        Returns
        -------
        np.ndarray
            Bending mode stresses for M1M3.
        """
        m1m3_bending_mode = BendModeToForce("M1M3", self.ofc.ofc_data)
        indices = self.ofc.ofc_data.dof_indices["M1M3_bending"]

        m1m3_stresses = m1m3_bending_mode.get_stresses_from_dof(
            self.ofc.controller.aggregated_state[indices[0] : indices[1]]
        )

        return m1m3_stresses

    def get_m2_bending_mode_stresses(self) -> np.ndarray:
        """Get the total M2 mirror stresses per bending mode.

        Returns
        -------
        np.ndarray
            Bending mode stresses for M2.
        """
        m2_bending_mode = BendModeToForce("M2", self.ofc.ofc_data)
        indices = self.ofc.ofc_data.dof_indices["M2_bending"]

        m2_stresses = m2_bending_mode.get_stresses_from_dof(
            self.ofc.controller.aggregated_state[indices[0] : indices[1]]
        )

        return m2_stresses

    def reject_correction(self):
        """Reject the correction of subsystems."""

        lv_dof = self.get_dof_lv()

        self.ofc.controller.aggregate_state(-lv_dof, self.ofc.ofc_data.dof_idx)

        (
            self.m2_hexapod_correction,
            self.cam_hexapod_correction,
            self.m1m3_correction,
            self.m2_correction,
        ) = self.ofc.get_all_corrections()

    def reset_wfe_correction(self):
        """Reset the current calculation contains the wavefront error and
        subsystem corrections to be empty.

        This function is needed for the long slew angle of telescope.
        """

        self._clear_wfe_collections()
        (
            self.m2_hexapod_correction,
            self.cam_hexapod_correction,
            self.m1m3_correction,
            self.m2_correction,
        ) = self.ofc.reset()

    def offset_dof(self, offset):
        """Add offset to the degrees of freedom.

        Parameters
        ----------
        offset : `np.array`
            Offset to apply to the degrees of freedom.
        """

        self.ofc.controller.aggregate_state(offset, self.ofc.ofc_data.dof_idx)

        # Update last visit DOF which is the last
        # applied dof not the aggregated one.
        self.ofc.lv_dof = offset

        (
            self.m2_hexapod_correction,
            self.cam_hexapod_correction,
            self.m1m3_correction,
            self.m2_correction,
        ) = self.ofc.get_all_corrections()

    def get_updated_corrections(self):
        """Get the updated corrections."""
        (
            self.m2_hexapod_correction,
            self.cam_hexapod_correction,
            self.m1m3_correction,
            self.m2_correction,
        ) = self.ofc.get_all_corrections()

    def _clear_wfe_collections(self):
        """Clear the collections of wavefront error contain the rejected
        one.
        """

        self.wavefront_errors.clear()
        self.rejected_wavefront_errors.clear()

    async def select_sources(self, ra, dec, sky_angle, obs_filter, mode):
        """Setup and run source selection algorithm.

        Parameters
        ----------
        ra : `float`
            Right ascension in degrees. The value should be in (0, 360).
        dec : `float`
            Declination in degrees. The value should be in (-90, 90).
        sky_angle : `float`
            The sky position angle in degrees (0 to 360). This is the angle
            measured relative to the north celestial pole (NCP), turning
            positive into the direction of the right ascension.
        obs_filter : `lsst.ts.idl.enums.MTAOS.FilterType`
            Filter used while collecting the images.
        mode : `lsst.ts.idl.enums.MTAOS.Mode`
            Enumeration specifying the wfs mode.

        Raises
        ------
        NotImplementedError
            This function is not supported yet (DM-28708).
        """
        # TODO: (DM-28708) Finish implementation of selectSources.
        raise NotImplementedError("This function is not supported yet (DM-28708).")

    async def pre_process(self, visit_id, config):
        """Pre-process image for WEP.

        The outputs of this command are donut images that are ready
        for curvature wavefront sensing.

        Parameters
        ----------
        visit_id : `int`
            Image visit id number.
        config : `dict`
            Configuration for the image processing algorithm.

        Raises
        ------
        NotImplementedError
            This function is not supported yet (DM-28708).
        """
        # TODO: (DM-28708) Finish implementation of preProcess.
        raise NotImplementedError("This function is not supported yet (DM-28708).")

    @timeit
    async def run_wep(
        self,
        visit_id,
        extra_id,
        config,
        run_name_extention="",
        **kwargs,
    ):
        """Process image or images with wavefront estimation pipeline.

        Parameters
        ----------
        visit_id : `int`
            Image visit id number.
        extra_id : `None` or `int`
            Additional image visit id number. If `None`, assume it is
            processing corner wavefront sensors data. This option is only valid
            if data is for the main camera.
        config : `dict`
            Configuration for the wavefront estimation pipeline.
        kwargs :
            Additional keyword arguments, required by the timer decorator.
        """

        if extra_id is None:
            self.log.debug(
                f"Processing MainCamera corner wavefront sensor on image {visit_id}."
            )
            await self.process_lsstcam_corner_wfs(
                visit_id=visit_id,
                config=config,
                run_name_extention=run_name_extention,
            )
        else:
            # If data is intra/extra it must be ComCam at this point. Main
            # camera intra/extra data will be processed exclusively using OCPS.
            self.log.debug(
                f"Processing intra/extra pair: {visit_id}/{extra_id}. Expecting ComCam data."
            )

            await self.process_comcam(
                intra_id=visit_id,
                extra_id=extra_id,
                config=config,
                run_name_extention=run_name_extention,
            )

    async def process_lsstcam_corner_wfs(
        self,
        visit_id,
        config,
        run_name_extention="",
    ):
        """Process LSSTCam Corner Wavefront Sensor data.

        Parameters
        ----------
        visit_id : `int`
            Id of the image to process corner wavefront sensor.
        config : `dict`
            A dictionary with additional configuration for the pipeline task.
        run_name_extention : `str`, optional
            A string to be appended to the run name. Default is "".

        Raises
        ------
        RuntimeError
            If there is an ongoing wep process.
            If pipeline process fails.

        See Also
        --------
        interrupt_wep_process : Interrupt an ongoing wep process.
        """

        self.log.debug(f"Processing LSSTCam corner wavefront sensor: {visit_id}.")

        run_name = f"{self.run_name}{run_name_extention}"

        async with self.handle_wep_process(
            instrument="lsstCam",
            exposures_str=f"exposure IN ({visit_id}) "
            f"AND detector IN ({get_formatted_corner_wavefront_sensors_ids()})",
            run_name=run_name,
            config=config,
        ):
            await self.wep_process.wait()

        self.wavefront_errors.append(
            self._gather_outputs(
                run_name=run_name,
                visit_id=visit_id,
                instrument="lsstCam",
            )
        )

    async def process_comcam(
        self,
        intra_id,
        extra_id,
        config,
        run_name_extention="",
    ):
        """Process ComCam intra/extra focal images.

        Parameters
        ----------
        intra_id : `int`
            Id of the intra-focal image.
        extra_id : `int`
            Id of the extra-focal image.
        config : `dict`
            A dictionary with additional configuration for the pipeline task.
        run_name_extention : `str`, optional
            A string to be appended to the run name.

        Raises
        ------
        RuntimeError
            If there is an ongoing wep process.

        See Also
        --------
        interrupt_wep_process : Interrupt an ongoing wep process.
        """

        self.log.debug(f"Processing ComCam intra/extra pair: {intra_id}/{extra_id}.")

        run_name = f"{self.run_name}{run_name_extention}"

        async with self.handle_wep_process(
            instrument="comcam",
            exposures_str=f"exposure IN ({intra_id}, {extra_id})",
            run_name=run_name,
            config=config,
        ):
            await self.wep_process.wait()

        self.wavefront_errors.append(
            self._gather_outputs(
                run_name=run_name,
                visit_id=intra_id,
                instrument="comcam",
            )
        )

    async def query_ocps_results(self, instrument, intra_id, extra_id, timeout=300):
        """Query the OCPS results."""
        self.wavefront_errors.append(
            await self._poll_butler_outputs(
                intra_id=intra_id,
                extra_id=extra_id,
                instrument=instrument,
                timeout=timeout,
            )
        )

    @contextlib.asynccontextmanager
    async def handle_wep_process(
        self,
        instrument: str,
        exposures_str: str,
        run_name: str = "",
        config: Optional[dict] = None,
    ):
        """A context manager to start and cleanup the WEP pipeline task
        process.

        This async context manager takes care of initializing a the wep
        pipeline task in the background and then cleaning up when it is
        done.

        When using this async context manager, one must wait until the
        `wep_process` finishes before allowing the context to finish, otherwise
        the process will be cleaned up before it is done.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument to generate wep process for.
        exposures_str : `str`
            A string that can be used by the pipeline task to query the data
            to be processed.
        run_name : `str`
            Optional extention to the run name.
        config : `dict`, optional
            User-provided configuration overrides.
        """

        try:
            log_task, config_file = await self._start_wep_process(
                instrument=instrument,
                exposures_str=exposures_str,
                run_name=run_name,
                config=config,
            )

            yield

            await self._close_pending_task(log_task)

            if self.wep_process is not None and self.wep_process.returncode != 0:
                copied_config_file_name = os.path.basename(config_file.name)
                self.log.debug(
                    f"WEP process failed, copying configuration file to {copied_config_file_name}."
                )
                shutil.copyfile(config_file.name, copied_config_file_name)
            config_file.close()

        finally:
            await self._finish_wep_process()

    async def _start_wep_process(
        self,
        instrument: str,
        exposures_str: str,
        run_name: str = "",
        config: Optional[dict] = None,
    ) -> asyncio.Task:
        """Start a wep process.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument to generate wep process for.
        exposures_str : `str`
            A string that can be used by the pipeline task to query the data
            to be processed.
        run_name : `str`, optional
            Optional extention to the run name. Default is "".
        config : `dict`, optional
            User-provided configuration overrides. Default is `None`.

        Returns
        -------
        `asyncio.Task`
            Task with the process background logger generated by `log_stream`.

        See Also
        --------
        log_stream : Log messages from input stream asynchronously.
        _finish_wep_process : Finalize a wep process.
        """

        async with self._wep_process_start_lock:
            if (self.wep_process is not None) and (self.wep_process.returncode is None):
                raise RuntimeError(
                    "There is an ongoing wep process. To run a different process, "
                    "interrupt the first one with 'interrupt_wep_process'."
                )

            self.wep_process_started_task = asyncio.Future()

            await self.define_visit(
                exposures_str=exposures_str,
                instrument=instrument,
            )

            config_file = self._save_wep_configuration(
                instrument=instrument,
                config=config,
            )

            self.log.debug(
                f"Run name: {run_name}. Pipeline configuration in {config_file}."
            )

            run_pipetask_cmd = self._generate_pipetask_command(
                run_name=run_name,
                instrument=instrument,
                config_filename=config_file.name,
                exposures_str=exposures_str,
            )

            self.log.debug(f"Running: {run_pipetask_cmd}")

            # Run pipeline task in a process asynchronously
            self.wep_process = await asyncio.create_subprocess_shell(
                run_pipetask_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        self.wep_process_started_task.set_result(True)

        return (
            asyncio.create_task(self.log_stream(self.wep_process.stderr)),
            config_file,
        )

    async def define_visit(self, exposures_str: str, instrument: str) -> None:
        """Define visit for a pair of images.

        This is required so that the DM pipeline can process the pair of
        intra/extra focal images together.

        Parameters
        ----------
        exposures_str : `str`
            A string that can be used by the pipeline task to query the data
            to be processed.
        instrument : `str`
            Name of the instrument.
        """
        loop = asyncio.get_running_loop()

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
            self.log.debug(
                "Defining visit: "
                f"data_path={self.data_path}, "
                f"collections={self.collections}, "
                f"instrument_name={self.data_instrument_name[instrument]}, "
                f"exposures_str={exposures_str}."
            )

            define_visit_task = loop.run_in_executor(
                pool,
                functools.partial(
                    define_visit,
                    data_path=self.data_path,
                    collections=self.collections.split(","),
                    instrument_name=self.data_instrument_name[instrument],
                    exposures_str=exposures_str,
                ),
            )

            try:
                await define_visit_task
            except Exception:
                self.log.exception("Error defining visit. Pipeline task may fail.")

    async def interrupt_wep_process(self):
        """Interrupt a currently executing processing."""

        if self.wep_process is not None:
            self.log.debug("Waiting for wep process to start.")
            await self.wep_process_started_task
            self.log.debug("Terminating wep process.")
            self.wep_process.terminate()
        else:
            self.log.debug("No wep process running. Nothing to do.")

    async def _finish_wep_process(self) -> None:
        """Finalize wep process.

        Raises
        ------
        RuntimeError
            If wep process failed, is still executing or is not set.

        Notes
        -----
        `self.wep_process.returncode` contains the exit value of the process
        started with asyncio.create_subprocess_shell. A value of zero means the
        process finished successfully, anything else is considered an error. If
        we get return code different than zero, assume the pipeline task
        failed and raise an exception with the error report.

        See Also
        --------
        _start_wep_process : Start a wep process.
        """
        if self.wep_process is None:
            raise RuntimeError("wep_process not set.")
        elif self.wep_process.returncode is None:
            raise RuntimeError("wep_process still executing.")
        elif self.wep_process.returncode != 0:
            self.log.debug(f"Process returned: {self.wep_process.returncode}")

            stdout, stderr = await self.wep_process.communicate()

            if len(stdout) > 0:
                self.log.debug(stdout.decode())

            if len(stderr) > 0:
                self.log.error(stderr.decode())

            raise RuntimeError(f"Error running pipeline task: {stderr.decode()}")
        else:
            self.wep_process = None

    def generate_wep_configuration(
        self,
        instrument: str,
        config: dict,
    ) -> dict:
        """Generate configuration dictionary for running the WEP pipeline based
        on a reference image id and a configuration dictionary.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument.
        config : `dict`
            Additional configuration overrides provided by the user.

        Returns
        -------
        `dict`
            Configuration dictionary validated against the WEP schema.
        """
        wep_configuration = config.copy()
        wep_configuration["instrument"] = self.pipeline_instrument[instrument]

        return self.wep_configuration_validation[instrument].validate(wep_configuration)

    def _save_wep_configuration(
        self,
        instrument,
        config,
    ) -> tempfile._TemporaryFileWrapper:
        """Save wep configuration to a temporary yaml file for running the WEP
        pipeline, based on a reference image id and a configuration dictionary.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument.
        config : `dict`
            Configuration for the WEP pipeline task.

        Returns
        -------
        config_file : `tempfile._TemporaryFileWrapper[str]`
            Handler for the generated configuration file.
        """

        # TODO: Implement configuration when user runs select_sources
        # beforehand.

        wep_configuration = self.generate_wep_configuration(
            instrument=instrument, config=config
        )

        config_file = tempfile.NamedTemporaryFile(suffix=".yaml")

        config_file.write(yaml.safe_dump(wep_configuration).encode())

        config_file.flush()

        return config_file

    def _generate_pipetask_command(
        self,
        run_name,
        instrument,
        config_filename,
        exposures_str,
    ) -> str:
        """Generate pipetask command to execute as a process.

        Parameters
        ----------
        run_name : `str`
            Name of the run.
        instrument : `str`
            Name of the instrument.
        config_filename : `str`
            Name of the configuration file.
        exposures_str : `str`
            String expressing a query for the images to be processed.

        Returns
        -------
        run_pipetask_cmd : `str`
            A formatted string with the command line execution for the
            pipeline task.
        """

        run_pipetask_cmd = writePipetaskCmd(
            self.data_path,
            run_name,
            self.pipeline_instrument[instrument],
            self.collections,
            pipelineYaml=config_filename,
        )

        run_pipetask_cmd += f' -d "{exposures_str}"'
        run_pipetask_cmd += f" -j {self.pipeline_n_processes}"

        return run_pipetask_cmd

    async def get_image_info(self, visit_id):
        """Get image information from the butler.

        Parameters
        ----------
        visit_id : `int`
            Visit id of the image.

        Returns
        -------
        filter : `str`
            Filter of the image.
        rotation_angle : `float`
            Rotation angle of the image.
        elevation : `float`
            Elevation of the image.

        Raises
        ------
        ValueError
            If the visit id is not found in the butler.
        """
        butler = dafButler.Butler(self.data_path)
        refs = butler.query_datasets("raw", where=f"visit={visit_id}")

        if len(refs) == 0:
            raise ValueError(
                f"Visit {visit_id} has no associated images in the butler."
            )

        image = butler.get(refs[0])
        filter = image.getFilter().bandLabel
        rotation_angle = image.getMetadata().get("ROTPA")
        elevation = (
            image.getMetadata().get("ELSTART") + image.getMetadata().get("ELEND")
        ) / 2

        return filter, rotation_angle, elevation

    async def _poll_butler_outputs(
        self,
        intra_id: int,
        extra_id: int,
        instrument: str,
        timeout: int,
        poll_interval: int = 5,
    ) -> list:
        """
        Poll the Butler for the outputs of a given run
        and intra/extra image id, with a timeout.

        Parameters
        ----------
        intra_id : `int`
            Id of the intra-focal image.
        extra_id : `int`
            Id of the extra-focal image.
        instrument : `str`
            Camera used to take the data.
        timeout : `int`, optional
            Maximum time to wait for the outputs (in seconds).
        poll_interval : `int`, optional
            How often to poll for the data (in seconds).

        Returns
        -------
        `list`
            List of wavefront errors from the Butler.

        Raises
        ------
        TimeoutError
            If the dataset is not available within the specified timeout.
        """
        self.log.debug("Polling butler for WEP outputs.")

        butler = dafButler.Butler(self.data_path, collections=[self.run_name])
        start_time = time.time()
        elapsed_time = 0.0
        n_tables = 9

        pair_id = extra_id if extra_id is not None else intra_id
        while elapsed_time < timeout:
            try:
                self.log.info(
                    f"Querying datasets: zernike_table_name={self.zernike_table_name}, "
                    f"{self.run_name=} {pair_id=}."
                )
                refs = butler.query_datasets(
                    self.zernike_table_name,
                    collections=[self.run_name],
                    where=f"visit in ({pair_id})",
                )
                if refs.count() >= n_tables:
                    self.log.debug(f"Query returned {refs.count()} results.")
                    break
                else:
                    self.log.debug(
                        f"Query returned {refs.count()} entries, waiting for {n_tables}. Continuing."
                    )
            except Exception:
                self.log.exception(
                    f"Collection '{self.run_name}' not found. Waiting {poll_interval}s."
                )
                continue
            finally:
                await asyncio.sleep(poll_interval)
                elapsed_time = time.time() - start_time
        else:
            self.log.error(f"Polling loop timed out {timeout=}s, {elapsed_time=}s.")
            raise TimeoutError(
                f"Timeout: Could not find outputs for run '{self.run_name}' "
                f"and visit id {pair_id} within {timeout} seconds."
            )

        self.log.debug(
            f"run_name: {self.run_name}, visit_id: {pair_id} yielded: {refs}"
        )

        return [
            (
                ref.dataId["detector"],
                butler.get(
                    self.zernike_table_name,
                    dataId=ref.dataId,
                    collections=[self.run_name],
                ),
            )
            for ref in refs
        ]

    def _gather_outputs(
        self,
        run_name: str,
        visit_id: int,
        instrument: str,
    ) -> list:
        """Gather outputs from the given run for a given visit id.

        Parameters
        ----------
        run_name : `str`
            Name of the run.
        visit_id : `int`
            Id of the visit.
        instrument : `str`
            Camera used to take the data.

        Returns
        -------
        `list`
            List of wavefront errors from the butler.
        """
        self.log.debug("Data processing completed successfully. Gathering output.")

        butler = dafButler.Butler(self.data_path)

        # We may need to run the following in an executor so we won't block the
        # event loop.

        datasetRefs = list(
            butler.registry.queryDatasets(
                datasetType="postISRCCD", collections=[run_name]
            )
        )
        for ref in datasetRefs:
            self.log.debug(ref.dataId)

        # Get output
        refs = butler.query_datasets(
            self.zernike_table_name,
            collections=[run_name],
        )

        self.log.debug(f"run_name: {run_name}, visit_id: {visit_id} yielded: {refs}")

        return [
            (
                ref.dataId["detector"],
                butler.get(
                    self.zernike_table_name,
                    dataId=ref.dataId,
                    collections=[run_name],
                ),
            )
            for ref in refs
        ]

    def reject_unreasonable_wfe(self, listOfWfErr):
        """Reject the wavefront error that is unreasonable.

        The input listOfWfErr might be updated after calling this function.
        Some elements might be pop out for the bad values.

        Parameters
        ----------
        listOfWfErr : list[lsst.ts.wep.ctrlIntf.SensorWavefrontData]
            List of wavefront error data.
        """

        # Need to have the algorithm to analyze the wavefront error is
        # reasonable or not. At this moment, just assume everything is good.

        return []

    @timeit
    def calculate_corrections(self, **kwargs):
        """Calculate the correction of subsystems based on the average
        wavefront error of multiple exposure images in a single visit.

        Parameters
        ----------
        kwargs :
            Additional keyword arguments, required by the timer decorator.

        Raises
        ------
        RuntimeError
            No FWHM sensor data to use.
        """

        try:
            sensor_ids, zk_indices, wfe = self.get_wavefront_errors()

            self._calculate_corrections(
                wfe=wfe, zk_indices=zk_indices, sensor_ids=sensor_ids, **kwargs
            )

        finally:
            # Clear the queue
            self._clear_wfe_collections()

    def get_wavefront_errors(self):
        """Get wavefront errors.

        Returns
        -------
        sensor_ids : `np.ndarray [int]`
            Array with sensor ids.
        zk_indices: `np.ndarray [int]`
            Array with the zernike noll indices used.
        wfe : `np.ndarray`
            Array of arrays with the zernike coeficients for each field index.
        """

        wfe_data_container = (
            self.wavefront_errors.getListOfWavefrontErrorAvgInTakenData()
        )

        return self.get_sensor_ids_wfe_from_data_container(wfe_data_container)

    def get_rejected_wavefront_errors(self):
        """Get rejected wavefront errors.

        Returns
        -------
        sensor_ids : `np.ndarray [int]`
            Array with sensor ids.
        wfe : `np.ndarray`
            Array of arrays with the zernike coeficients for each field index.
        """

        wfe_data_container = (
            self.rejected_wavefront_errors.getListOfWavefrontErrorAvgInTakenData()
        )

        return self.get_sensor_ids_wfe_from_data_container(wfe_data_container)

    def _calculate_corrections(self, wfe, zk_indices, sensor_ids, **kwargs):
        """Compute corrections from input wavefront errors.

        Parameters
        ----------
        wfe : `np.ndarray`
            2D array with wavefront errors (in microns). Each element contains
            the wavefront errors for a specific field index.
        zk_indices : `np.ndarray [int]`
            Array with the zernike noll indices used.
        sensor_ids : `np.ndarray [int]`
            Array with sensor ids.
        **kwargs : `dict`
            User input keyword arguments. Optional standard kwargs:
                rot: `float`
                    Camera rotation angle in degrees (default 0).
                filter_name: `string`
                    Name of the filter used for the observations.
        """
        self.ofc.ofc_data.zn_selected = zk_indices
        wavefront_error = np.zeros((len(sensor_ids), np.max(zk_indices) - 4 + 1))

        try:
            for i in range(len(sensor_ids)):
                wavefront_error[i][zk_indices - 4] += wfe[i]
        except Exception:
            self.log.exception(f"{wfe=}")
            raise

        rotation_angle = kwargs.get("rotation_angle", 0.0)
        filter_name = kwargs.get("filter_name", "")

        self.log.debug(
            "_calculate_corrections: "
            f"{wfe=}, "
            f"{wavefront_error=}, "
            f"{sensor_ids=}, "
            f"{filter_name=}, "
            f"{rotation_angle=}."
        )

        (
            self.m2_hexapod_correction,
            self.cam_hexapod_correction,
            self.m1m3_correction,
            self.m2_correction,
        ) = self.ofc.calculate_corrections(
            wfe=wavefront_error,
            sensor_ids=sensor_ids,
            filter_name=filter_name,
            rotation_angle=rotation_angle,
        )

        self.log.debug(
            f"{self.m2_hexapod_correction=}, "
            f"{self.cam_hexapod_correction=}, "
            f"{self.m1m3_correction=}, "
            f"{self.m2_correction=}."
        )

    def add_correction(self, wavefront_errors, config=None):
        """Compute ofc corrections from user-defined wavefront erros.

        Parameters
        ----------
        wavefront_errors : `np.array` or `list` of `float`
            Input wavefront errors (in um). If an array or list it must have
            the same number of elements of the intrinsic zernike coeffients.
        config : `dict`, optional
            Optional additional configuration parameters to customize ofc.
            Default is `None`.

        Raises
        ------
        RuntimeError
            No sensor ids to use.
        """

        self.log.debug(f"Currently configured with {self.instrument} instrument.")

        # Get the sensor ids, filter_name and rotation_angle from config.
        # If sensor_ids are not available raise an error.
        filter_name = config.get("filter_name", "")
        rotation_angle = config.get("rotation_angle", 0.0)
        sensor_ids = config.get("sensor_ids", None)
        if sensor_ids is None:
            raise RuntimeError("No sensor ids to use.")

        # Retrieve corresponding sensor_names
        sensor_names = get_sensor_names(self.ofc.ofc_data, sensor_ids)

        # Get the intrinsic zernike coeffients. Will consider white light for
        # now but may use last filter set in select sources in the future.
        self.log.debug("Assuming white light filter to compute aberration.")

        # Note that it subtracts the users input wavefront from the intrinsic
        # data. The ofc will return corrections to remove the measured
        # aberration. That means, if we want to "add" an aberration we have to
        # pass the negative of what we want.

        final_wfe = np.copy(
            get_intrinsic_zernikes(
                self.ofc.ofc_data,
                filter_name,
                sensor_names,
                rotation_angle + 2 * self.ofc.ofc_data.rotation_offset,
            )[:, self.ofc.ofc_data.zn_idx]
        )

        for wfe in final_wfe:
            wfe -= np.array(wavefront_errors)

        self._calculate_corrections(
            wfe=final_wfe,
            zk_indices=np.arange(4, wfe.shape[0] + 4),
            **(config if config is not None else dict()),
        )

    async def set_ofc_data_values(self, **kwargs):
        """Set ofc data values.

        Parameters
        ----------
        **kwargs : `dict`
            Input keyword arguments. The method does not expect any particular
            input.

        Returns
        -------
        original_ofc_data_values : `dict`
            Original values in `ofc_data`.

        Notes
        -----
        For each input argument, check whether it is a valid entry in
        `self.ofc.ofc_data`. If yes, save the original value to the
        `original_ofc_data_values` dictionary and so on. If it fails to set a
        particular value, undo all the other changes and raise the exception.

        Uppon success, return the original values so users can restore it
        later.
        """

        original_ofc_data_values = dict()

        try:
            for key in kwargs:
                if key == "name":
                    self.log.debug(f"Configuring ofc_data for new instrument: {key}.")
                    await self.ofc.ofc_data.configure_instrument(kwargs[key])
                elif hasattr(self.ofc.ofc_data, key):
                    self.log.debug(f"Overriding ofc_data parameter {key}.")
                    original_ofc_data_values[key] = copy.copy(
                        getattr(self.ofc.ofc_data, key)
                    )

                    # Check if there is a type annotation and try to cast the
                    # values as such.
                    if (key in self.ofc.ofc_data.__annotations__) and (
                        self.ofc.ofc_data.__annotations__[key] != np.ndarray
                    ):
                        setattr(
                            self.ofc.ofc_data,
                            key,
                            self.ofc.ofc_data.__annotations__[key](kwargs[key]),
                        )

                    elif (key in self.ofc.ofc_data.__annotations__) and (
                        self.ofc.ofc_data.__annotations__[key] == np.ndarray
                    ):
                        setattr(
                            self.ofc.ofc_data,
                            key,
                            np.array(kwargs[key]),
                        )

                    elif key == "comp_dof_idx":
                        if not isinstance(kwargs[key], dict):
                            raise RuntimeError(
                                f"comp_dof_idx must be a dictionary. Got {type(kwargs[key])}."
                            )

                        new_comp_dof_idx = kwargs[key]

                        for comp_dof_idx_key in new_comp_dof_idx:
                            new_comp_dof_idx[comp_dof_idx_key] = np.array(
                                kwargs[key][comp_dof_idx_key], dtype=bool
                            )
                        self.log.info(
                            f"{self.ofc.ofc_data.comp_dof_idx=}\n{new_comp_dof_idx=}"
                        )
                        self.ofc.ofc_data.comp_dof_idx = new_comp_dof_idx
                        self.ofc.controller.reset_history()
                        original_ofc_data_values[key] = (
                            self.ofc.ofc_data.default_comp_dof_idx
                        )

                    elif key == "controller_filename":
                        self.ofc.ofc_data.controller_filename = kwargs[key]

                    elif key == "xref":
                        self.ofc.ofc_data.xref = kwargs[key]

                    elif key == "zk_selected":
                        self.ofc.ofc_data.zn_selected = kwargs[key]

        except Exception:
            self.log.error(
                "Error setting value in ofc_data. Restoring original values."
            )
            for key in original_ofc_data_values:
                setattr(self.ofc.ofc_data, key, original_ofc_data_values[key])
            raise
        else:
            return original_ofc_data_values

    async def log_stream(self, stream: asyncio.subprocess.PIPE) -> None:
        """Log messages from input stream asynchronously.

        Parameters
        ----------
        stream : `asyncio.subprocess.PIPE`
            Output stream pipe to process and log.
        """

        while not stream.at_eof():
            new_line = await stream.readline()
            if len(new_line) > 0:
                self.log.debug(new_line.decode().strip())

    def _get_visit_info(self, instrument: str, exposure: int) -> VisitInfo:
        """Get visit info from the butler.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument.
        exposure : `int`
            exposure id of data to retrieve information from.

        Returns
        -------
        `VisitInfo`
            Object with information about a single exposure of an imaging
            camera.
        """
        return dafButler.Butler(self.data_path).get(
            "raw.visitInfo",
            dataId={
                "instrument": self.data_instrument_name[instrument],
                "exposure": exposure,
                "detector": self.reference_detector,
            },
            collections=self.collections.split(","),
        )

    @staticmethod
    def get_sensor_ids_wfe_from_data_container(data_container):
        """Parse data container generated from calling
        `WavefrontCollection.getListOfWavefrontErrorAvgInTakenData` into an
        array with field indices and an array of wavefront errors.

        Parameters
        ----------
        data_container : `dict`
            Dictionary returned by
            `WavefrontCollection.getListOfWavefrontErrorAvgInTakenData()`.

        Returns
        -------
        sensor_ids : `np.ndarray [int]`
            Array with sensor ids.
        zk_indices : `np.ndarray`
            Array of arrays with the zernike indices.
        wfe : `np.ndarray`
            Array of arrays with the zernike coeficients.
        """
        sensor_ids = np.array([sensor_id for sensor_id in data_container])

        zk_indices = np.array(
            [data_container[sensor_id][0] for sensor_id in data_container]
        )
        wfe = np.array([data_container[sensor_id][1] for sensor_id in data_container])

        return sensor_ids, zk_indices, wfe

    async def _close_pending_task(self, task: asyncio.Task) -> None:
        """Close a pending task and log any exception.

        Parameters
        ----------
        task : `asyncio.Task`
            Task to close.
        """

        if not task.done():
            task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log.debug(f"Ignoring exception in task: {e}.")
