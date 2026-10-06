# This file is part of ts_mtaos.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import asyncio
import glob
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import numpy as np
import pytest
import yaml

from lsst.daf import butler as dafButler
from lsst.ts import mtaos, salobj  # type: ignore[attr-defined]
from lsst.ts.ofc import OFCData
from lsst.ts.wep.utils import getModulePath as getModulePathWep
from lsst.ts.wep.utils import runProgram, writeCleanUpRepoCmd
from lsst.ts.xml import type_hints

# standard command timeout (sec)
STD_TIMEOUT = 60
SHORT_TIMEOUT = 5
TEST_CONFIG_DIR = Path(__file__).parents[1].joinpath("tests", "testData", "config")


class CscTestCase(salobj.BaseCscTestCase, unittest.IsolatedAsyncioTestCase):
    def basic_make_csc(
        self,
        initial_state: salobj.State | int,
        config_dir: str,
        simulation_mode: int | str,
    ) -> mtaos.MTAOS:
        return mtaos.MTAOS(config_dir=config_dir, simulation_mode=simulation_mode)

    @classmethod
    def setUpClass(cls) -> None:
        cls._randomize_topic_subname = True
        data_dir = mtaos.getModulePath().joinpath("tests", "tmp")
        isr_dir = data_dir.joinpath("input")

        cls.dataDir = data_dir
        cls.isrDir = isr_dir
        # Let the mtaos to set WEP based on this path variable
        os.environ["ISRDIRPATH"] = isr_dir.as_posix()

        cls.data_path = os.path.join(getModulePathWep(), "tests", "testData", "gen3TestRepo")
        cls.run_name = "run1"

        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.data_path)  # type: ignore
        registry = butler.registry

        # This is the expected index of the maximum zernike coefficient.
        cls.zernike_coefficient_maximum_expected = {1, 2}

        if cls.run_name in list(registry.queryCollections()):
            cleanUpCmd = writeCleanUpRepoCmd(cls.data_path, cls.run_name)
            runProgram(cleanUpCmd)

    def setUp(self) -> None:
        self.dataDir = mtaos.getModulePath().joinpath("tests", "tmp")
        self.isrDir = self.dataDir.joinpath("input")

        # Let the mtaos to set WEP based on this path variable
        os.environ["ISRDIRPATH"] = self.isrDir.as_posix()

    def tearDown(self) -> None:
        try:
            os.environ.pop("ISRDIRPATH")
        except KeyError:
            pass

        logFile = Path(mtaos.getLogDir()).joinpath("mtaos.log")
        if logFile.exists():
            logFile.unlink()

    @classmethod
    def tearDownClass(cls) -> None:
        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.data_path)  # type: ignore

        if cls.run_name in list(butler.registry.queryCollections()):
            runProgram(writeCleanUpRepoCmd(cls.data_path, cls.run_name))

    def _getCsc(self) -> mtaos.MTAOS:
        # This is instantiated after calling self.make_csc().
        return self.csc

    def _getRemote(self) -> salobj.Remote:
        # This is instantiated after calling self.make_csc().
        return self.remote

    def _make_local_model_for_dof_tests(self) -> mtaos.Model:
        ofc_data = OFCData("lsstfam")
        with mtaos.getModulePath().joinpath("tests", "testData", "state0inDof.yaml").open() as fp:
            ofc_data.dof_state0 = yaml.safe_load(fp)

        return mtaos.Model(instrument=ofc_data.name, data_path=None, ofc_data=ofc_data)

    async def testBinScript(self) -> None:
        cmdline_args = ["--log-to-file", "--log-level", "20"]
        await self.check_bin_script("MTAOS", 0, "run_mtaos", cmdline_args=cmdline_args)

    @patch("lsst.ts.mtaos.model.Butler")
    async def testStandardStateTransitions(self, MockButler: Mock) -> None:
        async with self.make_csc(initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0):
            enabled_commands = {
                "resetCorrection",
                "issueCorrection",
                "rejectCorrection",
                "selectSources",
                "preProcess",
                "runWEP",
                "runOFC",
                "addAberration",
                "interruptWEP",
                "startClosedLoop",
                "stopClosedLoop",
            }

            self.assert_software_versions(await self.remote.evt_softwareVersions.aget(timeout=STD_TIMEOUT))

            await self.check_standard_state_transitions(
                enabled_commands=enabled_commands,
                timeout=STD_TIMEOUT,
            )

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_configuration(self, MockButler: Mock) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            self.assertEqual(self.csc.summary_state, salobj.State.STANDBY)
            await self.assert_next_summary_state(salobj.State.STANDBY)

            invalid_files = glob.glob(os.path.join(TEST_CONFIG_DIR, "invalid_*.yaml"))
            bad_config_names = [os.path.basename(name) for name in invalid_files]
            bad_config_names.append("no_such_file.yaml")

            for bad_config_name in bad_config_names:
                with self.subTest(bad_config_name=bad_config_name):
                    with salobj.assertRaisesAckError():
                        await self.remote.cmd_start.set_start(
                            configurationOverride=bad_config_name, timeout=STD_TIMEOUT
                        )

            valid_files = glob.glob(os.path.join(TEST_CONFIG_DIR, "valid_*.yaml"))
            good_config_names = [os.path.basename(name) for name in valid_files]

            for good_config_name in good_config_names:
                await salobj.set_summary_state(self.remote, salobj.State.STANDBY)

                config_data = None
                with open(TEST_CONFIG_DIR / good_config_name) as fp:
                    config_data = yaml.safe_load(fp)

                await self.remote.cmd_start.set_start(
                    configurationOverride=good_config_name, timeout=STD_TIMEOUT
                )

                self.assertEqual(self.csc.visit_id_offset, config_data["visit_id_offset"])
                self.assertEqual(self.csc.model.instrument, config_data["instrument"])
                self.assertEqual(self.csc.model.run_name, config_data["run_name"])
                self.assertEqual(self.csc.model.collections, config_data["collections"])
                self.assertEqual(self.csc.m1m3_stress_limit, config_data["m1m3_stress_limit"])
                self.assertEqual(self.csc.m2_stress_limit, config_data["m2_stress_limit"])
                self.assertEqual(
                    self.csc.stress_scale_approach,
                    config_data["stress_scale_approach"],
                )
                self.assertEqual(
                    self.csc.stress_scale_factor,
                    config_data["stress_scale_factor"],
                )
                self.assertEqual(
                    self.csc.model.pipeline_instrument,
                    config_data["pipeline_instrument"],
                )
                self.assertEqual(
                    self.csc.model.pipeline_n_processes,
                    config_data["pipeline_n_processes"],
                )
                self.assertEqual(self.csc.model.zernike_table_name, config_data["zernike_table_name"])

    @patch("lsst.ts.mtaos.model.Butler")
    async def testResetCorrection(self, MockButler: Mock) -> None:
        async with self.make_csc(initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            remote = self._getRemote()
            dof_before_reset = await self.assert_next_sample(
                remote.evt_degreeOfFreedom, flush=False, timeout=SHORT_TIMEOUT
            )
            remote.evt_m2Correction.flush()
            remote.evt_m1m3Correction.flush()
            remote.evt_m2HexapodCorrection.flush()
            remote.evt_cameraHexapodCorrection.flush()

            await remote.cmd_resetCorrection.set_start(timeout=STD_TIMEOUT)
            dof = await self.assert_next_sample(
                remote.evt_degreeOfFreedom, flush=False, timeout=SHORT_TIMEOUT
            )

            dofAggr = dof.aggregatedDoF
            dofVisit = dof.visitDoF
            self.assertEqual(len(dofAggr), 50)
            self.assertEqual(len(dofVisit), 50)
            self.assertEqual(np.sum(np.abs(dofAggr)), np.sum(np.abs(dof_before_reset.aggregatedDoF)))
            self.assertEqual(np.sum(np.abs(dofVisit)), np.sum(np.abs(dof_before_reset.visitDoF)))

            await self.assert_next_sample(
                remote.evt_m2HexapodCorrection,
                flush=False,
                timeout=STD_TIMEOUT,
            )
            await self.assert_next_sample(
                remote.evt_cameraHexapodCorrection,
                flush=False,
                timeout=STD_TIMEOUT,
            )
            await self.assert_next_sample(
                remote.evt_m2Correction,
                flush=False,
                timeout=STD_TIMEOUT,
            )
            await self.assert_next_sample(
                remote.evt_m1m3Correction,
                flush=False,
                timeout=STD_TIMEOUT,
            )

    async def _checkCorrIsZero(self, remote: salobj.Remote) -> None:
        await self.assert_next_sample(
            remote.evt_m2HexapodCorrection,
            flush=False,
            timeout=STD_TIMEOUT,
            x=0,
            y=0,
            z=0,
            u=0,
            v=0,
            w=0,
        )

        await self.assert_next_sample(
            remote.evt_cameraHexapodCorrection,
            flush=False,
            timeout=STD_TIMEOUT,
            x=0,
            y=0,
            z=0,
            u=0,
            v=0,
            w=0,
        )

        corrM1M3 = await remote.evt_m1m3Correction.next(flush=False, timeout=STD_TIMEOUT)
        actForcesM1M3 = corrM1M3.zForces
        self.assertEqual(len(actForcesM1M3), 156)
        self.assertEqual(np.sum(np.abs(actForcesM1M3)), 0)

        corrM2 = await remote.evt_m2Correction.next(flush=False, timeout=STD_TIMEOUT)
        actForcesM2 = corrM2.zForces
        self.assertEqual(len(actForcesM2), 72)
        self.assertEqual(np.sum(np.abs(actForcesM2)), 0)

    @patch("lsst.ts.mtaos.model.Butler")
    async def testIssueCorrectionError(self, MockButler: Mock) -> None:
        async with self.make_csc(initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            remote = self._getRemote()

            with self.assertRaises(salobj.AckError):
                await remote.cmd_issueCorrection.set_start(timeout=30.0)

            dof = await self.assert_next_sample(
                remote.evt_rejectedDegreeOfFreedom, flush=False, timeout=SHORT_TIMEOUT
            )
            dofAggr = dof.aggregatedDoF
            dofVisit = dof.visitDoF
            self.assertEqual(len(dofAggr), 50)
            self.assertEqual(len(dofVisit), 50)
            self.assertEqual(np.sum(np.abs(dofVisit)), 0)

            await self.assert_next_sample(
                remote.evt_rejectedM2HexapodCorrection,
                flush=False,
                timeout=SHORT_TIMEOUT,
            )

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_handle_summary_state_fault_stores_previous_dofs_copy(self, MockButler: Mock) -> None:
        """Test that handling transition to FAULT stores a copy of the current
        DOFs in previous_dofs."""
        async with self.make_csc(initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0):
            csc = self.csc
            csc._model = self._make_local_model_for_dof_tests()
            aggregated_state = csc.model.get_dof_aggr().copy()
            aggregated_state += 0.5
            csc.model.set_dof_aggr(aggregated_state)

            with patch.object(type(csc), "summary_state", new_callable=PropertyMock) as summary_state:
                summary_state.return_value = salobj.State.FAULT
                await csc.handle_summary_state()

            np.testing.assert_array_equal(csc.previous_dofs, aggregated_state)
            self.assertIsNot(csc.previous_dofs, csc.model.get_dof_aggr())

            updated_dofs = aggregated_state + 10.0
            csc.model.set_dof_aggr(updated_dofs)
            np.testing.assert_array_equal(csc.previous_dofs, aggregated_state)

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_handle_summary_state_disabled_preserves_previous_dofs(self, MockButler: Mock) -> None:
        """Test that handling transition to DISABLED does not modify
        previous_dofs."""
        async with self.make_csc(initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0):
            csc = self.csc
            csc._model = self._make_local_model_for_dof_tests()
            previous_dofs = csc.model.get_dof_aggr().copy()
            previous_dofs += 0.5
            csc.previous_dofs = previous_dofs.copy()
            current_dofs = csc.model.get_dof_aggr().copy()
            current_dofs += 1.5
            csc.model.set_dof_aggr(current_dofs)

            with patch.object(type(csc), "summary_state", new_callable=PropertyMock) as summary_state:
                summary_state.return_value = salobj.State.DISABLED
                await csc.handle_summary_state()

            np.testing.assert_array_equal(csc.previous_dofs, previous_dofs)

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_execute_issue_correction_stores_previous_dofs(self, MockButler: Mock) -> None:
        """Test that executing issueCorrection stores a copy of the current
        DOFs in previous_dofs."""
        async with self.make_csc(initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0):
            csc = self.csc
            csc._model = self._make_local_model_for_dof_tests()
            updated_dofs = csc.model.get_dof_aggr().copy()
            updated_dofs += 0.5

            async def handle_corrections() -> None:
                csc.model.set_dof_aggr(updated_dofs)
                csc._store_previous_dofs()

            csc.handle_corrections = AsyncMock(side_effect=handle_corrections)
            csc.pubEvent_mirrorStresses = AsyncMock()

            await csc._execute_issue_correction()

            np.testing.assert_array_equal(csc.previous_dofs, updated_dofs)
            self.assertIsNot(csc.previous_dofs, csc.model.get_dof_aggr())

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_dof_restored_after_fault_recovery(self, MockButler: Mock) -> None:
        """Test DOFs are restored after FAULT → STANDBY → ENABLED cycle.

        This test verifies behavior (B): DOFs stored at the moment of FAULT
        are restored when returning to ENABLED.
        """
        async with self.make_csc(initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0):
            # Go to ENABLED state
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            csc = self.csc

            # Set known DOFs representing state at moment of FAULT
            original_dofs = csc.model.get_dof_aggr().copy()
            original_dofs += 0.5
            csc.model.set_dof_aggr(original_dofs)

            # Simulate FAULT (stores DOFs at moment of fault)
            with patch.object(type(csc), "summary_state", new_callable=PropertyMock) as summary_state:
                summary_state.return_value = salobj.State.FAULT
                await csc.handle_summary_state()

            # Verify DOFs were stored
            np.testing.assert_array_equal(csc.previous_dofs, original_dofs)

            # Recover: STANDBY → ENABLED (configure() restores DOFs)
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            # Verify DOFs were restored
            restored_dofs = csc.model.get_dof_aggr()
            np.testing.assert_array_equal(restored_dofs, original_dofs)

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_addAberration(self, MockButler: Mock) -> None:
        async with self.make_csc(initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            remote = self._getRemote()

            # Flush all events before command is sent
            remote.evt_degreeOfFreedom.flush()
            remote.evt_m2HexapodCorrection.flush()
            remote.evt_cameraHexapodCorrection.flush()
            remote.evt_m1m3Correction.flush()
            remote.evt_m2Correction.flush()

            # set control algorithm
            config = dict(name="lsstfam", filter_name="G", sensor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8])

            await remote.cmd_addAberration.set_start(
                wf=np.zeros(19), config=yaml.safe_dump(config), timeout=STD_TIMEOUT
            )

            await self.assert_next_sample(
                remote.evt_degreeOfFreedom,
                flush=False,
                timeout=SHORT_TIMEOUT,
            )

            await self.assert_next_sample(remote.evt_m2HexapodCorrection, flush=False, timeout=SHORT_TIMEOUT)
            await self.assert_next_sample(
                remote.evt_cameraHexapodCorrection, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(remote.evt_m1m3Correction, flush=False, timeout=SHORT_TIMEOUT)
            await self.assert_next_sample(remote.evt_m2Correction, flush=False, timeout=SHORT_TIMEOUT)

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_degree_of_freedom_pid_diagnostics(self, MockButler: Mock) -> None:
        async with self.make_csc(initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            remote = self._getRemote()
            remote.evt_degreeOfFreedom.flush()
            remote.evt_rejectedDegreeOfFreedom.flush()

            wavefront_errors = np.zeros(19)
            wavefront_errors[0] = 0.1
            config = dict(name="lsstfam", filter_name="G", sensor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            await remote.cmd_addAberration.set_start(
                wf=wavefront_errors, config=yaml.safe_dump(config), timeout=STD_TIMEOUT
            )

            dof = await self.assert_next_sample(
                remote.evt_degreeOfFreedom, flush=False, timeout=SHORT_TIMEOUT
            )
            pid_diagnostics = dict(
                zip(
                    (
                        "opticalState",
                        "integralError",
                        "integralErrorClipped",
                        "derivativeTerm",
                        "derivativeTermFiltered",
                    ),
                    self.csc.model.get_pid_diagnostics(),
                )
            )
            self.assertTrue(np.any(np.asarray(dof.opticalState) != 0))
            for name, value in pid_diagnostics.items():
                with self.subTest(event="degreeOfFreedom", field=name):
                    np.testing.assert_array_almost_equal(getattr(dof, name), value)

            await remote.cmd_rejectCorrection.start(timeout=STD_TIMEOUT)

            rejected_dof = await self.assert_next_sample(
                remote.evt_rejectedDegreeOfFreedom, flush=False, timeout=SHORT_TIMEOUT
            )
            for name, value in pid_diagnostics.items():
                with self.subTest(event="rejectedDegreeOfFreedom", field=name):
                    np.testing.assert_array_almost_equal(getattr(rejected_dof, name), value)

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_addAberration_with_config(self, MockButler: Mock) -> None:
        async with self.make_csc(initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            remote = self._getRemote()

            # Flush all events before command is sent
            remote.evt_degreeOfFreedom.flush()
            remote.evt_m2HexapodCorrection.flush()
            remote.evt_cameraHexapodCorrection.flush()
            remote.evt_m1m3Correction.flush()
            remote.evt_m2Correction.flush()

            # set control algorithm
            config = dict(name="lsstfam", xref="x0", sensor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8])

            await remote.cmd_addAberration.set_start(
                wf=np.zeros(19), config=yaml.safe_dump(config), timeout=STD_TIMEOUT
            )

            await self.assert_next_sample(
                remote.evt_degreeOfFreedom,
                flush=False,
                timeout=SHORT_TIMEOUT,
            )

            await self.assert_next_sample(remote.evt_m2HexapodCorrection, flush=False, timeout=SHORT_TIMEOUT)
            await self.assert_next_sample(
                remote.evt_cameraHexapodCorrection, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(remote.evt_m1m3Correction, flush=False, timeout=SHORT_TIMEOUT)
            await self.assert_next_sample(remote.evt_m2Correction, flush=False, timeout=SHORT_TIMEOUT)

            # Change alpha
            config = dict(
                alpha=(self.csc.model.ofc.ofc_data.alpha / 2).tolist(),
                sensor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            )

            await remote.cmd_addAberration.set_start(
                wf=np.zeros(19), config=yaml.safe_dump(config), timeout=STD_TIMEOUT
            )

            await self.assert_next_sample(
                remote.evt_degreeOfFreedom,
                flush=False,
                timeout=SHORT_TIMEOUT,
            )

            await self.assert_next_sample(remote.evt_m2HexapodCorrection, flush=False, timeout=SHORT_TIMEOUT)
            await self.assert_next_sample(
                remote.evt_cameraHexapodCorrection, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(remote.evt_m1m3Correction, flush=False, timeout=SHORT_TIMEOUT)
            await self.assert_next_sample(remote.evt_m2Correction, flush=False, timeout=SHORT_TIMEOUT)

            # Change alpha
            new_comp_dof_idx = dict(
                m2HexPos=np.zeros(5, dtype=bool).tolist(),
                camHexPos=np.ones(5, dtype=bool).tolist(),
                M1M3Bend=np.zeros(20, dtype=bool).tolist(),
                M2Bend=np.zeros(20, dtype=bool).tolist(),
            )
            updated_config = dict(comp_dof_idx=new_comp_dof_idx, sensor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8])

            await remote.cmd_addAberration.set_start(
                wf=np.zeros(19),
                config=yaml.safe_dump(updated_config),
                timeout=STD_TIMEOUT,
            )

            await self.assert_next_sample(
                remote.evt_degreeOfFreedom,
                flush=False,
                timeout=SHORT_TIMEOUT,
            )

            await self.assert_next_sample(remote.evt_m2HexapodCorrection, flush=False, timeout=SHORT_TIMEOUT)
            await self.assert_next_sample(
                remote.evt_cameraHexapodCorrection, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(remote.evt_m1m3Correction, flush=False, timeout=SHORT_TIMEOUT)
            await self.assert_next_sample(remote.evt_m2Correction, flush=False, timeout=SHORT_TIMEOUT)

    @patch("lsst.ts.mtaos.model.Butler")
    @pytest.mark.xfail(
        reason="This test fails if there is an issue with git lfs (OSW-2007).",
        raises=salobj.AckError,
    )
    @pytest.mark.csc_integtest
    async def test_run_wep_comcam(self, MockButler: Mock) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            ofc_data = OFCData("lsstfam")

            dof_state0 = yaml.safe_load(
                mtaos.getModulePath().joinpath("tests", "testData", "state0inDof.yaml").open().read()
            )
            ofc_data.dof_state0 = dof_state0

            self.csc._model = mtaos.Model(
                instrument=ofc_data.name,
                data_path=self.data_path,
                ofc_data=ofc_data,
                run_name=self.run_name,
                collections="refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic",
                pipeline_instrument=dict(comcam="lsst.obs.lsst.LsstCam"),
                data_instrument_name=dict(comcam="LSSTCam"),
                reference_detector=94,
            )

            remote = self._getRemote()
            self.remote.evt_wavefrontError.flush()
            self.remote.evt_wepDuration.flush()

            await remote.cmd_runWEP.set_start(
                visitId=4021123106001,
                extraId=4021123106002,
            )

            await self.assert_next_sample(
                self.remote.evt_wavefrontError,
                flush=False,
                timeout=STD_TIMEOUT,
            )
            await self.assert_next_sample(
                self.remote.evt_wepDuration,
                flush=False,
                timeout=STD_TIMEOUT,
            )

    @patch("lsst.ts.mtaos.model.Butler")
    @pytest.mark.csc_integtest
    async def test_run_wep_comcam_disable_after_execution_error(self, MockButler: Mock) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            ofc_data = OFCData("lsstfam")

            dof_state0 = yaml.safe_load(
                mtaos.getModulePath().joinpath("tests", "testData", "state0inDof.yaml").open().read()
            )
            ofc_data.dof_state0 = dof_state0

            self.csc._model = mtaos.Model(
                instrument=ofc_data.name,
                data_path=self.data_path,
                ofc_data=ofc_data,
                run_name=self.run_name,
                collections="refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic",
                pipeline_instrument=dict(comcam="lsst.obs.lsst.LsstCam"),
                data_instrument_name=dict(comcam="LSSTCam"),
                reference_detector=94,
            )

            remote = self._getRemote()
            self.remote.evt_wavefrontError.flush()
            self.remote.evt_wepDuration.flush()

            with self.assertRaises(salobj.AckError):
                await remote.cmd_runWEP.set_start(
                    visitId=4021123106003,  # Passing inexistent data.
                    extraId=4021123106004,
                )

            await salobj.set_summary_state(self.remote, salobj.State.STANDBY)

    @patch("lsst.ts.mtaos.model.Butler")
    @pytest.mark.xfail(
        reason="This test fails if there is an issue with git lfs (OSW-2007).",
        raises=salobj.AckError,
    )
    @pytest.mark.csc_integtest
    async def test_run_wep_lsst_cwfs(self, MockButler: Mock) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            ofc_data = OFCData("lsstfam")

            dof_state0 = yaml.safe_load(
                mtaos.getModulePath().joinpath("tests", "testData", "state0inDof.yaml").open().read()
            )
            ofc_data.dof_state0 = dof_state0

            self.csc._model = mtaos.Model(
                instrument=ofc_data.name,
                data_path=self.data_path,
                ofc_data=ofc_data,
                run_name=self.run_name,
                collections="refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic",
                reference_detector=94,
            )

            remote = self._getRemote()
            self.remote.evt_wavefrontError.flush()
            self.remote.evt_wepDuration.flush()

            await remote.cmd_runWEP.set_start(
                visitId=4021123106000,
            )

            await self.assert_next_sample(
                self.remote.evt_wavefrontError,
                flush=False,
                timeout=STD_TIMEOUT,
            )
            await self.assert_next_sample(
                self.remote.evt_wepDuration,
                flush=False,
                timeout=STD_TIMEOUT,
            )

    @patch("lsst.ts.mtaos.model.Butler")
    @pytest.mark.csc_integtest
    async def test_run_wep_lsst_cwfs_ocps(self, MockButler: Mock) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            ofc_data = OFCData("lsstfam")

            dof_state0 = yaml.safe_load(
                mtaos.getModulePath().joinpath("tests", "testData", "state0inDof.yaml").open().read()
            )
            ofc_data.dof_state0 = dof_state0

            self.csc._model = mtaos.Model(
                instrument=ofc_data.name,
                data_path=self.data_path,
                ofc_data=ofc_data,
                run_name=self.run_name,
                collections="refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic",
                reference_detector=94,
            )

            remote = self._getRemote()
            self.remote.evt_wavefrontError.flush()
            self.remote.evt_wepDuration.flush()

            with self.assertRaises(salobj.AckError):
                await remote.cmd_runWEP.set_start(
                    visitId=4021123106000,
                    extraId=None,
                    useOCPS=True,
                )

    @patch("lsst.ts.mtaos.model.Butler")
    @pytest.mark.csc_integtest
    async def test_interruptWEP(self, MockButler: Mock) -> None:
        async with self.make_csc(initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            ofc_data = OFCData("lsstfam")

            dof_state0 = yaml.safe_load(
                mtaos.getModulePath().joinpath("tests", "testData", "state0inDof.yaml").open().read()
            )
            ofc_data.dof_state0 = dof_state0

            self.csc._model = mtaos.Model(
                instrument=ofc_data.name,
                data_path=self.data_path,
                ofc_data=ofc_data,
                run_name=self.run_name,
                collections="refcats/gen2,LSSTCam/calib,LSSTCam/raw/all,LSSTCam/aos/intrinsic",
                pipeline_instrument=dict(comcam="lsst.obs.lsst.LsstCam"),
                data_instrument_name=dict(comcam="LSSTCam"),
                reference_detector=94,
            )

            remote = self._getRemote()
            self.remote.evt_wavefrontError.flush()
            self.remote.evt_wepDuration.flush()

            run_wep_task = asyncio.create_task(
                remote.cmd_runWEP.set_start(
                    visitId=4021123106001,
                    extraId=4021123106002,
                )
            )

            await asyncio.sleep(SHORT_TIMEOUT)

            await remote.cmd_interruptWEP.start(timeout=SHORT_TIMEOUT)

            with self.assertRaises(salobj.AckError):
                await run_wep_task

    def assert_software_versions(self, sofware_versions: type_hints.BaseMsgType) -> None:
        """Assert software versions payload is correctly populated.

        Raises
        ------
        AssertionError
            If software_versions does not match expected values.
        """
        # cscVersion matches csc version
        assert sofware_versions.cscVersion == mtaos.__version__

        # subsystemVersions must not be empty
        assert len(sofware_versions.subsystemVersions) > 0

        assert "ts_ofc" in sofware_versions.subsystemVersions
        assert "ts_wep" in sofware_versions.subsystemVersions
        assert "lsst_distrib" in sofware_versions.subsystemVersions

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_pointing_config_disabled_no_remote(self, MockButler: Mock) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await self.remote.cmd_start.set_start(
                configurationOverride="disable_pointing_correction.yaml",
                timeout=STD_TIMEOUT,
            )

            # Verify CSC did not initialize mtptg remote and flag is False
            self.assertFalse(self.csc.enable_pointing_correction)
            self.assertNotIn("mtptg", self.csc.remotes)

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_pointing_config_enabled_with_matrix(self, MockButler: Mock) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await self.remote.cmd_start.set_start(timeout=STD_TIMEOUT)

            # Verify CSC flag and remote presence
            self.assertTrue(self.csc.enable_pointing_correction)
            self.assertIn("mtptg", self.csc.remotes)
            # Sanity: compute returns two floats
            self.assertIsNotNone(self.csc.model)
            d = np.zeros(50)
            d[2] = 2500
            d[1] = -3000
            dx, dy, ca, ce = self.csc.model.compute_pointing_correction_offset(d)
            self.assertAlmostEqual(dx, 0.0, places=3)
            self.assertAlmostEqual(dy, 0.0, places=3)
            self.assertAlmostEqual(ca, -0.6, places=3)
            self.assertAlmostEqual(ce, 0.5, places=3)

    async def test_pointing_config_enabled_empty_matrix_rejected(self) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            with salobj.assertRaisesAckError():
                await self.remote.cmd_start.set_start(
                    configurationOverride="empty_pointing_correction_matrix.yaml",
                    timeout=STD_TIMEOUT,
                )

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_pointing_correction_reconfig_adds_mtptg(self, MockButler: Mock) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await self.remote.cmd_start.set_start(
                configurationOverride="disable_pointing_correction.yaml",
                timeout=STD_TIMEOUT,
            )
            self.assertFalse(self.csc.enable_pointing_correction)
            self.assertNotIn("mtptg", self.csc.remotes)

            await salobj.set_summary_state(self.remote, salobj.State.STANDBY)

            # Valid reconfiguration to enable pointing correction
            await self.remote.cmd_start.set_start(
                configurationOverride="enable_pointing_correction.yaml",
                timeout=STD_TIMEOUT,
            )
            self.assertTrue(self.csc.enable_pointing_correction)
            self.assertIn("mtptg", self.csc.remotes)

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_configure_filter_change_override_gains(self, MockButler: Mock) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            # Start with default config to test _init.yaml defaults.
            await self.remote.cmd_start.set_start(timeout=STD_TIMEOUT)

            # Default values from _init.yaml
            self.assertEqual(self.csc.filter_change_gain_n_iter, 2)
            self.assertEqual(self.csc.filter_change_gains, (0.75, 0.0, 0.0))

            # A config override can change the defaults.
            await salobj.set_summary_state(self.remote, salobj.State.STANDBY)
            await self.remote.cmd_start.set_start(
                configurationOverride="closed_loop_filter_change_gain.yaml",
                timeout=STD_TIMEOUT,
            )
            self.assertEqual(self.csc.filter_change_gain_n_iter, 3)
            # Mixed behavior in one config: 0.5 is a valid override value and
            # null means "do not override".
            self.assertEqual(self.csc.filter_change_gains, (1.0, 0.5, None))

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_execute_ofc_filter_change_gains_override_and_restore(self, MockButler: Mock) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await self.remote.cmd_start.set_start(
                configurationOverride="valid_comcam.yaml", timeout=STD_TIMEOUT
            )

            csc = self.csc
            controller = csc.model.ofc.controller
            controller.kp, controller.ki, controller.kd = 1.0, 2.0, 3.0

            # Stub out event publishers and OFC-data mutation to keep this unit
            # test focused on gain override/restore behavior.
            csc.pubEvent_degreeOfFreedom = AsyncMock()
            csc.pubEvent_mirrorStresses = AsyncMock()
            csc.pubEvent_m2HexapodCorrection = AsyncMock()
            csc.pubEvent_cameraHexapodCorrection = AsyncMock()
            csc.pubEvent_m1m3Correction = AsyncMock()
            csc.pubEvent_m2Correction = AsyncMock()
            csc.pubEvent_ofcDuration = AsyncMock()
            csc.model.set_ofc_data_values = AsyncMock(return_value={})

            csc.execution_times.setdefault("CALCULATE_CORRECTIONS", [])

            seen_gains: list[tuple[float, float, float]] = []

            def calculate_corrections_stub(*args: object, **kwargs: object) -> None:
                seen_gains.append((controller.kp, controller.ki, controller.kd))

            csc.model.calculate_corrections = calculate_corrections_stub

            csc.filter_change_gains = (0.9, 0.1, 0.1)
            await csc._execute_ofc(
                userGain=0.0,
                config="",
                timeout=STD_TIMEOUT,
                apply_filter_change_override=True,
                raise_on_large_defocus=False,
            )

            kp, ki, kd = seen_gains[-1]
            self.assertTrue(np.allclose(kp, 0.9))
            self.assertTrue(np.allclose(ki, 0.1))
            self.assertTrue(np.allclose(kd, 0.1))
            self.assertTrue(np.allclose(controller.kp, 1.0))
            self.assertTrue(np.allclose(controller.ki, 2.0))
            self.assertTrue(np.allclose(controller.kd, 3.0))

            def calculate_corrections_raises(*args: object, **kwargs: object) -> None:
                seen_gains.append((controller.kp, controller.ki, controller.kd))
                raise RuntimeError("boom")

            csc.model.calculate_corrections = calculate_corrections_raises

            csc.filter_change_gains = (0.5, 0.1, 0.0)
            with self.assertRaises(RuntimeError):
                await csc._execute_ofc(
                    userGain=0.0,
                    config="",
                    timeout=STD_TIMEOUT,
                    apply_filter_change_override=True,
                    raise_on_large_defocus=False,
                )

            kp, ki, kd = seen_gains[-1]
            self.assertTrue(np.allclose(kp, 0.5))
            self.assertTrue(np.allclose(ki, 0.1))
            self.assertTrue(np.allclose(kd, 0.0))
            self.assertTrue(np.allclose(controller.kp, 1.0))
            self.assertTrue(np.allclose(controller.ki, 2.0))
            self.assertTrue(np.allclose(controller.kd, 3.0))

            # Partial override: override kp and explicitly set ki=0.0;
            # keep kd unchanged.
            csc.model.calculate_corrections = calculate_corrections_stub
            csc.filter_change_gains = (1.0, 0.0, None)
            await csc._execute_ofc(
                userGain=0.2,
                config="",
                timeout=STD_TIMEOUT,
                apply_filter_change_override=True,
                raise_on_large_defocus=False,
            )
            kp, ki, kd = seen_gains[-1]
            self.assertTrue(np.allclose(kp, 1.0))
            self.assertTrue(np.allclose(ki, 0.0))
            self.assertTrue(np.allclose(kd, 3.0))
            self.assertTrue(np.allclose(controller.kp, 1.0))
            self.assertTrue(np.allclose(controller.ki, 2.0))
            self.assertTrue(np.allclose(controller.kd, 3.0))

    def _get_active_comp_dof_idx(self, ofc_data: OFCData) -> dict[str, np.ndarray]:
        """Get the active DOF mask for each component."""
        return {
            comp: ofc_data.dof_idx_mask[
                comp_data["startIdx"] : comp_data["startIdx"] + comp_data["idxLength"]
            ].copy()
            for comp, comp_data in ofc_data.comp_dof_idx.items()
        }

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_execute_ofc_skips_ofc_data_configuration(self, MockButler: Mock) -> None:
        """Verify runtime OFC parameters do not mutate ofc_data.

        Closed-loop corrections configure OFC data once at the start of the
        loop. Per-image runtime values, such as filter and rotation angle,
        should still be passed to the correction calculation without calling
        ``set_ofc_data_values``.
        """
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await self.remote.cmd_start.set_start(
                configurationOverride="valid_comcam.yaml", timeout=STD_TIMEOUT
            )

            csc = self.csc
            csc.pubEvent_degreeOfFreedom = AsyncMock()
            csc.pubEvent_mirrorStresses = AsyncMock()
            csc.pubEvent_m2HexapodCorrection = AsyncMock()
            csc.pubEvent_cameraHexapodCorrection = AsyncMock()
            csc.pubEvent_m1m3Correction = AsyncMock()
            csc.pubEvent_m2Correction = AsyncMock()
            csc.pubEvent_ofcDuration = AsyncMock()
            csc.model.set_ofc_data_values = AsyncMock()
            csc.model.calculate_corrections = Mock()
            csc.execution_times.setdefault("CALCULATE_CORRECTIONS", [])

            config = {
                "filter_name": "r",
                "rotation_angle": 12.3,
            }

            await csc._execute_ofc(
                userGain=0.0,
                config=yaml.safe_dump(config),
                timeout=STD_TIMEOUT,
                apply_filter_change_override=False,
                raise_on_large_defocus=False,
                configure_ofc_data=False,
            )

            csc.model.set_ofc_data_values.assert_not_called()
            csc.model.calculate_corrections.assert_called_once()
            call_kwargs = csc.model.calculate_corrections.call_args.kwargs
            self.assertEqual(call_kwargs["filter_name"], config["filter_name"])
            self.assertEqual(call_kwargs["rotation_angle"], config["rotation_angle"])

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_execute_ofc_restores_comp_dof_idx(self, MockButler: Mock) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await self.remote.cmd_start.set_start(
                configurationOverride="valid_comcam.yaml", timeout=STD_TIMEOUT
            )

            csc = self.csc
            csc.pubEvent_degreeOfFreedom = AsyncMock()
            csc.pubEvent_mirrorStresses = AsyncMock()
            csc.pubEvent_m2HexapodCorrection = AsyncMock()
            csc.pubEvent_cameraHexapodCorrection = AsyncMock()
            csc.pubEvent_m1m3Correction = AsyncMock()
            csc.pubEvent_m2Correction = AsyncMock()
            csc.pubEvent_ofcDuration = AsyncMock()
            csc.execution_times.setdefault("CALCULATE_CORRECTIONS", [])

            default_comp_dof_idx = {
                comp: mask.copy() for comp, mask in csc.model.ofc.ofc_data.default_comp_dof_idx.items()
            }
            comp_dof_idx = {comp: mask.copy() for comp, mask in default_comp_dof_idx.items()}
            comp_dof_idx["m2HexPos"][0] = False
            config = {
                "comp_dof_idx": {comp: mask.tolist() for comp, mask in comp_dof_idx.items()},
                "filter_name": "r",
                "rotation_angle": 12.3,
            }
            active_during_calculation = []
            calculation_kwargs = []

            def calculate_corrections_stub(*args: object, **kwargs: object) -> None:
                active_during_calculation.append(self._get_active_comp_dof_idx(csc.model.ofc.ofc_data))
                calculation_kwargs.append(kwargs)

            csc.model.calculate_corrections = calculate_corrections_stub

            await csc._execute_ofc(
                userGain=0.0,
                config=yaml.safe_dump(config),
                timeout=STD_TIMEOUT,
                apply_filter_change_override=False,
                raise_on_large_defocus=False,
            )

            for comp, mask in comp_dof_idx.items():
                self.assertTrue(np.array_equal(active_during_calculation[0][comp], mask))
            self.assertEqual(calculation_kwargs[0]["filter_name"], config["filter_name"])
            self.assertEqual(calculation_kwargs[0]["rotation_angle"], config["rotation_angle"])

            active_comp_dof_idx = self._get_active_comp_dof_idx(csc.model.ofc.ofc_data)
            for comp, mask in default_comp_dof_idx.items():
                self.assertTrue(np.array_equal(active_comp_dof_idx[comp], mask))

    @patch("lsst.ts.mtaos.model.Butler")
    async def test_ofc_data_configuration_restores_at_context_end(self, MockButler: Mock) -> None:
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await self.remote.cmd_start.set_start(
                configurationOverride="valid_comcam.yaml", timeout=STD_TIMEOUT
            )

            csc = self.csc
            default_comp_dof_idx = {
                comp: mask.copy() for comp, mask in csc.model.ofc.ofc_data.default_comp_dof_idx.items()
            }
            comp_dof_idx = {comp: mask.copy() for comp, mask in default_comp_dof_idx.items()}
            comp_dof_idx["m2HexPos"][0] = False
            config = {
                "comp_dof_idx": {comp: mask.tolist() for comp, mask in comp_dof_idx.items()},
            }

            async with csc._ofc_data_configuration(config):
                active_comp_dof_idx = self._get_active_comp_dof_idx(csc.model.ofc.ofc_data)
                for comp, mask in comp_dof_idx.items():
                    self.assertTrue(np.array_equal(active_comp_dof_idx[comp], mask))

            active_comp_dof_idx = self._get_active_comp_dof_idx(csc.model.ofc.ofc_data)
            for comp, mask in default_comp_dof_idx.items():
                self.assertTrue(np.array_equal(active_comp_dof_idx[comp], mask))


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
