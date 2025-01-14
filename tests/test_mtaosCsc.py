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

import asyncio
import glob
import os
import unittest
from pathlib import Path

import numpy as np
import pytest
import yaml
from lsst.daf import butler as dafButler
from lsst.ts import mtaos, salobj
from lsst.ts.ofc import OFCData
from lsst.ts.wep.utils import getModulePath as getModulePathWep
from lsst.ts.wep.utils import runProgram, writeCleanUpRepoCmd

# standard command timeout (sec)
STD_TIMEOUT = 60
SHORT_TIMEOUT = 5
TEST_CONFIG_DIR = Path(__file__).parents[1].joinpath("tests", "testData", "config")


class CscTestCase(salobj.BaseCscTestCase, unittest.IsolatedAsyncioTestCase):
    def basic_make_csc(self, initial_state, config_dir, simulation_mode):
        return mtaos.MTAOS(config_dir=config_dir, simulation_mode=simulation_mode)

    @classmethod
    def setUpClass(cls):
        cls._randomize_topic_subname = True
        cls.dataDir = mtaos.getModulePath().joinpath("tests", "tmp")
        cls.isrDir = cls.dataDir.joinpath("input")

        # Let the mtaos to set WEP based on this path variable
        os.environ["ISRDIRPATH"] = cls.isrDir.as_posix()

        cls.data_path = os.path.join(
            getModulePathWep(), "tests", "testData", "gen3TestRepo"
        )
        cls.run_name = "run1"

        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.data_path)
        registry = butler.registry

        # This is the expected index of the maximum zernike coefficient.
        cls.zernike_coefficient_maximum_expected = {1, 2}

        if cls.run_name in list(registry.queryCollections()):
            cleanUpCmd = writeCleanUpRepoCmd(cls.data_path, cls.run_name)
            runProgram(cleanUpCmd)

    def setUp(self):
        self.dataDir = mtaos.getModulePath().joinpath("tests", "tmp")
        self.isrDir = self.dataDir.joinpath("input")

        # Let the mtaos to set WEP based on this path variable
        os.environ["ISRDIRPATH"] = self.isrDir.as_posix()

    def tearDown(self):
        try:
            os.environ.pop("ISRDIRPATH")
        except KeyError:
            pass

        logFile = Path(mtaos.getLogDir()).joinpath("mtaos.log")
        if logFile.exists():
            logFile.unlink()

    @classmethod
    def tearDownClass(cls):
        # Check that run doesn't already exist due to previous improper cleanup
        butler = dafButler.Butler(cls.data_path)

        if cls.run_name in list(butler.registry.queryCollections()):
            runProgram(writeCleanUpRepoCmd(cls.data_path, cls.run_name))

    def _getCsc(self):
        # This is instantiated after calling self.make_csc().
        return self.csc

    def _getRemote(self):
        # This is instantiated after calling self.make_csc().
        return self.remote

    async def testBinScript(self):
        cmdline_args = ["--log-to-file", "--log-level", "20"]
        await self.check_bin_script("MTAOS", 0, "run_mtaos", cmdline_args=cmdline_args)

    async def testStandardStateTransitions(self):
        async with self.make_csc(
            initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0
        ):
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
            }

            self.assert_software_versions(
                await self.remote.evt_softwareVersions.aget(timeout=STD_TIMEOUT)
            )

            await self.check_standard_state_transitions(
                enabled_commands=enabled_commands,
                timeout=STD_TIMEOUT,
            )

    async def test_configuration(self):
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

                self.assertEqual(
                    self.csc.visit_id_offset, config_data["visit_id_offset"]
                )
                self.assertEqual(self.csc.model.instrument, config_data["instrument"])
                self.assertEqual(self.csc.model.run_name, config_data["run_name"])
                self.assertEqual(self.csc.model.collections, config_data["collections"])
                self.assertEqual(
                    self.csc.m1m3_stress_limit, config_data["m1m3_stress_limit"]
                )
                self.assertEqual(
                    self.csc.m2_stress_limit, config_data["m2_stress_limit"]
                )
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
                self.assertEqual(
                    self.csc.model.zernike_table_name, config_data["zernike_table_name"]
                )

    async def testResetCorrection(self):
        async with self.make_csc(
            initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0
        ):
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
            self.assertEqual(
                np.sum(np.abs(dofAggr)), np.sum(np.abs(dof_before_reset.aggregatedDoF))
            )
            self.assertEqual(
                np.sum(np.abs(dofVisit)), np.sum(np.abs(dof_before_reset.visitDoF))
            )

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

        corrM1M3 = await remote.evt_m1m3Correction.next(
            flush=False, timeout=STD_TIMEOUT
        )
        actForcesM1M3 = corrM1M3.zForces
        self.assertEqual(len(actForcesM1M3), 156)
        self.assertEqual(np.sum(np.abs(actForcesM1M3)), 0)

        corrM2 = await remote.evt_m2Correction.next(flush=False, timeout=STD_TIMEOUT)
        actForcesM2 = corrM2.zForces
        self.assertEqual(len(actForcesM2), 72)
        self.assertEqual(np.sum(np.abs(actForcesM2)), 0)

    async def testIssueCorrectionError(self):
        async with self.make_csc(
            initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0
        ):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            remote = self._getRemote()

            with self.assertRaises(salobj.AckError):
                await remote.cmd_issueCorrection.set_start(timeout=10.0)

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

    async def test_addAberration(self):
        async with self.make_csc(
            initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0
        ):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            remote = self._getRemote()

            # Flush all events before command is sent
            remote.evt_degreeOfFreedom.flush()
            remote.evt_m2HexapodCorrection.flush()
            remote.evt_cameraHexapodCorrection.flush()
            remote.evt_m1m3Correction.flush()
            remote.evt_m2Correction.flush()

            # set control algorithm
            config = dict(filter_name="G", sensor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8])

            await remote.cmd_addAberration.set_start(
                wf=np.zeros(19), config=yaml.safe_dump(config), timeout=STD_TIMEOUT
            )

            await self.assert_next_sample(
                remote.evt_degreeOfFreedom,
                flush=False,
                timeout=SHORT_TIMEOUT,
            )

            await self.assert_next_sample(
                remote.evt_m2HexapodCorrection, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(
                remote.evt_cameraHexapodCorrection, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(
                remote.evt_m1m3Correction, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(
                remote.evt_m2Correction, flush=False, timeout=SHORT_TIMEOUT
            )

    async def test_addAberration_with_config(self):
        async with self.make_csc(
            initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0
        ):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            remote = self._getRemote()

            # Flush all events before command is sent
            remote.evt_degreeOfFreedom.flush()
            remote.evt_m2HexapodCorrection.flush()
            remote.evt_cameraHexapodCorrection.flush()
            remote.evt_m1m3Correction.flush()
            remote.evt_m2Correction.flush()

            # set control algorithm
            config = dict(xref="x0", sensor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8])

            await remote.cmd_addAberration.set_start(
                wf=np.zeros(19), config=yaml.safe_dump(config), timeout=STD_TIMEOUT
            )

            await self.assert_next_sample(
                remote.evt_degreeOfFreedom,
                flush=False,
                timeout=SHORT_TIMEOUT,
            )

            await self.assert_next_sample(
                remote.evt_m2HexapodCorrection, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(
                remote.evt_cameraHexapodCorrection, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(
                remote.evt_m1m3Correction, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(
                remote.evt_m2Correction, flush=False, timeout=SHORT_TIMEOUT
            )

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

            await self.assert_next_sample(
                remote.evt_m2HexapodCorrection, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(
                remote.evt_cameraHexapodCorrection, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(
                remote.evt_m1m3Correction, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(
                remote.evt_m2Correction, flush=False, timeout=SHORT_TIMEOUT
            )

            # Change alpha
            new_comp_dof_idx = dict(
                m2HexPos=np.zeros(5, dtype=bool).tolist(),
                camHexPos=np.ones(5, dtype=bool).tolist(),
                M1M3Bend=np.zeros(20, dtype=bool).tolist(),
                M2Bend=np.zeros(20, dtype=bool).tolist(),
            )
            config = dict(
                comp_dof_idx=new_comp_dof_idx, sensor_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8]
            )

            await remote.cmd_addAberration.set_start(
                wf=np.zeros(19), config=yaml.safe_dump(config), timeout=STD_TIMEOUT
            )

            await self.assert_next_sample(
                remote.evt_degreeOfFreedom,
                flush=False,
                timeout=SHORT_TIMEOUT,
            )

            await self.assert_next_sample(
                remote.evt_m2HexapodCorrection, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(
                remote.evt_cameraHexapodCorrection, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(
                remote.evt_m1m3Correction, flush=False, timeout=SHORT_TIMEOUT
            )
            await self.assert_next_sample(
                remote.evt_m2Correction, flush=False, timeout=SHORT_TIMEOUT
            )

    @pytest.mark.csc_integtest
    async def test_run_wep_comcam(self):
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            ofc_data = OFCData("comcam")

            dof_state0 = yaml.safe_load(
                mtaos.getModulePath()
                .joinpath("tests", "testData", "state0inDof.yaml")
                .open()
                .read()
            )
            ofc_data.dof_state0 = dof_state0

            self.csc.model = mtaos.Model(
                instrument=ofc_data.name,
                data_path=self.data_path,
                ofc_data=ofc_data,
                run_name=self.run_name,
                collections="refcats/gen2,LSSTCam/calib,LSSTCam/raw/all",
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
                timeout=SHORT_TIMEOUT,
            )
            await self.assert_next_sample(
                self.remote.evt_wepDuration,
                flush=False,
                timeout=SHORT_TIMEOUT,
            )

    @pytest.mark.csc_integtest
    async def test_run_wep_comcam_disable_after_execution_error(self):
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            ofc_data = OFCData("comcam")

            dof_state0 = yaml.safe_load(
                mtaos.getModulePath()
                .joinpath("tests", "testData", "state0inDof.yaml")
                .open()
                .read()
            )
            ofc_data.dof_state0 = dof_state0

            self.csc.model = mtaos.Model(
                instrument=ofc_data.name,
                data_path=self.data_path,
                ofc_data=ofc_data,
                run_name=self.run_name,
                collections="refcats/gen2,LSSTCam/calib,LSSTCam/raw/all",
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

    @pytest.mark.csc_integtest
    async def test_run_wep_lsst_cwfs(self):
        async with self.make_csc(
            initial_state=salobj.State.STANDBY,
            config_dir=TEST_CONFIG_DIR,
            simulation_mode=0,
        ):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            ofc_data = OFCData("lsst")

            dof_state0 = yaml.safe_load(
                mtaos.getModulePath()
                .joinpath("tests", "testData", "state0inDof.yaml")
                .open()
                .read()
            )
            ofc_data.dof_state0 = dof_state0

            self.csc.model = mtaos.Model(
                instrument=ofc_data.name,
                data_path=self.data_path,
                ofc_data=ofc_data,
                run_name=self.run_name,
                collections="refcats/gen2,LSSTCam/calib,LSSTCam/raw/all",
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
                timeout=SHORT_TIMEOUT,
            )
            await self.assert_next_sample(
                self.remote.evt_wepDuration,
                flush=False,
                timeout=SHORT_TIMEOUT,
            )

    @pytest.mark.csc_integtest
    async def test_interruptWEP(self):
        async with self.make_csc(
            initial_state=salobj.State.STANDBY, config_dir=None, simulation_mode=0
        ):
            await salobj.set_summary_state(self.remote, salobj.State.ENABLED)

            ofc_data = OFCData("comcam")

            dof_state0 = yaml.safe_load(
                mtaos.getModulePath()
                .joinpath("tests", "testData", "state0inDof.yaml")
                .open()
                .read()
            )
            ofc_data.dof_state0 = dof_state0

            self.csc.model = mtaos.Model(
                instrument=ofc_data.name,
                data_path=self.data_path,
                ofc_data=ofc_data,
                run_name=self.run_name,
                collections="refcats/gen2,LSSTCam/calib,LSSTCam/raw/all",
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

    def assert_software_versions(self, sofware_versions) -> None:
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


if __name__ == "__main__":
    # Do the unit test
    unittest.main()
