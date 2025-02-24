===============
Version History
===============

v0.17.0
-------

* Added new closed loop task commands `do_startClosedLoop` and `do_stopClosedLoop` in `mtaos.py``.
v0.16.1
-------

* Deprecate `annularZernikeCoeff` topic from WavefrontError events.

v0.16.0
-------

* Introduced a minimum threshold for applying active optics corrections.
  Forces are now only applied if they exceed a defined threshold in both `issue_m2_correction` and `issue_m1m3_correction` methods.

* Updated `do_runWEP` and `do_runOFC` methods in `mtaos.py` to improve handling of wavefront error collections, preventing reprocessing of already processed data and clearing wavefront error collections after calculating corrections.

* Added support for OCPS (Operational Corrections Processing System) in the `do_runWEP` method.

* Updated the CSC and unit tests to use the new WEP 11.50 Zernikes Astropy table outputs.

* Added functionality to publish corrections and stresses via `offsetDOF` and `runOFC` commands.

* Introduced `userGain` as an optional parameter in MTAOS for fine-tuning adjustments.

* Enhanced logging throughout `model.py` to improve debugging and traceability of corrections.

* Updated `model.py` to allow sparse Zernike coefficients, increasing compatibility and flexibility.

* Added functions and unit tests to enforce safety limits for bending modes.

* Fixed several issues in `model.py`, including:

  - Correcting the calculation of offsets and forces in `_calculate_corrections`.
  - Fixing the `set_ofc_data_values` method to handle `comp_dof_idx` and data value copies correctly.
  - Resolved a bug in `_poll_butler_outputs` that caused an infinite loop, blocking the async event loop.

* Corrected index handling in `calculate_corrections` and addressed residue handling in `add_correction` unit tests.

* Ensured the `offset_dof` calculations conform to the proper correction definitions by reversing the sign of the offset.

* Fixed visit ID and extra ID handling in Butler queries.

* Updated unit tests for MTAOS CSC to align with changes in the CSC, OFC, and WEP updates.

* Added callback methods for simulator tests to emulate reset offset commands for M1M3 and M2 mirrors.

* Improved unit tests for new bending mode safety limits and the latest `ts_ofc` v4.0 values.

* Reset history when updating `comp_dof_idx` in OFC to ensure consistency.

* Improved the pooling loop in `_poll_butler_outputs` for better async behavior.

* Adjusted integration tests in the CI script for optimization and focus on critical testing paths.

* Update to use version 11.5.0 of ts_wep.

* Publish and use sparse zernikes.

* Add OCPS option for running WEP.
  
* Publish mirror stresses when using `runOFC` command.

* Publish corrections when using `offsetDOF` command.

* Add bending mode safety limits to prevent corrections from exceeding the mirror stress limits.

* Add unit tests for the new safety limits.

* Add `pubEvent_mirrorStresses` method to publish mirror stresses.

v0.15.0
-------

* Update to use version 3.2.0 of ts_ofc

v0.14.0
-------

* In ``mtaos.py``, implement the ``resetOffsetOFC`` command.

* In ``tests/test_mtaosCscWithSimulators.py``, update test_addAberration_issueCorrection_xref_x0 to flush the degreeOfFreedom event before running the test.

  This is necessary because now the CSC publishes the state once it goes to enabled and the test needs to ignore that initial state published.

* In ``tests/test_mtaosCscWithSimulators.py`` add unit test for the new offsetDOF command implementation.

* In ``mtaos.py``, add end_enable method and publish DoF state.

* In ``mtaos.py``, implement offsetDOF command.

* In ``model.py``, add method to offset the degrees of freedom.

  This allows us to add offsets to M1M3 and M2 bending modes as well as rigid body motions of the hexapods.

v0.13.3
-------

* Update Jenkinsfile to checkout the work branches for ts_wep.

* Update lint github action to pin python 3.11.

* In mtaos, update do_runOFC to allow users to pass in configuration.

* In model.py, update call to query datasets from the butler to retrieve the wavefront errors.

* Update unit tests to conform with latest changes in wep.

* In config_schema, remove configuration option from cutout pipeline.

* In ``utility.py``, mark ``getCamType`` as deprecated.

* In mtaos, pass data instrument name to the model class if it is defined in the configuration.

* In config_schema, add option to override the data instrument name.

v0.13.2
-------

* Add SConstruct file to allow building package with scons.

v0.13.1
-------

* Update ``tests/test_mtaosCsc.py`` to work with the kafka version of salobj.
* Update to work with ``ts_wep>=7``.

v0.13.0
-------

* Remove compatibility with xml<19.
* Update to ts-pre-commit-config 0.6.

v0.12.2
-------

* Add stubs for the new commands introduced in the CSC in xml 19.
  For now only add backward compatibility.

v0.12.1
-------

* Add support for ts-pre-commit-config.
* Update package setup files.
* Add git workflows to check version history is updated and linting.
* Run isort.

v0.12.0
-------

* Updates to work with ts_wep 6.
* Update Jenkinsfile to remove root workaround.

v0.11.3
-------

* In ``Model._generate_pipetask_command`` stop adding refcats to the collections.
* Update unit tests to work with latest version of ``ts_wep``.

v0.11.2
-------

* In `Model` class:

  * Add new `define_visits` coroutine that executes `utility.define_visits` in a process pool.
    The method is called in `_start_wep_process` before running the pipeline task.
    This is required by the current version of the pipeline task to process more than one exposure at a time.

* Add utility method to define visits.

* Update executable script: bin/run_mtaos.py -> bin/run_mtaos.

* Add .hypothesis/ to gitignore and expand ignore to all .log files.

* Rename package ``lsst.ts.MTAOS`` -> ``lsst.ts.mtaos``.

* In CI Jenkinsfile, enable abort previous build.

v0.11.1
-------

* Replace reference to MTHexapodID -> salIndex, for compatibility with salobj >7.1.
* Update Jenkinsfile to replace HOME -> WHOME.

v0.11.0
-------

* Upgrade CSC to work with salobj 7/xml 11.

v0.10.2
-------

* Fix bug in `begin_disable` that would prevent CSC from going out of ENABLED if last time`runWEP` execution failed.
* Update `Model.process_lsstcam_corner_wfs` to restrict processing to corner wavefront sensor detectors.
  Without this additional restriction the pipeline task would process (with isr, source selection, etcs) all the detectors, taking a considerable ammount of unnecessary compute and time to complete.
* Add `get_formatted_corner_wavefront_sensors_ids` utility method to generate a comma-separated string with the ids of the corner wavefront sensors for LSSTCam.

v0.10.1
-------

* Fill `softwareVersions.subsystemVersions` event attribute with information about ts_ofc, ts_wep and lsst_distrib packages.

v0.10.0
-------

* In Jenkinsfile, separate running tests marked as integtest and csc_integtest from the other unit tests. 
  Run non-marked tests first and, if successful, run integtest and csc_integtest respectively.
  The integration tests take quite some time and resources to execute so if a unit test fail we should not run those.
* In `tests/test_mtaosCsc.py` add test_run_wep_lsst_cwfs (annotated as `csc_integtest`) to test processing corner wavefront sensor.
* In `Model.run_wep` enable `process_lsstcam_corner_wfs`.
* Rename test test_runWEP -> test_run_wep_comcam
* Add integration tests for `Model.process_lsstcam_corner_wfs`.
* In `Model` add `process_lsstcam_corner_wfs` method to process LSSTCam corner wavefront sensor data.
* Move `process_comcam` tests from `tests/test_model.py` to `tests/wep_integration/test_comcam.py`. 
  Test case is now decorated with `integtest` to allow us to differentiate them from the other tests.
* In test_model, convert `TestModel` to an `unittest.IsolatedAsyncioTestCase` and merge `test_log_stream` into it. 
  Remove `TestAsyncModel`, the `process_*` tests will be moved into their own test module.
* In test_mtaosCsc, decorate tests involving WEP command with `csc_integtest` to allow them to be differentiated from other tests.
* In test_utility, reduce sleep time to speed up `timeit` test.

v0.9.0
------

* Add unit tests for `interruptWEP` command.
* Add xml 10/11 backward compatible command `interruptWEP`.
  The command won't be available for xml 10, but CSC will continue to work and automatically support when it is released.
* Add unit test for `Model.process_comcam` when pipeline task fails to execute.
* Add unit test for `Model.log_stream`.
* Add `support_interrupt_wep_cmd` utility method to support backward compatibility between xml 10 and xml 11.
* In `Model` refactor `log_stream` to handle `eof` condition.
* Add mechanism no interrupt an execution of the wep process.
* Update MTAOS to work with latest version of wep.

v0.8.0
------

* Add new (backward compatible) CSC configuration parameter `wep_config`, which allows users to specify a default configuration override for the CSC to use in the `runWep` command.
* Reorganize import statements in test_model.py unit test.
* Add unit tests for `Model.generate_wep_configuration`.
* In `Model` class: 
  * Add `expand_wep_configuration` method that will get a dictionary and a visit_info object and expand it such that it contains information for the `generateDonutCatalogOnlineTask` pipeline task.
  * Add `_get_visit_info` method to encapsulate usage of butler to retrieve image information. 
    This allows us wrap the method and provide better unit testing for the `Model.generate_wep_configuration` method.
  * Reformat docstrings to fit pep8 standards.

v0.7.8
------

* In `Model`, asynchronously log output of pipeline task.
* In `MTAOS.do_runWEP`, implement mechanism to differentiate wep runs using private identity (who sent the command?) and the send timestamp.
* In `MTAOS.do_runWEP`, fix use of `safe_dump` to `safe_load`, to convert input configuration string into python object.
* In `Model`, add interface to create different run names for each time MTAOS is processing data.
* In `Model`, raise an exception if the pipeline process fails.
  This causes the command to be rejected as failed, which is the behavior we want.

v0.7.7
------

* Update phosim_utils branch to main instead of master in CI job.

v0.7.6
------
* Update name of `ts_wep` task in `config_schema.py` from `EstimateZernikesFamTask` to `EstimateZernikesScienceSensorTask`.

v0.7.5
------

* Fix publishing Degrees of Freedom event when `issueCorrection` fails.

v0.7.4
------

* Update Jenkinsfile to notify gate keeper (tribeiro) on slack when build suffers a regression and when it is fixed.
* In `test_model`, update `test_process_comcam` to check the shape of the return arrays and the index of the maximum zernike coefficient instead of the values themselves.

v0.7.3
------

* Add visit_id_offset to configuration schema.
* Add visit_id_offset CSC configuration parameter to work around type of visitId being a long in runWEP and preProcess commands.
* Add unit tests for CSC configuration.
* Fix publishing wavefront errors.
* Fix gain feature in model.
* Fix pubTel_ofcDuration and pubTel_wepDuration methods in CSC. Rename to ``pubEvent_*`` and fix publishing of event topic instead of telemetry.
* In `rejectCorrection` publish degrees of freedom and corrections after rejecting correction.
* Fix setting user gain in model class.
* Deprecate the use of userGain in runOFC. It will now use the yaml configuration payload.
* In Model class use default ofc gain when initializing the class.
* Publish wepDuration at the end of runWEP.

v0.7.2
------

* Support the setting of **xref**.
* Add LSSTCam/calib to collections path in test Gen3 pipelines and fix the syntax of butler ``get()``.

v0.7.1
------

* Fix unit tests for reversed intra/extra image selection.

v0.7.0
------

* Implement ``runWEP`` command.
  The current implementation is designed to work for ComCam intra/extra data.
  It is also limited in a way that we cannot provide the target ahead of time for the pipeline task to select the sources.
* Add user-guide documentation on using ``runWEP``.
* Update UML class diagram.
* Enable pytest-black in unit tests.
* Fix bugs reported by Bo when trying to set ofc values in addAberration.
* Update model unit tests for fixed intra/extra definition.

v0.6.0
------

* In Jenkinsfile, run pytest in the entire package instead of only the `tests/` folder, to capture pep8 and black violations in the entire repo.
* Refactor module names to the current telescope and site standards (lower_camel_case).
* Refactor additional parts of the code to be compliant with the current style guide.
* Implement new version of OFC.
* In CSC:
  * Refactor log-to-file interface.
  * In `addAberration` command:
    * Stop issuing corrections. Users need to send a `issueAberration` for the aberrations to be applied.
    * Implement `config` feature, to allow users to customize ofc behavior.
    * Add some unit tests for `addAberration` config feature.
* Update tests/Sconscript to allow running scons with licensed version of OpenSplice.

v0.5.6
------

* Fixed a trailing space.

v0.5.5
------

* Fixed a too long comment line.

v0.5.4
------

* Reformat code using black 20.

v0.5.3
------

* Implement addAberration command.
* Remove `asynctest` and use `unittest.IsolatedAsyncioTestCase` instead.
* Fix version history.
* Minor documentation updates.

v0.5.2
------

* Refactor of the Model class to prepare it for integration with wep pipeline task.
* Modernize naming conventions in Model class and remove unused methods.
* Chance how execution time is calculated to use a decorator that stored the information in a dictionary and put that logic on the CSC instead.
* Remove simulation mode and ModelSim
* Implement new salobj configuration schema, replacing schema yaml file by string in a python module.
* Add support to publish CSC version.
* Update docs configuration.

v0.5.1
------

* Fix reference to undefined name `issue_corrections_tasks` -> `issued_corrections`.

v0.5.0
------

* Update MTAOS CSC to reflect new xml interface discussed in tstn-026.

v0.4.5
-------------
* Use the latest **ts_wep** that removes the dependency of ``sims`` package.
* Update the M2 interface based on the **ts_xml** v7.0.0.

v0.4.4
-------------
* Use the ``sims_w_2020_42``.
* Use the **ts_salobj** v6.0.3.
* Remove the deprecated functions for the new version of **ts_salobj**.
* Update the **user-guide.rst** for the use of CSC.

v0.4.3
-------------
* Update the M2 interface based on the **ts_xml** v6.1.0.
* Do some minor fixes.
* Update the test cases of CSC.
* Reformat the documents to improve the readibility.
* Use the ``sims_w_2020_29``.

v0.4.2
-------------
* Reformat the **rst** documents to follow the standard.
* Add the user manual.
* Publish the document to `MTAOS document <https://ts-mtaos.lsst.io>`_.

v0.4.1
-------------
* Reformat the code by ``black``.
* Add the ``black`` check to ``.githooks``.
* Ignore ``flake8`` check of E203 ans W503 for the ``black``.

v0.4.0
-------------
* Configure the ``state0`` in degree of freedom (DOF) from MTAOS files.
* Use the scientific pipeline ``w_2020_20``.

v0.3.9
-------------
* Add the **CollOfListOfWfErr** class to support the multiple exposures in a single visit.
* Use the scientific pipeline ``w_2020_15``.

v0.3.8
-------------
* Adapt to **ts_xml** v5.0.0.
* Add the logs directory.
* Support the change of debug level of log files.
* Use the **CscTestCase** from **ts_salobj** for CSC test.
* Remove the ``bin.src`` directory.
* Remove the dependency of **version.py**.

v0.3.7
-------------
* Adapt to **ts_xml** v4.7.0.

v0.3.6
-------------
* Use ``calcTime`` instead of ``duration`` and ``simulation_mode`` instead of ``initial_simulation_mode``.

v0.3.5
-------------
* Restrict some commands can only be executed in the **Enabled** state.

v0.3.4
-------------
* Support the log file for debug.

v0.3.3
-------------
* Support the configurable CSC and simulation mode.

v0.3.2
-------------
* Add the **Model** class and related test cases.

v0.3.1
-------------
* Workaround the Jenkins permission in **Jenkinsfile**.

v0.3.0
-------------
* Integrate with the PhoSim with the scientific pipeline tag: ``sims_w_2019_20``.
* Add the **Jenkinsfile**.
* Update the documentation.

v0.2.0
-------------
* Integrate with **ts_wep** and **ts_ofc**.

v0.1.0
-------------
* Initial version of **ts_MTAOS**.
