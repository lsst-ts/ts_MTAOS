# This file is part of ts_MTAOS.
#
# Developed for Vera C. Rubin Observatory Telescope and Site Systems.
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

__all__ = [
    "CONFIG_SCHEMA",
    "TELESCOPE_DOF_SCHEMA",
    "WEP_HEADER_CONFIG",
    "ISR_CONFIG",
    "GENERATE_DONUT_CATALOG_CONFIG",
    "SCIENCE_SENSOR_PIPELINE_CONFIG",
    "CWFS_PIPELINE_CONFIG",
]

import yaml

CONFIG_SCHEMA = yaml.safe_load(
    """
$schema: http://json-schema.org/draft-07/schema#
$id: https://github.com/lsst-ts/ts_MTAOS/blob/master/python/lsst/ts/MTAOS/schema_config.py
# title must end with one or more spaces followed by the schema version, which
# must begin with "v"
title: MTAOS v5
description: Schema for MTAOS configuration files
type: object

properties:

  camera:
    description: Type of camera for wavefront estimation pipeline (WEP) to use
    type: string
    enum: [lsstCam, lsstFamCam, comcam]

  instrument:
    description: Type of instrument for optical feedback control (OFC) to use
    type: string

  data_path:
    description: Path to the data butler.
    type: string

  run_name:
    description: >-
      Which name to use when running the pipeline task. This defines
      the location where the data is written in the butler.
    type: string

  collections:
    description: Name of the collections where the data is written in the butler.
    type: string

  pipeline_instrument:
    description: >-
      A dictionary that maps the name of the instrument to the name used in
      the pipeline task.
      type: object
      properties:
        comcam:
          type: string
        lsstCam:
          type: string
        lsstFamCam:
          type: string
      additionalProperties: false

  data_instrument_name:
    description: >-
      A dictionary that maps the name of the instrument to the name used in
      the pipeline task.
      type: object
      properties:
        comcam:
          type: string
        lsstCam:
          type: string
        lsstFamCam:
          type: string
      additionalProperties: false

  pipeline_n_processes:
    description: Number of processes to use when running pipeline.
    type: integer

  zernike_table_name:
    description: Name of the table in the butler with zernike coefficients.
    type: string

  reference_detector:
    description: Which detector to use as a reference for determining the boresight information.
    type: integer

  visit_id_offset:
    description: >-
      Offset applied to the visit ID. TODO (DM-31365): Remove workaround to
      visitId being of type long in MTAOS runWEP command.
    type: integer
    minimum: 0

  wep_config:
    description: >-
      A yaml configuration file to use as default values for the wep.
    type: string

  use_ocps:
    description: >-
      Whether to use the OCS or not. If False, the OCS is not used.
    type: boolean
    default: true

  used_dofs:
    description: >-
      Which degrees of freedom to use in the MTAOS system.
    type: array
    items:
      type: integer
      minimum: 0
      maximum: 49
    minItems: 1
    maxItems: 50
    default: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  m1m3_stress_limit:
    description: >-
      Stress limit for M1M3 in psi.
    type: number

  m2_stress_limit:
    description: >-
      Stress limit for M2 in psi.
    type: number

  stress_scale_approach:
    description: >-
      Approach to scale the bending modes.
    type: string
    enum: [scale, truncate]

  stress_scale_factor:
    description: >-
      Factor to scale the bending modes when rss'ing
      the individual bending mode stresses.
    type: number

required:
  - camera
  - instrument
  - data_path
  - m1m3_stress_limit
  - m2_stress_limit
  - stress_scale_approach
  - stress_scale_factor

additionalProperties: false
"""
)

TELESCOPE_DOF_SCHEMA = yaml.safe_load(
    """
$schema: http://json-schema.org/draft-07/schema#
$id: https://github.com/lsst-ts/ts_MTAOS/blob/master/python/lsst/ts/MTAOS/schema_config.py
# title must end with one or more spaces followed by the schema version, which
# must begin with "v"
title: TelescopeDoF v1
description: Schema for MTAOS configuration files
type: object

definitions:
  hexapod:
    type: object
    properties:
      dX:
        description: Delta in X (um)
        type: number
        default: 0
      dY:
        description: Delta in Y (um)
        type: number
        default: 0
      dZ:
        description: Delta in Z (um)
        type: number
        default: 0
      rX:
        description: Rotation in X (arcsec)
        type: number
        default: 0
      rY:
        description: Rotation in Y (arcsec)
        type: number
        default: 0

  bendingModes:
    type: object
    properties:
      mode1:
        description: Bending mode 1 (um)
        type: number
        default: 0
      mode2:
        description: Bending mode 2 (um)
        type: number
        default: 0
      mode3:
        description: Bending mode 3 (um)
        type: number
        default: 0
      mode4:
        description: Bending mode 4 (um)
        type: number
        default: 0
      mode5:
        description: Bending mode 5 (um)
        type: number
        default: 0
      mode6:
        description: Bending mode 6 (um)
        type: number
        default: 0
      mode7:
        description: Bending mode 7 (um)
        type: number
        default: 0
      mode8:
        description: Bending mode 8 (um)
        type: number
        default: 0
      mode9:
        description: Bending mode 9 (um)
        type: number
        default: 0
      mode10:
        description: Bending mode 10 (um)
        type: number
        default: 0
      mode11:
        description: Bending mode 11 (um)
        type: number
        default: 0
      mode12:
        description: Bending mode 12 (um)
        type: number
        default: 0
      mode13:
        description: Bending mode 13 (um)
        type: number
        default: 0
      mode14:
        description: Bending mode 14 (um)
        type: number
        default: 0
      mode15:
        description: Bending mode 15 (um)
        type: number
        default: 0
      mode16:
        description: Bending mode 16 (um)
        type: number
        default: 0
      mode17:
        description: Bending mode 17 (um)
        type: number
        default: 0
      mode18:
        description: Bending mode 18 (um)
        type: number
        default: 0
      mode19:
        description: Bending mode 19 (um)
        type: number
        default: 0
      mode20:
        description: Bending mode 20 (um)
        type: number
        default: 0

properties:

  M2Hexapod:
    description: Initial M2 Hexapod DoF
    $ref: '#/definitions/hexapod'

  cameraHexapod:
    description: Initial camera Hexapod DoF
    $ref: '#/definitions/hexapod'

  M1M3Bending:
    description: Initial M1M3 bending modes
    $ref: '#/definitions/bendingModes'

  M2Bending:
    description: Initial M2 bending modes
    $ref: '#/definitions/bendingModes'

required:
  - M2Hexapod
  - cameraHexapod
  - M1M3Bending
  - M2Bending

additionalProperties: false
"""
)

WEP_HEADER_CONFIG = yaml.safe_load(
    """
$schema: http://json-schema.org/draft-07/schema#
$id: https://github.com/lsst-ts/ts_MTAOS/blob/master/python/lsst/ts/MTAOS/schema_config.py
# title must end with one or more spaces followed by the schema version, which
# must begin with "v"
type: object
additionalProperties: false
properties:

  description:
    type: string
    description: Description of this pipeline configuration.
    default: wep basic processing test pipeline

  instrument:
    type: string
    description: >-
      Specify the corresponding instrument for the data we will be using.
    default: lsst.obs.lsst.LsstComCam

  tasks:
    type: object
    default:
      isr:
        class: lsst.ip.isr.isrTask.IsrTask
    """
)

ISR_CONFIG = yaml.safe_load(
    """isr:
  type: object
  additionalProperties: false
  properties:
    class:
      type: string
      default: lsst.ip.isr.isrTask.IsrTask
    config:
      additionalProperties: false
      type: object
      properties:
        connections.outputExposure:
          type: string
          default: postISRCCD
        doBias:
          type: boolean
          default: False
        doVariance:
          type: boolean
          default: False
        doLinearize:
          type: boolean
          default: False
        doCrosstalk:
          type: boolean
          default: False
        doDefect:
          type: boolean
          default: False
        doNanMasking:
          type: boolean
          default: False
        doInterpolate:
          type: boolean
          default: False
        doBrighterFatter:
          type: boolean
          default: False
        doDark:
          type: boolean
          default: False
        doFlat:
          type: boolean
          default: False
        doApplyGains:
          type: boolean
          default: True
        doFringe:
          type: boolean
          default: False
        doOverscan:
          type: boolean
          default: True
  """
)

GENERATE_DONUT_CATALOG_CONFIG = yaml.safe_load(
    """generateDonutCatalogWcsTask:
  type: object
  additionalProperties: false
  properties:
    class:
      type: string
      default: >-
        lsst.ts.wep.task.generateDonutCatalogWcsTask.GenerateDonutCatalogWcsTask
    config:
      properties:
        filterName:
          type: string
  """
)

SCIENCE_SENSOR_PIPELINE_CONFIG = yaml.safe_load(
    """CutOutDonutsScienceSensorTask:
  type: object
  additionalProperties: false
  properties:
    class:
      type: string
      default: lsst.ts.wep.task.cutOutDonutsScienceSensorTask.CutOutDonutsScienceSensorTask
    config:
      type: object
      additionalProperties: false
      properties:
        donutStampSize:
          type: integer
          default: 160
        initialCutoutPadding:
          type: integer
          default: 40
calcZernikesTask:
  type: object
  additionalProperties: false
  properties:
    class:
      type: string
      default: lsst.ts.wep.task.calcZernikesTask.CalcZernikesTask
"""
)

CWFS_PIPELINE_CONFIG = yaml.safe_load(
    """cutOutDonutsCwfsTask:
  type: object
  additionalProperties: false
  properties:
    class:
      type: string
      default: lsst.ts.wep.task.cutOutDonutsCwfsTask.CutOutDonutsCwfsTask
    config:
      type: object
      additionalProperties: false
      properties:
        donutStampSize:
          type: integer
          default: 160
        initialCutoutPadding:
          type: integer
          default: 40
calcZernikesTask:
  type: object
  additionalProperties: false
  properties:
    class:
      type: string
      default: lsst.ts.wep.task.calcZernikesTask.CalcZernikesTask
"""
)
