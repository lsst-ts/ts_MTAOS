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

__all__ = ["CONFIG_SCHEMA", "TELESCOPE_DOF_SCHEMA"]

import yaml

CONFIG_SCHEMA = yaml.safe_load(
    """
$schema: http://json-schema.org/draft-07/schema#
$id: https://github.com/lsst-ts/ts_MTAOS/blob/master/python/lsst/ts/MTAOS/schema_config.py
# title must end with one or more spaces followed by the schema version, which
# must begin with "v"
title: MTAOS v1
description: Schema for MTAOS configuration files
type: object

properties:

  camera:
    description: Type of camera for wavefront estimation pipeline (WEP) to use
    type: string
    enum: [lsstCam, lsstFamCam, comcam]
    default: comcam

  instrument:
    description: Type of instrument for optical feedback control (OFC) to use
    type: string
    enum: [lsst, comcam, sh, cmos]
    default: comcam

  defaultIsrDir:
    description: >
      Default instrument signature removal (ISR) directory.
      This setting will be override by the 'ISRDIRPATH' path variable.
    type: string
    default: /home/lsst/input

  defaultSkyFilePath:
    description: >
      Default sky file path relative to the root of module.
      This is for the test only.
    type: string
    default: tests/testData/phosimOutput/realComCam/skyComCamInfo.txt

required:
  - camera
  - instrument

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