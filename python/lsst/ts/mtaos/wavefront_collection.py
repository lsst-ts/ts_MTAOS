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

__all__ = ["WavefrontCollection"]

import warnings
from collections import deque

import astropy.units as u
import numpy as np
from astropy.table import QTable


class WavefrontCollection(object):
    def __init__(self, maxLeng):
        """Collection of list of wavefront sensor data.

        Parameters
        ----------
        maxLeng : `int`
            Maximum length of collection.
        """

        # Collection of list of wavefront error data
        self._collectionData = deque(maxlen=int(maxLeng))

        # Collection of taken data of list of wavefront error data.
        # This is designed for the MtaosCsc to publish the event of wavefront
        # error. The published data will be in this collection.
        # This is a list of tuples with (sensor_id, np.ndarray)
        self._collectionDataTaken = dict()
        # Number of data taken from collectionData into collectionDataTaken
        self._numDataTaken = 0

    def getNumOfData(self):
        """Get the number of data.

        Returns
        -------
        `int`
            Number of data.
        """

        return len(self._collectionData)

    def getNumOfDataTaken(self):
        """Get the number of taken data.

        Returns
        -------
        `int`
            Number of taken data.
        """

        return self._numDataTaken

    def append(self, zernikes_data):
        """Add the list of wavefront error data to collection.

        Parameters
        ----------
        zernikes_data : `list` of `tuple` with [`int`, `astropy.table.QTable`]
        or `list` of `tuple` with [`int`, `np.ndarray`]
            List of wavefront error data. Each element contains tuple which the
            first elements specify the sensor id and the second is an astropy
            table with the zernike coeffients or an array with zernike
            coefficients.
        """

        self._collectionData.append(zernikes_data)

    def pop(self):
        """Pop the list of wavefront error data from collection.

        Returns
        -------
        listOfWfErr : `tuple` with [`int`, `np.ndarray`]
            List of wavefront error data.
        """
        try:
            data = self._collectionData.popleft()
            for sensor_id, zernikes_data in data:
                if isinstance(zernikes_data, QTable):
                    zk_indices = np.array(
                        [
                            int(col[1:])
                            for col in zernikes_data.colnames
                            if col.startswith("Z")
                        ]
                    )

                    z_columns = [
                        col for col in zernikes_data.colnames if col.startswith("Z")
                    ]
                    average_row = zernikes_data[zernikes_data["label"] == "average"][0]
                    zk_values = np.array(
                        [average_row[col].to(u.um).value for col in z_columns]
                    )

                    self._collectionDataTaken[sensor_id] = (zk_indices, zk_values)

                elif isinstance(zernikes_data, np.ndarray):
                    if sensor_id in self._collectionDataTaken:
                        self._collectionDataTaken[sensor_id] = np.vstack(
                            (self._collectionDataTaken[sensor_id], zernikes_data)
                        )
                    else:
                        self._collectionDataTaken[sensor_id] = np.array(
                            zernikes_data, ndmin=2
                        )
            self._numDataTaken += 1
        except IndexError:
            data = []

        return data

    def clear(self):
        """Clear the collection."""

        self._collectionData.clear()
        self._collectionDataTaken = dict()
        self._numDataTaken = 0

    def getListOfWavefrontErrorAvgInTakenData(self):
        """Get the list of average wavefront error in taken data.

        Returns
        -------
        wfe_avg : `dict`
            Dictionary with average wavefront errors for each sensor.

        Raises
        ------
        RuntimeError
            No data in the collection of taken data.
        """
        if not self._collectionDataTaken:
            warnings.warn(
                "No data in the collection of taken data. Returning empty wfe_avg.",
                UserWarning,
            )
            return dict()

        first_entry = next(iter(self._collectionDataTaken.values()))
        if isinstance(first_entry, tuple):
            wfe_avg = self._collectionDataTaken.copy()
        else:
            wfe_avg = dict(
                [
                    (
                        sensor_id,
                        (
                            np.arange(
                                4, len(self._collectionDataTaken[sensor_id][0]) + 4
                            ),
                            np.mean(
                                np.array(self._collectionDataTaken[sensor_id]), axis=0
                            ),
                        ),
                    )
                    for sensor_id in self._collectionDataTaken
                ]
            )

        return wfe_avg


if __name__ == "__main__":
    pass
