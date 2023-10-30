#  simulator-2022-nca - Simpy simulator of online scheduling between edge nodes
#  Copyright (c) 2021 - 2022. Gabriele Proietti Mattia <pm.gabriele@outlook.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import math

import matplotlib.pyplot as plt
import numpy as np

from log import Log

MODULE = "TrafficModel"


class TrafficModel:

    def __init__(self, raw_path=None, max_x=1000, cycles=1, shift=0, parsed_x_limit=None, steady=False, steady_for=1000, steady_every=3000):
        self._raw_path = raw_path
        self._raw_data = None

        self._raw_data_x_max = 0.0
        self._raw_data_y_max = 0.0
        self._raw_data_x_max_forced = parsed_x_limit

        self._max_x = max_x
        self._cycles = cycles
        self._shift = shift

        # steady
        self._steady = steady
        self._steady_for = steady_for
        self._steady_every = steady_every

        self._parse()

    def _parse(self):
        self._raw_data = np.genfromtxt(self._raw_path, delimiter=",", dtype=float)

        self._raw_data_x_max = self._raw_data_x_max_forced if self._raw_data_x_max_forced is not None else max(self._raw_data[:, 0])
        self._raw_data_y_max = max(self._raw_data[:, 1])

        self._x_scale_factor = self._max_x / self._raw_data_x_max

        Log.minfo(MODULE, f"parsed raw file: x_max={self._raw_data_x_max}, y_max={self._raw_data_y_max}")

    def get_traffic_at(self, x):
        def normalize(x):
            return ((x + self._shift) / self._x_scale_factor * self._cycles) % self._raw_data_x_max

        if self._steady:
            x_scaled = x % (self._steady_every + self._steady_for)
            n_steady = math.floor(x / (self._steady_for + self._steady_every))

            if self._steady_every <= x_scaled < self._steady_every + self._steady_for:
                # Log.mdebug(MODULE, f"x={x}, x_scaled={x_scaled}, n_steady={n_steady}, normalized={self._steady_every * (1 + n_steady) + self._steady_for * n_steady}")

                return np.interp(
                    normalize(self._steady_every * (1 + n_steady) + self._steady_for * n_steady),
                    self._raw_data[:, 0],
                    self._raw_data[:, 1]
                )

        return np.interp(
            normalize(x),
            self._raw_data[:, 0],
            self._raw_data[:, 1]
        )


def test():
    t = TrafficModel("./traffic/namex/namex-traffic-daily-20210420.csv", cycles=1, max_x=10000, shift=0, steady=True, steady_every=2000, steady_for=2000)
    #t = TrafficModel("./traffic/city/data/traffic_node_0.csv", cycles=1, max_x=8000, shift=0, steady=False, steady_every=2000, steady_for=2000)

    x = []
    y = []

    for i in range(10000):
        x.append(i)
        y.append(t.get_traffic_at(i))

    plt.plot(x, y)
    plt.show()
    plt.close()

# test()
