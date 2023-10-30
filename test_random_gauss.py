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

import random
import time

FPS = 100
SIGMA = 0.0013
NOT_OVER = False

cumulative_time = 0.0
total_in_second = 0
seconds_passed = 0

for i in range(10000):
    int_time = random.gauss(1/FPS, SIGMA)
    while int_time < 0 or (NOT_OVER and 1/int_time > FPS):
        int_time = random.gauss(1/FPS, SIGMA)

    print(f"{1/int_time}")

    cumulative_time += int_time
    total_in_second += 1
    time.sleep(int_time)

    if cumulative_time > 1.0:
        print(f"{seconds_passed}s: {total_in_second} jobs")

        cumulative_time = 0.0
        total_in_second = 1
        seconds_passed += 1
