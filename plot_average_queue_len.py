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

import math
import sqlite3

from plot import Plot, PlotUtils
from utils import Utils

N_NODES = 5

DB_FILE_FOLDER = "20231008-202835"
DB_FILE = f"./_log/learning/D_SARSA/OTHER_CLUSTERS/{DB_FILE_FOLDER}/log.db"

avg_every_seconds = 120

LEGEND = [f"Worker \#{i + 1}" for i in range(N_NODES)]

db = sqlite3.connect(DB_FILE)
cur = db.cursor()

simulation_time = 0
res = cur.execute("select max(generated_at) from jobs")
for line in res:
    simulation_time = math.ceil(line[0])

avg_loads = []
avg_loads_x = [0]
current_sec = 0
current_sums = [0 for i in range(N_NODES)]
current_jobs = 0
res = cur.execute("select state_snapshot, generated_at from jobs order by generated_at")
for line in res:
    sec = int(line[1])

    if sec != current_sec:
        print(f"{sec} != {current_sec}")
        avgs = [v / current_jobs for v in current_sums]
        avg_loads.append(avgs)

        current_sec = sec
        current_sums = [0 for i in range(N_NODES)]
        current_jobs = 0
        avg_loads_x.append(sec)
        # print(avg_loads)

    current_jobs += 1

    loads = [0 for i in range(N_NODES)]
    for i, c in enumerate(line[0]):
        if i == 0 or i == N_NODES * 2 + 1:
            continue

        loads[(i - 1) % N_NODES] += int(c)

    for i, v in enumerate(loads):
        current_sums[i] += v

    print(f"s={sec}, loads={loads}")

print(f"len(avg_loads)={len(avg_loads)}")

avg_loads_y = [[0] for i in range(N_NODES)]
avg_loads_sums = [0 for i in range(N_NODES)]
avg_loads_x_final = [0]

for i, loads in enumerate(avg_loads):
    # print(loads)
    if i != 0 and i % avg_every_seconds == 0:
        for j, v in enumerate(avg_loads_sums):
            avg_loads_y[j].append(v / avg_every_seconds)
        avg_loads_sums = [0 for i in range(N_NODES)]
        avg_loads_x_final.append(i)

    for j, v in enumerate(loads):
        avg_loads_sums[j] += v

print(len(avg_loads_y))
print(len([avg_loads_x for _ in range(N_NODES)]))

PlotUtils.use_tex()

figure_filename = f"./_plots/average-queue-len_{DB_FILE_FOLDER}_{Utils.current_time_string()}.pdf"
Plot.multi_plot([avg_loads_x_final for _ in range(N_NODES)], avg_loads_y, "Time (s)", "Queue Length (\%)",
                fullpath=figure_filename,
                legend=LEGEND)
