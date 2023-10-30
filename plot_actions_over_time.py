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
import os
import sqlite3

from log import Log
from plot import Plot, PlotUtils
from utils import Utils

MODULE = "PlotRewardOverTime"

N_NODES = 5

LEGEND = [
    "Reject",
    "Cloud",
    "Worker \#1 (S = 0.2)",
    "Worker \#2 (S = 0.9)",
    "Worker \#3 (S = 0.7)",
    "Worker \#4 (S = 0.7)",
    "Worker \#5 (S = 0.6)",
    "Cluster \#2",
    "Cluster \#3",
    "Up Cluster"
]
# for i in range(N_NODES):
#    LEGEND.append(f"Node\#{i} (S={round(1.0 - 0.2 * i, 1)})")

DB_FILE_FOLDER = "20231029-100459"
DB_FILE = f"./_log/learning/D_SARSA/OTHER_CLUSTERS/{DB_FILE_FOLDER}/log.db"
# DB_FILE = f"./_log/no-learning/LEAST_LOADED_AWARE_CLOUD/{DB_FILE_FOLDER}/log.db"

db = sqlite3.connect(DB_FILE)
cur = db.cursor()

res = cur.execute("select max(generated_at) from jobs")
for line in res:
    simulation_time = math.ceil(line[0])

Log.mdebug(MODULE, f"simulation_time={simulation_time}")

# compute jobs per second
res = cur.execute(
    "select cast(generated_at as integer), count(*) from jobs where node_uid = 0 group by cast(generated_at as integer)")

jobs_per_second = {}
for line in res:
    jobs_per_second[line[0]] = line[1]

# compute how many actions per node in seconds
actions = []

num_of_jobs_per_t = {}
# num of jobs
num_d = {}
res = cur.execute(
    f"select cast(generated_at as integer), count(*) from jobs group by cast(generated_at as integer)")
for line in res:
    num_d[line[0]] = line[1]

# reject
reject_d = {}
res = cur.execute(
    f"select cast(generated_at as integer), count(*) from jobs where action = 0 group by cast(generated_at as integer)")
for line in res:
    reject_d[line[0]] = 100 * line[1] / num_d[line[0]]
actions.append(reject_d)

# cloud
cloud_d = {}
res = cur.execute(
    f"select cast(generated_at as integer), count(*) from jobs where action = 1 group by cast(generated_at as integer)")
for line in res:
    cloud_d[line[0]] = 100 * line[1] / num_d[line[0]]
actions.append(cloud_d)

# Other cluster
cluster_one={}
res = cur.execute(
    f"select cast(generated_at as integer), count(*) from jobs where action = {N_NODES + 3} group by cast(generated_at as integer)")
for line in res:
    cluster_one[line[0]] = 100 * line[1] / num_d[line[0]]
actions.append(cluster_one)

cluster_two={}
res = cur.execute(
    f"select cast(generated_at as integer), count(*) from jobs where action = {N_NODES + 4} group by cast(generated_at as integer)")
for line in res:
    cluster_two[line[0]] = 100 * line[1] / num_d[line[0]]
actions.append(cluster_two)


cluster_three={}
res = cur.execute(
    f"select cast(generated_at as integer), count(*) from jobs where action = {N_NODES + 5} group by cast(generated_at as integer)")
for line in res:
    cluster_three[line[0]] = 100 * line[1] / num_d[line[0]]
actions.append(cluster_three)

for node_i in range(2, N_NODES + 2):
    actions_d = {}
    res = cur.execute(
        f"select cast(generated_at as integer), count(*) from jobs where action = {node_i} group by cast(generated_at as integer)")

    for line in res:
        actions_d[line[0]] = 100 * line[1] / num_d[line[0]]
    actions.append(actions_d)




os.makedirs("./_plots", exist_ok=True)
figure_filename = f"./_plots/actions-over-time_{DB_FILE_FOLDER}_{Utils.current_time_string()}.pdf"

out_x = []
out_y = []

for i in range(len(actions)):
    out_x.append(actions[i].keys())
    out_y.append(actions[i].values())

PlotUtils.use_tex()
# Plot.multi_plot(out_x, out_y, "Time", "Percentage of actions", fullpath=figure_filename, legend=LEGEND)

# create the average
actions_avg_x = []
actions_avg_y = []
average_every_secs = 300

for action_d in actions:
    x = []
    y = []

    sum_v = 0
    elements = 0

    for i in range(simulation_time):
        if i in action_d.keys():
            sum_v += action_d[i]
        elements += 1

        if i % average_every_secs == 0:
            x.append(i)
            y.append(sum_v / elements)

            sum_v = 0
            elements = 0

    actions_avg_x.append(x)
    actions_avg_y.append(y)

print(len(actions_avg_x))
figure_filename = f"./_plots/actions-over-time-avg_{DB_FILE_FOLDER}_{Utils.current_time_string()}.pdf"
Plot.multi_plot(actions_avg_x, actions_avg_y, "Time (s)", "Percentage of actions (\%)", fullpath=figure_filename,
                legend=LEGEND, figh=3.7, legend_position='upper right')
