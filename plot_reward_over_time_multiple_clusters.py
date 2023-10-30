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
from plot import PlotUtils, Plot
from utils import Utils
from utils_plot import UtilsPlot

MODULE = "PlotRewardOverTime"

DB_FILE_FOLDER = "20231023-184124"
DB_FILE = f"./_log/learning/D_SARSA/OTHER_CLUSTERS/{DB_FILE_FOLDER}/log.db"

db = sqlite3.connect(DB_FILE)
cur = db.cursor()

res = cur.execute("select max(generated_at) from jobs")
for line in res:
    simulation_time = math.ceil(line[0])

Log.mdebug(MODULE, f"simulation_time={simulation_time}")

average_every_secs = 60

x_rewards_0, y_rewards_0 = UtilsPlot.plot_data_reward_over_time(0, DB_FILE, average_every_secs=average_every_secs)
x_rewards_1, y_rewards_1 = UtilsPlot.plot_data_reward_over_time(6, DB_FILE, average_every_secs=average_every_secs)
x_rewards_2, y_rewards_2 = UtilsPlot.plot_data_reward_over_time(9, DB_FILE, average_every_secs=average_every_secs)

# print(x_rewards)
# print(y_rewards)

x_eps = []
y_eps = []

# eps
res = cur.execute(
    f"select cast(generated_at as integer), avg(eps) from jobs where node_uid = 0 group by cast(generated_at as integer)")
sum_reward = 0.0
added = 0
for line in res:
    t = line[0]
    reward = line[1]

    sum_reward += reward
    added += 1

    if t % average_every_secs == 0 and t > 0:
        # print(f"t={t}, added={added}, avg={sum_reward / added}")
        x_eps.append(t)
        y_eps.append(sum_reward / added)
        added = 0
        sum_reward = 0.0

os.makedirs("./_plots", exist_ok=True)
figure_filename = f"./_plots/reward-over-time-multiple-clusters_{DB_FILE_FOLDER}_{Utils.current_time_string()}.pdf"

PlotUtils.use_tex()
Plot.multi_plot([x_rewards_0, x_rewards_1, x_rewards_2], [y_rewards_0, y_rewards_1, y_rewards_2], x_label="Time (s)",
                y_label=r"$\iota$ (in-deadline rate)", fullpath=figure_filename,
                legend=["Cluster 1", "Cluster 2", "Cluster 3"])

"""

print(x_eps)
print(y_eps)

cmap_def = plt.get_cmap("tab10")

PlotUtils.use_tex()
fig, ax = plt.subplots()
# make a plot
ax.plot(x_rewards, y_rewards, marker=r"$\triangle$", markersize=3.0, markeredgewidth=1, linewidth=0.7,
        color=cmap_def(0))
ax.set_xlabel("Time")
ax.set_ylabel("in-deadline Rate")

ax2 = ax.twinx()
ax2.plot(x_eps, y_eps, marker=None, markersize=3.0, markeredgewidth=1, linewidth=1, color=cmap_def(1))
ax2.set_xlabel("Time")
ax2.set_ylabel(r"$\epsilon$")

plt.savefig(figure_filename)

# PlotUtils.use_tex()
# Plot.plot([x_rewards, x_eps], [y_rewards, y_eps], "Time", "Reward", fullpath=figure_filename)

"""
