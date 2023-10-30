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

from matplotlib import pyplot as plt

from log import Log
from plot import PlotUtils
from utils import Utils
from utils_plot import UtilsPlot

markers = [r"$\triangle$", r"$\square$", r"$\diamondsuit$", r"$\otimes$", r"$\star$"]

MODULE = "PlotRewardOverTime"

DB_FILE_FOLDER = "20231030-133417"
DB_FILE = f"./_log/learning/D_SARSA/OTHER_CLUSTERS/{DB_FILE_FOLDER}/log.db"
# DB_FILE = f"./_log/no-learning/LEAST_LOADED_NOT_AWARE/{DB_FILE_FOLDER}/log.db"

N_CLUSTERS = 4
CLUSTERS_IDS = [0, 6, 9]
JOB_TYPES = 3

db = sqlite3.connect(DB_FILE)
cur = db.cursor()

res = cur.execute("select max(generated_at) from jobs")
for line in res:
    simulation_time = math.ceil(line[0])

Log.mdebug(MODULE, f"simulation_time={simulation_time}")

average_every_secs = 250

reward_x_arr = []
reward_y_arr = []

x_rewards, y_rewards = UtilsPlot.plot_data_reward_over_time(0, DB_FILE, average_every_secs=average_every_secs)
reward_x_arr.append(x_rewards)
reward_y_arr.append(y_rewards)

x_rewards, y_rewards = UtilsPlot.plot_data_reward_over_time(6, DB_FILE, average_every_secs=average_every_secs)
reward_x_arr.append(x_rewards)
reward_y_arr.append(y_rewards)

x_rewards, y_rewards = UtilsPlot.plot_data_reward_over_time(9, DB_FILE, average_every_secs=average_every_secs)
reward_x_arr.append(x_rewards)
reward_y_arr.append(y_rewards)

x_eps = []
y_eps = []

# eps
res = cur.execute(f"select cast(generated_at as integer), avg(eps) from jobs where node_uid = 0 group by cast(generated_at as integer)")
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
figure_filename = f"./_plots/reward-over-time_{DB_FILE_FOLDER}_{Utils.current_time_string()}.pdf"

print(x_eps)
print(y_eps)

cmap_def = plt.get_cmap("tab10")

PlotUtils.use_tex()
fig, ax = plt.subplots(nrows=4, ncols=1, gridspec_kw={'height_ratios': [1, 2, 2, 2]})

# make a plot
for i in range(len(reward_x_arr)):
    print(f"Plotting reward line={i} len_x={len(reward_y_arr[i])} len_y={len(reward_x_arr[i])}")
    ax[0].plot(reward_x_arr[i], reward_y_arr[i], marker=markers[(i%N_CLUSTERS) % len(markers)], markersize=3.0, markeredgewidth=1, linewidth=0.7, color=cmap_def(i))

# ax[0].set_xlabel("Time")
ax[0].set_ylabel("Reward/s")
ax[0].set_xlim([0, simulation_time])

ax2 = ax[0].twinx()
ax2.plot(x_eps, y_eps, marker=None, markersize=3.0, markeredgewidth=1, linewidth=1, color=cmap_def(1))
ax2.set_xlabel("Time")
ax2.set_ylabel(r"$\epsilon$")

# N_CLUSTERS
#
# client fps
#

LEGEND = ["60FPS Client (Min. 50)", "30FPS Client (Min. 20)", "15FPS Client (Min. 10)"]
FPS_LIMITS_MAX = [60, 30, 15]
FPS_LIMITS_MIN = [50, 20, 10]

db = sqlite3.connect(DB_FILE)
cur = db.cursor()

simulation_time = 0
res = cur.execute("select max(generated_at) from jobs")
for line in res:
    simulation_time = math.ceil(line[0])

cur.close()
db.close()

x_arr = []
y_arr = []

for cluster_id in CLUSTERS_IDS:
    x, y = UtilsPlot.plot_data_average_client_fps_time(cluster_id, DB_FILE, job_type=0, max_fps=FPS_LIMITS_MAX[0], average_every_secs=average_every_secs)
    x_arr.append(x)
    y_arr.append(y)

    x, y = UtilsPlot.plot_data_average_client_fps_time(cluster_id, DB_FILE, job_type=1, max_fps=FPS_LIMITS_MAX[1], average_every_secs=average_every_secs)
    x_arr.append(x)
    y_arr.append(y)

    x, y = UtilsPlot.plot_data_average_client_fps_time(cluster_id, DB_FILE, job_type=2, max_fps=FPS_LIMITS_MAX[2], average_every_secs=average_every_secs)
    x_arr.append(x)
    y_arr.append(y)

    # x, y = UtilsPlot.plot_data_average_client_fps_time(0, DB_FILE, job_type=3, max_fps=FPS_LIMITS_MAX[3], average_every_secs=average_every_secs)
    # x_arr.append(x)
    # y_arr.append(y)

    # x, y = UtilsPlot.plot_data_average_fps_over_time(0, llna_db_file, job_type=4, average_every_secs=average_every_secs)
    # x_arr.append(x)
    # y_arr.append(y)


# PlotUtils.use_tex()
# Plot.multi_plot(x_arr, y_arr, x_label="Time (s)", y_label="FPS", legend=["60FPS", "30FPS", "15FPS", "7FPS", "10FPS"],
#                 fullpath=figure_filename)

legend_arr = []

for i in range(len(y_arr)):
    line, = ax[1].plot(x_arr[i], y_arr[i], markerfacecolor='None', linewidth=0.6,
                       marker=markers[i % len(markers)], color=cmap_def(i% JOB_TYPES),
                       markersize=3, markeredgewidth=0.6)
    print(f"Plotting average fps, line={i}, i={i% JOB_TYPES}")
    # plt.hlines(FPS_LIMITS_MAX[i], 0, simulation_time, colors=cmap_def(i), linestyles='solid', label=LEGEND[i], linewidth=0.6)
    # plt.hlines(FPS_LIMITS_MIN[i], 0, simulation_time, colors=cmap_def(i), linestyles='solid', label=LEGEND[i], linewidth=0.6)
    if i < JOB_TYPES:
        ax[1].fill_between(x, FPS_LIMITS_MAX[i % JOB_TYPES], FPS_LIMITS_MIN[i% JOB_TYPES], color=cmap_def(i% JOB_TYPES), alpha=0.3, linewidth=0)

    legend_arr.append(line)

ax[3].legend(legend_arr, LEGEND, fontsize="small")  # , loc="lower right")

# ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("FPS")
ax[1].set_ylim([0, max(FPS_LIMITS_MAX)])
ax[1].set_xlim([0, simulation_time])

#
# lag
#

x_arr = []
y_arr = []

for cluster_id in CLUSTERS_IDS:
    x, y = UtilsPlot.plot_data_average_lag_over_time(cluster_id, DB_FILE, job_type=0, max_fps=FPS_LIMITS_MAX[0], average_every_secs=average_every_secs)
    x_arr.append(x)
    y_arr.append(y)

    x, y = UtilsPlot.plot_data_average_lag_over_time(cluster_id, DB_FILE, job_type=1, max_fps=FPS_LIMITS_MAX[1], average_every_secs=average_every_secs)
    x_arr.append(x)
    y_arr.append(y)

    x, y = UtilsPlot.plot_data_average_lag_over_time(cluster_id, DB_FILE, job_type=2, max_fps=FPS_LIMITS_MAX[2], average_every_secs=average_every_secs)
    x_arr.append(x)
    y_arr.append(y)

    #x, y = UtilsPlot.plot_data_average_lag_over_time(0, DB_FILE, job_type=3, max_fps=FPS_LIMITS_MAX[3], average_every_secs=average_every_secs)
    #x_arr.append(x)
    #y_arr.append(y)

for i in range(len(y_arr)):
    line, = ax[3].plot(x_arr[i], y_arr[i], markerfacecolor='None', linewidth=0.6,
                       marker=markers[i % len(markers)],  color=cmap_def(i% JOB_TYPES),
                       markersize=3, markeredgewidth=0.6)

ax[3].set_ylabel("Lag Time (ms)")
ax[3].set_xlim([0, simulation_time])
# ax[3].set_ylim([0, 150])


#
# response time
#

x_arr = []
y_arr = []

for cluster_id in CLUSTERS_IDS:
    x, y = UtilsPlot.plot_data_average_response_time_over_time(cluster_id, DB_FILE, job_type=0, max_fps=FPS_LIMITS_MAX[0], average_every_secs=average_every_secs)
    x_arr.append(x)
    y_arr.append(y)

    x, y = UtilsPlot.plot_data_average_response_time_over_time(cluster_id, DB_FILE, job_type=1, max_fps=FPS_LIMITS_MAX[1], average_every_secs=average_every_secs)
    x_arr.append(x)
    y_arr.append(y)

    x, y = UtilsPlot.plot_data_average_response_time_over_time(cluster_id, DB_FILE, job_type=2, max_fps=FPS_LIMITS_MAX[2], average_every_secs=average_every_secs)
    x_arr.append(x)
    y_arr.append(y)

    #x, y = UtilsPlot.plot_data_average_response_time_over_time(0, DB_FILE, job_type=3, max_fps=FPS_LIMITS_MAX[3], average_every_secs=average_every_secs)
    #x_arr.append(x)
    #y_arr.append(y)

for i in range(len(y_arr)):
    line, = ax[2].plot(x_arr[i], y_arr[i], markerfacecolor='None', linewidth=0.6,
                       marker=markers[i % len(markers)], color=cmap_def(i% JOB_TYPES),
                       markersize=3, markeredgewidth=0.6)
    # plt.hlines(FPS_LIMITS_MAX[i], 0, simulation_time, colors=cmap_def(i), linestyles='solid', label=LEGEND[i], linewidth=0.6)
    # plt.hlines(FPS_LIMITS_MIN[i], 0, simulation_time, colors=cmap_def(i), linestyles='solid', label=LEGEND[i], linewidth=0.6)
    # if i != 3:
    if i < JOB_TYPES:
        ax[2].fill_between(x, 1000 / FPS_LIMITS_MAX[i%JOB_TYPES], 1000 / FPS_LIMITS_MIN[i%JOB_TYPES], color=cmap_def(i% JOB_TYPES), alpha=0.3, linewidth=0)

ax[2].set_ylabel("Response Time (ms)")
ax[2].set_xlim([0, simulation_time])
# ax[2].set_ylim([15, 50])


#
# final
#

ax[3].set_xlabel("Time (s)")

fig.tight_layout(h_pad=0, w_pad=0)
fig.set_figwidth(6.4)  # 6.4
fig.set_figheight(7.5)  # 4.8

figure_filename = f"./_plots/plot_stacked_reward_fps_lag_res_multiple_cluster_time_{DB_FILE_FOLDER}_{Utils.current_time_string()}.pdf"

fig.subplots_adjust(
    top=0.979,
    bottom=0.067,
    left=0.095,
    right=0.917,
    hspace=0.15,
    wspace=0
)

# plt.show()

plt.savefig(figure_filename, bbox_inches='tight', transparent="True", pad_inches=0)

# PlotUtils.use_tex()
# Plot.plot([x_rewards, x_eps], [y_rewards, y_eps], "Time", "Reward", fullpath=figure_filename)
