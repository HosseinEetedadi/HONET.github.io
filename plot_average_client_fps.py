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


# least loaded not aware
import math
import os
import sqlite3

from matplotlib import pyplot as plt

from plot import PlotUtils
from utils import Utils
from utils_plot import UtilsPlot

average_every_secs = 120

LEGEND = ["60FPS Client", "60FPS Client", "30FPS Client", "15FPS Client"]
FPS_LIMITS_MAX = [60, 60, 30, 15]
FPS_LIMITS_MIN = [50, 40, 20, 10]
cmap_def = plt.get_cmap("tab10")

llna_db_folder = "20231008-202835"
# llna_db_file = f"./_log/no-learning/LEAST_LOADED_NOT_AWARE/{llna_db_folder}/log.db"
llna_db_file = f"{os.getcwd()}/_log/learning/D_SARSA/OTHER_CLUSTERS/{llna_db_folder}/log.db"
print(llna_db_file)

db = sqlite3.connect(llna_db_file)
cur = db.cursor()

simulation_time = 0
res = cur.execute("select max(generated_at) from jobs")
for line in res:
    simulation_time = math.ceil(line[0])

cur.close()
db.close()

x_arr = []
y_arr = []

x, y = UtilsPlot.plot_data_average_client_fps_time(0, llna_db_file, job_type=0, max_fps=FPS_LIMITS_MAX[0], average_every_secs=average_every_secs)
x_arr.append(x)
y_arr.append(y)

x, y = UtilsPlot.plot_data_average_client_fps_time(0, llna_db_file, job_type=1, max_fps=FPS_LIMITS_MAX[1], average_every_secs=average_every_secs)
x_arr.append(x)
y_arr.append(y)

x, y = UtilsPlot.plot_data_average_client_fps_time(0, llna_db_file, job_type=2, max_fps=FPS_LIMITS_MAX[2], average_every_secs=average_every_secs)
x_arr.append(x)
y_arr.append(y)

x, y = UtilsPlot.plot_data_average_client_fps_time(0, llna_db_file, job_type=3, max_fps=FPS_LIMITS_MAX[3], average_every_secs=average_every_secs)
x_arr.append(x)
y_arr.append(y)

# x, y = UtilsPlot.plot_data_average_fps_over_time(0, llna_db_file, job_type=4, average_every_secs=average_every_secs)
# x_arr.append(x)
# y_arr.append(y)

figure_filename = f"./_plots/average-client-fps-over-time_{llna_db_folder}_{Utils.current_time_string()}.pdf"

PlotUtils.use_tex()
# Plot.multi_plot(x_arr, y_arr, x_label="Time (s)", y_label="FPS", legend=["60FPS", "30FPS", "15FPS", "7FPS", "10FPS"],
#                 fullpath=figure_filename)

plt.clf()
fig, ax = plt.subplots()
markers = [r"$\triangle$", r"$\square$", r"$\diamondsuit$", r"$\otimes$", r"$\star$"]

legend_arr = []

for i in range(len(y_arr)):
    line, = ax.plot(x_arr[i], y_arr[i], markerfacecolor='None', linewidth=0.6,
                    marker=markers[i % len(markers)],
                    markersize=5, markeredgewidth=0.6)
    print(i)
    # plt.hlines(FPS_LIMITS_MAX[i], 0, simulation_time, colors=cmap_def(i), linestyles='solid', label=LEGEND[i], linewidth=0.6)
    # plt.hlines(FPS_LIMITS_MIN[i], 0, simulation_time, colors=cmap_def(i), linestyles='solid', label=LEGEND[i], linewidth=0.6)
    plt.fill_between(x, FPS_LIMITS_MAX[i], FPS_LIMITS_MIN[i], color=cmap_def(i), alpha=0.3, linewidth=0)

    legend_arr.append(line)

plt.legend(legend_arr, LEGEND, fontsize="small")  # , loc="lower right")

ax.set_xlabel("Time (s)")
ax.set_ylabel("FPS")
ax.set_ylim([0,max(FPS_LIMITS_MAX)])
fig.tight_layout()


os.makedirs("_plots", exist_ok=True)

plt.savefig(figure_filename)

plt.close(fig)
