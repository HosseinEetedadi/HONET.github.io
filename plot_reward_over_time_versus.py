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


# plot sarsa
from plot import PlotUtils, Plot
from utils import Utils
from utils_plot import UtilsPlot

# LEGEND = ["Sarsa", "Random", "Round Robin", "Least Loaded (No Cloud)", "Least Loaded w/ Cloud"]
# LEGEND = ["Sarsa", "Least Loaded (No Cloud)", "Least Loaded (No Cloud) (No bias)"]
# LEGEND = ["Sarsa", "Sarsa (No Cloud)", "Least Loaded Aware (No Cloud)", "Least Loaded (No Cloud)"]
LEGEND = ["Sarsa", "Sarsa (No Cloud)", "Least Loaded Aware (Cloud)", "Least Loaded Aware (No Cloud)", "Least Loaded (No Cloud)"]
# LEGEND = ["Least Loaded Aware (Cloud)", "Least Loaded Aware (No Cloud)", "Least Loaded (No Cloud)"]


average_every_secs = 120

sarsa_db_folder = "20230725-104816"
sarsa_db_file = f"./_log/learning/D_SARSA/WORKERS_OR_CLOUD/{sarsa_db_folder}/log.db"
print(sarsa_db_file)
x_sarsa_rewards, y_sarsa_rewards = UtilsPlot.plot_data_reward_over_time(0, sarsa_db_file, average_every_secs=average_every_secs)

sarsa_nc_db_folder = "20230725-104816"
sarsa_nc_db_file = f"./_log/learning/D_SARSA/ONLY_WORKERS/{sarsa_nc_db_folder}/log.db"
print(sarsa_nc_db_file)
x_sarsa_nc_rewards, y_sarsa_nc_rewards = UtilsPlot.plot_data_reward_over_time(0, sarsa_nc_db_file, average_every_secs=average_every_secs)

# plot random
random_db_folder = "20230725-104816"
random_db_file = f"./_log/no-learning/RANDOM/{random_db_folder}/log.db"
# print(random_db_file)
# x_random_rewards, y_random_rewards = UtilsPlot.plot_data_reward_over_time(0, random_db_file,
#                                                                           average_every_secs=average_every_secs)
# plot round robin
rr_db_folder = "20210929-061953"
rr_db_file = f"./_log/no-learning/ROUND_ROBIN/{rr_db_folder}/log.db"
print(rr_db_file)
# x_rr_rewards, y_rr_rewards = UtilsPlot.plot_data_reward_over_time(0, rr_db_file, average_every_secs=average_every_secs)

# least loaded
ll_db_folder = "20210929-061953"
ll_db_file = f"./_log/no-learning/LEAST_LOADED_AWARE/{ll_db_folder}/log.db"
print(ll_db_file)
x_ll_rewards, y_ll_rewards = UtilsPlot.plot_data_reward_over_time(0, ll_db_file, average_every_secs=average_every_secs)

# least loaded not aware
llna_db_folder = "20210929-061953"
llna_db_file = f"./_log/no-learning/LEAST_LOADED_NOT_AWARE/{llna_db_folder}/log.db"
print(llna_db_file)
x_llna_rewards, y_llna_rewards = UtilsPlot.plot_data_reward_over_time(0, llna_db_file, average_every_secs=average_every_secs)

# least loaded with cloud
llc_db_folder = "20210929-061953"
llc_db_file = f"./_log/no-learning/LEAST_LOADED_AWARE_CLOUD/{llc_db_folder}/log.db"
print(llc_db_file)
x_llc_rewards, y_llc_rewards = UtilsPlot.plot_data_reward_over_time(0, llc_db_file, average_every_secs=average_every_secs)

# plot
figure_filename = f"./_plots/reward-over-time-versus_{Utils.current_time_string()}.pdf"

# PlotUtils.use_tex()
# Plot.multi_plot([x_sarsa_rewards, x_random_rewards, x_rr_rewards, x_ll_rewards, x_llc_rewards],
#                 [y_sarsa_rewards, y_random_rewards, y_rr_rewards, y_ll_rewards, y_llc_rewards],
#                x_label="Time (s)", y_label="$\iota$ (in-deadline rate)", fullpath=figure_filename, legend=LEGEND)
Plot.multi_plot([x_sarsa_rewards, x_sarsa_nc_rewards, x_llc_rewards, x_ll_rewards, x_llna_rewards],
                [y_sarsa_rewards, y_sarsa_nc_rewards, y_llc_rewards, y_ll_rewards, y_llna_rewards],
                x_label="Time (s)", y_label="Reward-per-second", fullpath=figure_filename, legend=LEGEND, show_markers=False,
                legend_position="lower right")
