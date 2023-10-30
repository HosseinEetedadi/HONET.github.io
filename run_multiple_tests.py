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

import threading
from datetime import datetime
import shutil
import os

from log import Log
from node import Node
from simulate import Simulate

MODULE = "RunMultipleTest"

SESSION_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
N_TESTS = 5

LEARNING_TYPES = [
    Node.LearningType.D_SARSA,
    Node.LearningType.D_SARSA,
    Node.LearningType.NO_LEARNING,
    Node.LearningType.NO_LEARNING,
    Node.LearningType.NO_LEARNING
]

NO_LEARNING_POLICY = [
    None,
    None,
    Node.NoLearningPolicy.LEAST_LOADED_AWARE_CLOUD,
    Node.NoLearningPolicy.LEAST_LOADED_AWARE,
    Node.NoLearningPolicy.LEAST_LOADED_NOT_AWARE
]

ACTIONS_SPACES = [
    Node.ActionsSpace.WORKERS_OR_CLOUD,
    Node.ActionsSpace.ONLY_WORKERS,
    Node.ActionsSpace.WORKERS_OR_CLOUD,
    Node.ActionsSpace.ONLY_WORKERS,
    Node.ActionsSpace.ONLY_WORKERS
]


threads = []
for i in range(N_TESTS):
    log_dir = f"{Node.BASE_DIR_LOG}/py_files/{SESSION_ID}"
    os.makedirs(log_dir, exist_ok=True)
    shutil.copyfile("./run_multiple_tests.py", f"{log_dir}/run_multiple_tests.txt")
    shutil.copyfile("./simulate.py", f"{log_dir}/simulate.txt")

    def sim_thread():
        sim = Simulate(session_id=SESSION_ID,
                       simulation_time=30000,
                       learning_type=LEARNING_TYPES[i],
                       no_learning_policy=NO_LEARNING_POLICY[i],
                       actions_space=ACTIONS_SPACES[i],
                       multi_cluster=False,
                       rate_l=50.0,
                       sarsa_alpha=0.01,
                       sarsa_beta=0.01,
                       episode_len=200)
        sim.simulate()

    t = threading.Thread(target=sim_thread, args=[])
    threads.append(t)

    t.start()

for t in threads:
    t.join()

Log.minfo(MODULE, f"Finished SESSIONS_ID={SESSION_ID}")
