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


from __future__ import annotations

import os
import shutil
import signal
import sys
from datetime import datetime

import simpy

from cloud import Cloud
from log import Log
from node import Node
from service_data_storage import ServiceDataStorage
from service_discovery import ServiceDiscovery
import plot

"""
Run the simulation of deadline scheduling
"""

MODULE = "Main"

SIMULATION_TIME = 10000

SIMULATION_TOTAL_TIME = SIMULATION_TIME
# NODES = 6  # number of nodes

SESSION_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
LEARNING_TYPE = Node.LearningType.D_SARSA
NO_LEARNING_POLICY = Node.NoLearningPolicy.LEAST_LOADED
ACTIONS_SPACE = Node.ActionsSpace.OTHER_CLUSTERS

MULTI_CLUSTER = True

os.makedirs(f"_config/{SESSION_ID}")
shutil.copy(f"./run_simulation_d-sarsa.py", f"_config/{SESSION_ID}")
print(f"this is the session id: {SESSION_ID}")


def simulate(env):
    nodes = []
    # create nodes

    cloud = Cloud(env, latency_roundtrip_ms=20)

    nodes.append(create_node(env, 0, 0, 1, Node.NodeType.SCHEDULER, 1.0))
    nodes.append(create_node(env, 1, 0, 1, Node.NodeType.WORKER, 0.2))
    nodes.append(create_node(env, 2, 0, 1, Node.NodeType.WORKER, 0.9))
    nodes.append(create_node(env, 3, 0, 1, Node.NodeType.WORKER, 0.7))
    # nodes.append(create_node(env, 4, 0, 1, Node.NodeType.WORKER, 0.7))
    # nodes.append(create_node(env, 5, 0, 1, Node.NodeType.WORKER, 0.6))

    if MULTI_CLUSTER:
        nodes.append(create_node(env, 6, 1, 1, Node.NodeType.SCHEDULER, 1.0))
        # nodes.append(create_node(env, 7, 1, 1, Node.NodeType.WORKER, 0.9))
        nodes.append(create_node(env, 8, 1, 1, Node.NodeType.WORKER, 0.6))

        nodes.append(create_node(env, 9, 2, 1, Node.NodeType.SCHEDULER, 1.0))
        nodes.append(create_node(env, 10, 2, 1, Node.NodeType.WORKER, 0.2))
        nodes.append(create_node(env, 11, 2, 1, Node.NodeType.WORKER, 0.7))
        nodes.append(create_node(env, 12, 2, 1, Node.NodeType.WORKER, 0.6))
        # nodes.append(create_node(env, 13, 2, 1, Node.NodeType.WORKER, 0.5))

        nodes.append(create_node(env, 14, 3, 2, Node.NodeType.UP_SCHEDULER, 1))
        nodes.append(create_node(env, 15, 3, 2, Node.NodeType.UP_WORKER, 2))
        nodes.append(create_node(env, 16, 3, 2, Node.NodeType.UP_WORKER, 2))
        nodes.append(create_node(env, 17, 3, 2, Node.NodeType.UP_WORKER, 2))
        nodes.append(create_node(env, 18, 3, 2, Node.NodeType.UP_WORKER, 2))
        # nodes.append(create_node(env, 19, 3, 2, Node.NodeType.UP_WORKER, 600))
        # nodes.append(create_node(env, 20, 3, 2, Node.NodeType.UP_WORKER, 500))
        # nodes.append(create_node(env, 21, 3, 2, Node.NodeType.UP_WORKER, 800))
        # nodes.append(create_node(env, 22, 3, 2, Node.NodeType.UP_WORKER, 500))
        # nodes.append(create_node(env, 23, 3, 2, Node.NodeType.UP_WORKER, 800))
        # simpy.events.Event

    # add them discovery service
    discovery = ServiceDiscovery(4, 2, nodes, cloud)
    data_storage = ServiceDataStorage(nodes, SESSION_ID, LEARNING_TYPE, NO_LEARNING_POLICY, ACTIONS_SPACE)
    # init nodes services, and data
    for node in nodes:
        node.set_service_discovery(discovery)
        node.set_service_data_storage(data_storage)
    for node in nodes:
        node.init()
    cloud.set_service_discovery(discovery)

    Log.minfo(MODULE, "Started simulation")
    env.run(until=SIMULATION_TOTAL_TIME)

    Log.minfo(MODULE, f"Simulation ended: SESSION_ID={SESSION_ID}, LEARNING_TYPE={LEARNING_TYPE.name}, "
                      f"NO_LEARNING_POLICY={NO_LEARNING_POLICY.name}, ACTIONS_SPACE={ACTIONS_SPACE.name}")

    reject_count = 0
    for node in nodes:
        print(f"node = {node.get_uid()}, reject_count = {node.reject_count}")
        reject_count += node.reject_count

    print(f"Policy = D-SARSA, Much Servers In Second Level, Number of rejected jobs = {reject_count}")

    data_storage.done_simulation()


def get_die_after(node_id):
    if node_id == 1:
        return 4000
    return 0


def get_die_simulation(node_id):
    if node_id == 1:
        return True
    return False


def create_node(env, node_id, belong_to_cluster_id, level, node_type, machine_speed):
    return Node(env,
                node_id,
                SESSION_ID,
                level,
                simulation_time=SIMULATION_TIME,
                skip_plots=True,
                node_belong_to_cluster=belong_to_cluster_id,
                node_type=node_type,
                # die_simulation=get_die_simulation(node_id),
                die_after_seconds=get_die_after(node_id),
                die_duration=4000,
                # rates
                machine_speed=machine_speed,
                rate_l=30.0,
                # traffic model
                # rate_l_model_path="./traffic/namex/namex-traffic-daily-20210420.csv",
                # rate_l_model_path=f"./traffic/fixed/fixed_{i}.csv",
                # rate_l_model_path=f"./traffic/city/data/traffic_node_{i}.csv",
                # rate_l_model_path="./traffic/fictious/fictious_1.csv",
                rate_l_model_path_shift=0,  # i * 1200,  # 0,
                rate_l_model_path_cycles=3,
                rate_l_model_path_parse_x_max=None,
                rate_l_model_path_steady=False,
                rate_l_model_path_steady_for=2000,
                rate_l_model_path_steady_every=2000,
                # net
                # net_speed_client_scheduler_mbits=200,
                # net_speed_scheduler_scheduler_mbits=300,
                # net_speed_scheduler_worker_mbits=1000,
                # net_speed_scheduler_cloud_mbits=1000,
                net_speed_client_scheduler_mbits=200,
                net_speed_scheduler_scheduler_mbits=300,
                net_speed_scheduler_UPscheduler_mbits=3000,
                net_speed_UPscheduler_cloud_mbits=1000,
                net_speed_UPscheduler_UPworker_mbits=3000,
                net_speed_scheduler_worker_mbits=2000,
                net_speed_scheduler_cloud_mbits=500,

                # job info
                job_periodic_types=3,
                job_periodic_payload_sizes_mbytes=(0.050, 0.050, 0.050),
                job_periodic_duration_std_devs=(0.0003, 0.0003, 0.0003),
                job_periodic_percentages=(.33, .33, .34),
                job_periodic_deadlines=(0.016, 0.033, 0.070),
                job_periodic_durations=(0.010, 0.020, 0.055),
                job_periodic_arrival_time_std_devs=(0.001, 0.002, 0.01),
                job_periodic_rates_fps=(60, 30, 15),
                job_periodic_desired_rates_fps=(60, 30, 15),
                job_periodic_desired_rates_fps_max=(60, 30, 15),
                job_periodic_desired_rates_fps_min=(50, 20, 10),
                job_exponential_types=1,
                job_exponential_payload_sizes_mbytes=[0.1],
                job_exponential_duration_std_devs=[0.01],
                job_exponential_arrival_time_std_devs=[0.01],
                job_exponential_percentages=[1],
                job_exponential_deadlines=[0.300],
                job_exponential_durations=[0.100],
                job_exponential_rates_fps=[10],
                job_exponential_desired_rates_fps=[1],
                job_exponential_desired_rates_fps_min=[0],
                job_exponential_desired_rates_fps_max=[10],
                # node info
                max_jobs_in_queue=5,
                distribution_arrivals=Node.DistributionArrivals.POISSON,
                delay_probing=0.003,
                # learning
                sarsa_alpha=0.01,
                sarsa_beta=0.01,
                state_type=Node.StateType.JOB_TYPE,
                learning_type=Node.LearningType.D_SARSA,
               # no_learning_policy=Node.NoLearningPolicy.RANDOM,
                actions_space=ACTIONS_SPACE,
                pwr2_binary_policy="001111",
                tiling_num_tilings=26,
                # threshold=6,
                # use_model_from_session_name="20210412-125752",
                # distributions
                distribution_network_probing_sigma=0.0001,
                distribution_network_forwarding_sigma=0.00002,
                episode_length=60,
                eps=0.90,
                eps_decay=0.9995,
                eps_dynamic=True,
                eps_min=0.05,
                logging_info=True)


def main(argv):
    env = simpy.Environment()
    simulate(env)


#
# Signals
#

def signal_handler(signal, frame):
    Log.minfo(MODULE, "Interrupt received, closing gracefully")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

#
# Entrypoint
#

if __name__ == "__main__":
    main(sys.argv)

    # import cProfile
    # cProfile.run('main(sys.argv)')