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

import simpy

from cloud import Cloud
from job import Job
from log import Log
from node import Node
from service_data_storage import ServiceDataStorage
from service_discovery import ServiceDiscovery

"""
Run the simulation of deadline scheduling
"""

MODULE = "Simulate"


class Simulate:

    def __init__(self, session_id, simulation_time=10000, learning_type=Node.LearningType.NO_LEARNING,
                 no_learning_policy=Node.NoLearningPolicy, actions_space=Node.ActionsSpace.WORKERS_OR_CLOUD,
                 multi_cluster=False, rate_l=50.0, episode_len=20, sarsa_alpha=0.001, sarsa_beta=0.01):
        self._rate_l = rate_l
        self._session_id = session_id
        self._learning_type = learning_type
        self._no_learning_policy = no_learning_policy
        self._actions_space = actions_space
        self._multi_cluster = multi_cluster
        self._simulation_time = simulation_time
        self._episode_len = episode_len
        self._sarsa_alpha = sarsa_alpha
        self._sarsa_beta = sarsa_beta

        self._log_sim_params()

    def _simulate(self, env):
        nodes = []
        # create nodes

        cloud = Cloud(env, latency_roundtrip_ms=20)

        nodes.append(self.create_node(env, 0, 0, Node.NodeType.SCHEDULER, 1.0))
        nodes.append(self.create_node(env, 1, 0, Node.NodeType.WORKER, 1.0))
        # nodes.append(self.create_node(env, 2, 0, Node.NodeType.WORKER, 0.9))
        nodes.append(self.create_node(env, 2, 0, Node.NodeType.WORKER, 0.75))
        nodes.append(self.create_node(env, 3, 0, Node.NodeType.WORKER, 0.7))
        nodes.append(self.create_node(env, 4, 0, Node.NodeType.WORKER, 0.6))
        # nodes.append(self.create_node(env, 4, 0, Node.NodeType.WORKER, 0.6))

        if self._multi_cluster:
            nodes.append(self.create_node(env, 5, 1, Node.NodeType.SCHEDULER, 1.0))
            nodes.append(self.create_node(env, 6, 1, Node.NodeType.WORKER, 1.0))
            nodes.append(self.create_node(env, 7, 1, Node.NodeType.WORKER, 0.6))
            nodes.append(self.create_node(env, 8, 1, Node.NodeType.WORKER, 0.4))
            nodes.append(self.create_node(env, 9, 1, Node.NodeType.WORKER, 0.2))

            nodes.append(self.create_node(env, 10, 2, Node.NodeType.SCHEDULER, 1.0))
            nodes.append(self.create_node(env, 11, 2, Node.NodeType.WORKER, 1.0))
            nodes.append(self.create_node(env, 12, 2, Node.NodeType.WORKER, 0.6))
            nodes.append(self.create_node(env, 13, 2, Node.NodeType.WORKER, 0.4))
            nodes.append(self.create_node(env, 14, 2, Node.NodeType.WORKER, 0.2))

        # nodes.append(self.create_node(env, 5, 0, Node.NodeType.WORKER, 0.2))
        # nodes.append(self.create_node(env, 6, 0, Node.NodeType.WORKER, 0.3))
        # nodes.append(self.create_node(env, 7, 0, Node.NodeType.WORKER, 0.3))
        # nodes.append(self.create_node(env, 8, 0, Node.NodeType.WORKER, 0.2))
        # nodes.append(self.create_node(env, 9, 0, Node.NodeType.WORKER, 0.2))
        # nodes.append(self.create_node(env, 10, 0, Node.NodeType.WORKER, 0.2))

        # add them discovery service
        discovery = ServiceDiscovery(3, nodes, cloud)
        data_storage = ServiceDataStorage(nodes, self._session_id, self._learning_type, self._no_learning_policy,
                                          self._actions_space)

        # init nodes services, and data
        for node in nodes:
            node.set_service_discovery(discovery)
            node.set_service_data_storage(data_storage)
        for node in nodes:
            node.init()
        cloud.set_service_discovery(discovery)

        Log.minfo(MODULE, "Started simulation")
        env.run(until=self._simulation_time)
        self._log_sim_params()

        data_storage.done_simulation()

    def _log_sim_params(self):
        Log.minfo(MODULE, f"Simulation ended: SESSION_ID={self._session_id}, LEARNING_TYPE={self._learning_type.name}, "
                          f"NO_LEARNING_POLICY={self._no_learning_policy.name if self._no_learning_policy is not None else 'None'}, "
                          f"ACTIONS_SPACE={self._actions_space.name}, RATE_L={self._rate_l}")

    def create_node(self, env, node_id, belong_to_cluster_id, node_type, machine_speed):
        return Node(env,
                    node_id,
                    self._session_id,
                    simulation_time=self._simulation_time,
                    skip_plots=True,
                    node_belong_to_cluster=belong_to_cluster_id,
                    node_type=node_type,
                    # die_simulation=get_die_simulation(node_id),
                    die_after_seconds=Simulate.get_die_after(node_id),
                    die_duration=600,
                    # rates
                    machine_speed=machine_speed,
                    rate_l=self._rate_l,
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
                    net_speed_client_scheduler_mbits=200,
                    net_speed_scheduler_scheduler_mbits=300,
                    net_speed_scheduler_worker_mbits=1000,
                    net_speed_scheduler_cloud_mbits=1000,
                    # job info
                    job_types=4,
                    job_duration_type=Job.DurationType.GAUSSIAN,
                    job_payload_sizes_mbytes=(0.050, 0.050, 0.100, 0.150),
                    job_duration_std_devs=(0.0003, 0.0003, 0.0003, 0.0003),
                    job_percentages=(.25, .25, .25, .25),
                    job_deadlines=(0.016, 0.033, 0.070, 10),
                    job_durations=(0.005, 0.010, 0.015, 0.100),
                    # node info
                    max_jobs_in_queue=4,
                    distribution_arrivals=Node.DistributionArrivals.POISSON,
                    delay_probing=0.003,
                    # learning
                    sarsa_alpha=self._sarsa_alpha,
                    sarsa_beta=self._sarsa_beta,
                    state_type=Node.StateType.JOB_TYPE,
                    learning_type=self._learning_type,
                    no_learning_policy=self._no_learning_policy,
                    actions_space=self._actions_space,
                    pwr2_binary_policy="001111",
                    tiling_num_tilings=24,
                    # threshold=6,
                    # use_model_from_session_name="20210412-125752",
                    # distributions
                    distribution_network_probing_sigma=0.0001,
                    distribution_network_forwarding_sigma=0.00001,
                    distribution_network_latency_cloud_sigma=0.00001,
                    episode_length=self._episode_len,
                    eps=0.9,
                    eps_dynamic=True,
                    eps_min=0.05,
                    logging_info=False)

    @staticmethod
    def get_die_after(node_id):
        if node_id == 1:
            return 3000
        if node_id == 2:
            return 6000
        if node_id == 3:
            return 8000
        return 0

    @staticmethod
    def get_die_simulation(node_id):
        if node_id == 1:
            return True
        if node_id == 2:
            return True
        if node_id == 3:
            return True
        return False

    def simulate(self):
        env = simpy.Environment()
        self._simulate(env)
