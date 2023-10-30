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

import random
from typing import List

# from node import Node
from log import Log
from node import Node

"""
Implementation of the discovery service for nodes
"""

MODULE = "Discovery"
DEBUG = False


class ServiceDiscovery:
    """Module for allowing nodes cooperation"""

    def __init__(self, n_clusters, n_level, nodes, node_cloud):
        self._nodes = nodes  # type: 'List[Node]'
        self._n_nodes = len(nodes)
        self._coefficients = []
        self._n_clusters = n_clusters
        self.n_level = n_level
        self._node_cloud = node_cloud

        # mapping cluster_id -> list of nodes uids in the cluster
        self._clusters_allocation = {}
        self._Up_clusters_allocation = {}
        for cluster_i in range(self._n_clusters - 1):
            self._clusters_allocation[cluster_i] = []

        self._Up_clusters_allocation[self._n_clusters - 1] = []
        # mapping cluster_id -> node uid of the scheduler
        self._clusters_schedulers = {}
        self._Up_cluster_schedulers = {}
        # mapping of node_id -> node
        self._nodes_id_dict = {}
        self._Up_nodes_id_dict = {}
        for node in nodes:
            # associate node id to nodes dict
            if node.get_level() == 1:
                self._nodes_id_dict[node.get_uid()] = node
                # associate node id to cluster
                self._clusters_allocation[node.get_cluster_id()].append(node.get_uid())
                # register scheduler if one
                if node.get_type() == Node.NodeType.SCHEDULER:
                    self._clusters_schedulers[node.get_cluster_id()] = node.get_uid()
            elif node.get_level() == 2:
                self._Up_nodes_id_dict[node.get_uid()] = node
                self._Up_clusters_allocation[node.get_cluster_id()].append(node.get_uid())
                if node.get_type() == Node.NodeType.UP_SCHEDULER:
                    self._Up_cluster_schedulers[node.get_cluster_id()] = node.get_uid()
        for cluster_i in range(self._n_clusters - 1):
            self._clusters_allocation[cluster_i] = sorted(self._clusters_allocation[cluster_i])

        self._Up_clusters_allocation[self._n_clusters - 1] = sorted(self._Up_clusters_allocation[self._n_clusters - 1])
        # print(f"this is the value of clusters_schedulers:  {self._clusters_schedulers}")
        # print(f"This is the cluters_allocation: {self._clusters_allocation}")
        # print(f"This is the up_cluters_allocation: {self._Up_clusters_allocation}")
        # print(f"this is the value of up_clusters_schedulers:  {self._Up_cluster_schedulers}")

        # print("this is the cluster_id value after sorting:  ", self._clusters_allocation)
        Log.minfo(MODULE, f"Initialized {n_clusters} clusters = {self._clusters_allocation}")
        Log.minfo(MODULE, f"Initialized {n_clusters} schedulers = {self._clusters_schedulers}")

    def get_random_node(self, current_node_id=None):
        """Get a random node in list except the current one"""
        if current_node_id is not None:
            nodes_list = self._get_all_but_current(current_node_id)
        else:
            nodes_list = self.get_all_nodes()
        if len(nodes_list) == 0:
            return None

        return nodes_list[random.randint(0, len(nodes_list) - 1)]

    def get_n_random_nodes(self, n) -> List[Node]:
        picked_nodes = []
        all_nodes = [node for node in self._nodes]
        if n > self._n_nodes:
            return []

        for i in range(n):
            picked_nodes.append(all_nodes.pop(random.randint(0, len(all_nodes) - 1)))

        return picked_nodes

    def get_node_by_uid(self, node_id):
        """Return a node by its id"""
        if node_id not in self._nodes_id_dict.keys():
            raise RuntimeError(f"Requested node_id={node_id} is non-existent")
            # return None
        return self._nodes_id_dict[node_id]

    def get_up_node_by_uid(self, node_id):
        """Return a node by its id"""
        if node_id not in self._Up_nodes_id_dict.keys():
            raise RuntimeError(f"Requested node_id={node_id} is non-existent")
            # return None
        return self._Up_nodes_id_dict[node_id]


    def get_node_by_index(self, i):
        return self._nodes[i]

    def get_least_loaded_node(self):
        """Retrieve the node with the least loaded queue"""
        nodes_list = self.get_all_nodes()  # type: List[Node]
        min_value = nodes_list[0].get_current_load()
        min_i = 0
        for i in range(1, len(nodes_list)):
            if nodes_list[i].get_current_load() < min_value:
                min_value = nodes_list[i].get_current_load()
                min_i = i
        return nodes_list[min_i]

    def get_nodes_in_cluster(self, cluster_id):
        nodes_in_cluster = []
        for node_uid in self._clusters_allocation[cluster_id]:
            nodes_in_cluster.append(self.get_node_by_uid(node_uid))
        return nodes_in_cluster

    def get_workers_in_cluster(self, cluster_id):
        nodes_in_cluster = []
        for node_uid in self._clusters_allocation[cluster_id]:
            if self.get_node_by_uid(node_uid).get_type() == Node.NodeType.WORKER:
                nodes_in_cluster.append(self.get_node_by_uid(node_uid))
        return nodes_in_cluster


    def get_workers_in_up_cluster(self, cluster_id):
        nodes_in_up_cluster = []
        for node_uid in self._Up_clusters_allocation[cluster_id]:
            if self.get_up_node_by_uid(node_uid).get_type() == Node.NodeType.UP_WORKER:
                nodes_in_up_cluster.append(self.get_up_node_by_uid(node_uid))
        return nodes_in_up_cluster


    def get_worker_in_cluster_by_index(self, cluster_id, index):
        nodes_in_cluster = self.get_workers_in_cluster(cluster_id)
        return nodes_in_cluster[index]

    def get_workers_in_cluster_count(self, cluster_id):
        return len(self.get_workers_in_cluster(cluster_id))

    def get_workers_in_up_cluster_count(self, cluster_id):
        return len(self.get_workers_in_up_cluster(cluster_id))


    def get_nodes_ids_in_cluster(self, cluster_id):
        return self._clusters_allocation[cluster_id]

    def get_total_nodes_in_cluster(self, cluster_id):
        return len(self._clusters_allocation[cluster_id])

    def get_clusters_count(self):
        return self._n_clusters

    def get_node_cloud(self):
        return self._node_cloud

    def get_node_scheduler_for_cluster_id(self, cluster_id):
        """Retrieve the scheduler node uid from the cluster uid"""
        # print(f"here - DOWN **** cluster_Id = {cluster_id}")
        return self.get_node_by_uid(self._clusters_schedulers[cluster_id])


    def get_node_up_scheduler_for_cluster_id(self, cluster_id):
        """Retrieve the scheduler node uid from the cluster uid"""
        # print(f" here - UP ***** cluster_Id = {cluster_id}")
        return self.get_up_node_by_uid(self._Up_cluster_schedulers[cluster_id])

    def get_cluster_uid_from_index(self, current_cluster_id, cluster_index):
        """Convert a cluster index to cluster uid"""
        # print(f" this is the cluster index: {cluster_index}")


        clusters_arr = list(self._clusters_schedulers.keys())
        # print(f"before - this is the clusters_arr: {clusters_arr}")
        clusters_arr.remove(current_cluster_id)
        # print(f"after - this is the clusters_arr: {clusters_arr}")

        # check
        if cluster_index >= len(clusters_arr):
            return  clusters_arr[cluster_index - 1]
            # raise RuntimeError(f"get_cluster_uid_from_index: bad index={cluster_index}, max={len(clusters_arr) - 1}")

        return clusters_arr[cluster_index]

    def get_node_scheduler_for_cluster_index(self, current_cluster_id, cluster_index):
        """Retrieve the scheduler node uid from the cluster index"""
        cluster_uid = self.get_cluster_uid_from_index(current_cluster_id, cluster_index)
        return self.get_node_scheduler_for_cluster_id(cluster_uid)

    def get_all_nodes(self):
        return self._nodes

    def are_nodes_idle(self) -> bool:
        """Check if all nodes have clear queues and nothing running"""
        for node in self._nodes:
            if not node.is_idle():
                return False

        return True

    def get_faster_node_not_died(self):
        for nodes in self._nodes:
            if not nodes.is_died():
                return nodes
        raise RuntimeError("No alive node")

    #
    # Internals
    #

    def _get_all_but_current(self, current_node_id):
        """Prepare a list of all nodes but the current"""
        nodes_list = []
        for node in self._nodes:
            if node.get_uid() == current_node_id:
                continue
            nodes_list.append(node)
        return nodes_list
