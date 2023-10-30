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

import simpy

from job import Job


class Cloud:
    MODULE = "Cloud"

    def __init__(self, env: simpy.Environment,
                 # latencies
                 latency_roundtrip_ms = 20,
                 machine_speed=1.0,
                 # jobs
                 job_duration_type = Job.DurationType.GAUSSIAN):
        self._env = env
        self._latency_roundtrip_ms = latency_roundtrip_ms
        self._job_duration = job_duration_type
        self._machine_speed = machine_speed
        self._job_duration_type = job_duration_type
        self._service_discovery = None

        self._jobs_list = {}

    #
    # Callbacks
    #

    def _after_transmission_clb(self, event):
        job = self._jobs_list[event.value]
        actual_job_duration = Job.compute_duration(job, self._job_duration_type, self._machine_speed)

        job.a_executed(actual_job_duration)

        # simulate execution with no queue
        my_timeout = simpy.events.Timeout(self._env, actual_job_duration, job.get_uid())
        my_timeout.callbacks.append(self._after_execution_clb)

    def _after_execution_clb(self, event):
        # pop item from dict
        job = self._jobs_list.pop(event.value)

        # re-transmit back to the originator node (scheduler) or cluster where it had been forwarded
        if job.get_forwarded_to_cluster_id() < 0:
            originator_node = self._service_discovery.get_node_by_uid(job.get_originator_node_uid())
            originator_node.return_job_from_cloud(job)
        else:
            if job.get_forwarded_to_cluster_id() == 3:
                scheduler_node = self._service_discovery.get_node_up_scheduler_for_cluster_id(
                    job.get_forwarded_to_cluster_id())
            else:
                scheduler_node = self._service_discovery.get_node_scheduler_for_cluster_id(
                    job.get_forwarded_to_cluster_id())
            scheduler_node.return_job_from_cloud(job)

    #
    # Exported
    #

    def set_service_discovery(self, service_discovery):
        self._service_discovery = service_discovery

    def schedule_job(self, job: Job, level):
        self._jobs_list[job.get_uid()] = job
        # print(f"job = {job.get_uid()},  cloud --------> level = {level}")
        # simulate transmission latency with no queue
        my_timeout = simpy.events.Timeout(self._env, self._latency_roundtrip_ms / 1000, job.get_uid())
        my_timeout.callbacks.append(self._after_transmission_clb)
