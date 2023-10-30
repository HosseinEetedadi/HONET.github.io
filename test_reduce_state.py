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

from functools import reduce

MAX_JOBS_IN_QUEUE = 5
ACTIONS = [0, 1]

EPISODE = 1
UID = 0


def log_state_data(state):
    state_string = reduce(lambda x, y: str(x) + str(y), state)
    q_values = [0.0 for a in ACTIONS]
    for action, value in enumerate(q_values):
        print(f"Logging ({state_string}, {UID}, {EPISODE}, {action}, {value})")


# log to db
for i in range(MAX_JOBS_IN_QUEUE):
    for j in range(2):  # non-realtime / realtime
        log_state_data([i, j])

# test all states
job_types = 2
max_jobs_in_queue = 5


def list_rec(state_i, string):
    print(f"state_i={state_i} string={string}")
    out = []
    if state_i == job_types + 1:
        return string

    if state_i == job_types:
        for i in range(job_types):
            next_str = string + str(i)
            out.append(list_rec(state_i + 1, next_str))
        return out

    for i in range(max_jobs_in_queue):
        next_str = string + str(i)
        new_str_arr = list_rec(state_i + 1, next_str)

        # add to output only if states sum to max_jobs_in_queue
        for new_str in new_str_arr:
            state_sum = 0
            for j in range(job_types):
                state_sum += int(new_str[j])
            if state_sum <= max_jobs_in_queue:
                out.append(new_str)

    return out


all_states = list_rec(0, "")
print(all_states)
