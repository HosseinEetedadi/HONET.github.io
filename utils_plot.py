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

import sqlite3


class UtilsPlot:

    @staticmethod
    def plot_data_average_client_fps_time(node_uid, db_path, job_type=0, average_every_secs=15, max_fps=30):
        db = sqlite3.connect(db_path)
        cur = db.cursor()
        x_rewards = []
        y_rewards = []

        res = cur.execute(
            f'''
select 
	cast(finish_time as int), count(*)
from (
	select 
		id, generated_at+time_total as finish_time, (generated_at+time_total) - LAG(generated_at+time_total, 1) OVER (ORDER BY generated_at) as lag_time 
	from 
		jobs where node_uid = {node_uid} and type = {job_type} and executed = 1 and rejected = 0
	) 
where lag_time > 0
group by cast(finish_time as int)
            ''')

        sum_reward = 0.0
        added = 0

        for line in res:
            t = line[0]
            reward = line[1]

            sum_reward += reward
            added += 1

            if t % average_every_secs == 0 and t > 0:
                # print(f"t={t}, added={added}, avg={sum_reward / added}")
                x_rewards.append(t)
                y_rewards.append(sum_reward / added)
                added = 0
                sum_reward = 0.0

        cur.close()
        db.close()

        return x_rewards, y_rewards

    @staticmethod
    def plot_data_average_lag_over_time(node_uid, db_path, job_type=0, average_every_secs=15, max_fps=30):
        db = sqlite3.connect(db_path)
        cur = db.cursor()
        x_rewards = []
        y_rewards = []

        res = cur.execute(
            f'''
select 
	cast(finish_time as int), avg(time_total)-{1/max_fps:.3f}
from (
	select 
		id, time_total, generated_at+time_total as finish_time, (generated_at+time_total) - LAG(generated_at+time_total, 1) OVER (ORDER BY generated_at) as lag_time 
	from 
		jobs where node_uid = {node_uid} and type = {job_type} and executed = 1 and rejected = 0
	) 
where lag_time > 0
group by cast(finish_time as int)
''')

        sum_reward = 0.0
        added = 0
        for line in res:
            t = line[0]
            reward = max(0, line[1])

            sum_reward += reward*1000
            added += 1

            if t % average_every_secs == 0 and t > 0:
                # print(f"t={t}, added={added}, avg={sum_reward / added}")
                x_rewards.append(t)
                y_rewards.append(sum_reward / added)
                added = 0
                sum_reward = 0.0

        cur.close()
        db.close()

        return x_rewards, y_rewards

    @staticmethod
    def plot_data_average_response_time_over_time(node_uid, db_path, job_type=0, average_every_secs=15, max_fps=30):
        db = sqlite3.connect(db_path)
        cur = db.cursor()
        x_rewards = []
        y_rewards = []

        res = cur.execute(
            f'''
select 
	cast(finish_time as int), avg(time_total)
from (
	select 
		id, time_total, generated_at+time_total as finish_time, (generated_at+time_total) - LAG(generated_at+time_total, 1) OVER (ORDER BY generated_at) as lag_time 
	from 
		jobs where node_uid = {node_uid} and type = {job_type} and executed = 1 and rejected = 0
	) 
where lag_time > 0
group by cast(finish_time as int)
            ''')
        sum_reward = 0.0
        added = 0
        for line in res:
            t = line[0]
            reward = line[1] # = min(max_fps, line[1])

            sum_reward += reward*1000
            added += 1

            if t % average_every_secs == 0 and t > 0:
                # print(f"t={t}, added={added}, avg={sum_reward / added}")
                x_rewards.append(t)
                y_rewards.append(sum_reward / added)
                added = 0
                sum_reward = 0.0

        cur.close()
        db.close()

        return x_rewards, y_rewards

    @staticmethod
    def plot_data_reward_over_time(node_uid, db_path, average_every_secs=15):
        db = sqlite3.connect(db_path)
        cur = db.cursor()
        x_rewards = []
        y_rewards = []

        res = cur.execute(
            f"select cast(generated_at as integer), sum(reward) from jobs where node_uid = {node_uid} group by cast(generated_at as integer)")

        sum_reward = 0.0
        added = 0
        for line in res:
            t = line[0]
            reward = line[1]

            sum_reward += reward
            added += 1

            if t % average_every_secs == 0 and t > 0:
                # print(f"t={t}, added={added}, avg={sum_reward / added}")
                x_rewards.append(t)
                y_rewards.append(sum_reward / added)
                added = 0
                sum_reward = 0.0

        cur.close()
        db.close()

        return x_rewards, y_rewards

    @staticmethod
    def plot_data_in_deadline_over_time(node_uid, db_path, average_every_secs=15):
        db = sqlite3.connect(db_path)
        cur = db.cursor()
        x_rewards = []
        y_rewards = []

        res = cur.execute(
            f"select cast(generated_at as integer), 1.0-(cast(sum(over_deadline) as float)/count(*)) from jobs where node_uid = {node_uid} and executed = 1 and rejected = 0 group by cast(generated_at as integer)")

        sum_reward = 0.0
        added = 0
        for line in res:
            t = line[0]
            reward = line[1]

            sum_reward += reward
            added += 1

            if t % average_every_secs == 0 and t > 0:
                # print(f"t={t}, added={added}, avg={sum_reward / added}")
                x_rewards.append(t)
                y_rewards.append(sum_reward / added)
                added = 0
                sum_reward = 0.0

        cur.close()
        db.close()

        return x_rewards, y_rewards
