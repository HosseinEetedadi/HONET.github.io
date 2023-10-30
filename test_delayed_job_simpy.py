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

env = simpy.Environment()


def my_timeout_clb(event):
    print(f"[MyTimeout] triggered! e={event.value} e={event}")


def process_logging():
    my_timeout = simpy.events.Timeout(env, 45, 10)
    my_timeout.callbacks.append(my_timeout_clb)

    while True:
        yield env.timeout(1)
        print(f"[Logging] env.now={env.now}")


def init(env: simpy.Environment):
    env.process(process_logging())


def main():
    init(env)
    env.run(100)


if __name__ == '__main__':
    main()
