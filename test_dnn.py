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

import numpy

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

NODES = 20

model = Sequential()
model.add(Dense(NODES + 10, input_dim=NODES, activation='relu'))
model.add(Dense(NODES + 10, activation='relu'))
model.add(Dense(NODES + 10, activation='relu'))
model.add(Dense(NODES, activation='linear'))
# model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

state = [i for i in range(NODES)]

print(state)
print(len(state))


def prepare_input(array):
    new_state = array.copy()
    new_state.append(1)
    new_state.pop(0)
    return numpy.array([new_state])


print(model.predict(numpy.array([state])))
print()
print(model.predict(numpy.reshape(state, [1, NODES])))
print()

for i in range(1000):
    print(model.predict(prepare_input(state)))
