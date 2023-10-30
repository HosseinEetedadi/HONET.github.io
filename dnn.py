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

import os
import sys
from datetime import datetime
from math import floor, log10

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
# import pandas as pd
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import data_utils

from log import Log


# print(tf.debugging.set_log_device_placement(True))


#
# https://datascienceplus.com/keras-regression-based-neural-networks/
#

class DNNQ:
    MODULE_NAME = "DNNQ"

    def __init__(self, n_nodes=30, batch_size=64, epochs=2, train_txt="", weights_path="", fitting=False):
        """Init the DNNQ
            n_nodes number of fog nodes
        """
        self._DIR_MODELS = "dnnq"
        self._DIR_TB_LOG = "dnnq/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(self._DIR_MODELS, exist_ok=True)
        os.makedirs(self._DIR_TB_LOG, exist_ok=True)

        self._n_nodes = n_nodes
        """Number of nodes, number of input neurons for state and output neurons"""
        self._train_txt = train_txt
        """Path of the data file for training"""
        self._epochs = epochs
        """Number of epoch"""
        self._batch_size = batch_size
        """Number of samples per batch"""
        self._update_ref_dnn_batch_freq = 10
        """Batches after which the reference DNN is reloaded with updated weights"""
        self._rate_learning = 0.4

        # keras pre-model configuration
        if fitting is False:
            tf.keras.backend.set_learning_phase(0)

        # init the model
        self._model = Sequential()
        self._model.add(Dense(self._n_nodes + 10, input_dim=self._n_nodes, activation='relu'))
        # self._model.add(Dense(self._n_nodes + 10, activation='relu'))
        self._model.add(Dense(self._n_nodes + 10, activation='relu'))
        self._model.add(Dense(self._n_nodes, activation='linear'))
        # self._model.summary()
        self._model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
        # self._model.compile(loss=self._huber_loss, optimizer=Adam(lr=self._rate_learning), metrics=['mse', 'mae'])

        # load weights if requested
        if len(weights_path) > 0:
            self._model.load_weights(weights_path)

        # tensorboard callback
        self._tb_callback = TensorBoard(log_dir=self._DIR_TB_LOG, histogram_freq=1, update_freq="batch",
                                        profile_batch=0)

    def fit(self):
        latest_weights_path = self._DIR_MODELS + "/dnnq-" + str(self._n_nodes) + ".latest.hdf5"
        periodic_weights_path = self._DIR_MODELS + "/dnnq-" + str(
            self._n_nodes) + ".weights.{epoch:02d}-{loss:.4f}.hdf5"

        # create the data generators
        train_generator = DataGenerator(self._train_txt, self._n_nodes, to_fit=True, batch_size=self._batch_size,
                                        ref_dnn_update_batch_freq=self._update_ref_dnn_batch_freq,
                                        ref_dnn_latest_weights_path=latest_weights_path)
        test_generator = DataGenerator(self._train_txt, self._n_nodes, to_fit=True, batch_size=self._batch_size,
                                       ref_dnn_update_batch_freq=self._update_ref_dnn_batch_freq,
                                       ref_dnn_latest_weights_path=latest_weights_path)

        # save weights callback, since the ref dnn is updated every _update_ref_dnn_batch_freq batches here we save the
        # model one batch before
        save_freq = (self._update_ref_dnn_batch_freq - 1) * self._batch_size

        model_checkpoint_l = ModelCheckpoint(latest_weights_path, save_weights_only=True, save_freq=save_freq)
        model_checkpoint = ModelCheckpoint(periodic_weights_path, save_freq=20 * self._batch_size,
                                           save_weights_only=True)

        # fit the model
        self._model.fit(train_generator, epochs=self._epochs, verbose=1, validation_data=test_generator,
                        callbacks=[model_checkpoint, model_checkpoint_l, self._tb_callback])

    def predict(self, states):
        input_array = np.array([states])
        return self._model.predict(input_array)

    def _save_weights(self):
        self._model.save_weights("{}/model_{}.h5".format(self._DIR_MODELS, self._n_nodes))

    def _load_weight(self):
        self._model.load_weights("{}/model_{}.h5".format(self._DIR_MODELS, self._n_nodes))


class DataGenerator(data_utils.Sequence):
    MODULE_NAME = "DataGenerator"

    def __init__(self, data_file_path="", n_nodes=30, batch_size=32, ref_dnn_update_batch_freq=1000,
                 ref_dnn_latest_weights_path="", shuffle=True, to_fit=True):
        """Create a data generator creating batch of data from log file
        :param ref_dnn_update_batch_freq: after how many batches reload the ref dnnq
        :param ref_dnn_latest_weights_path: the path where to find the updated weights for the ref dnn
        """
        Log.minfo(DataGenerator.MODULE_NAME, "Initializing")
        self._data_file_path = data_file_path
        """File path of the data file"""
        self._batch_size = batch_size
        """Lines to put in a batch (m parameter)"""
        self._n_nodes = n_nodes
        """Number of nodes (n parameter)"""
        self._to_fit = to_fit
        """If data are used for fitting"""
        self._ref_dnn_update_batch_freq = ref_dnn_update_batch_freq
        """Number of batch after that the reference DNNQ will be reloaded with new weights"""
        self._total_requested_batches = 0
        """Total number of requested batches"""
        self._ref_dnn_latest_weights_path = ref_dnn_latest_weights_path
        """Path of the updated DNNQ weights"""
        self._rate_learning = 0.5
        self._rate_discount = 0.3
        self._k = 10
        """Max queue length"""

        # load the ref dnn
        self._ref_dnn = DNNQ(self._n_nodes)  # type: DNNQ
        """The reference DNNQ from which will be derived the Y array (nx1)"""

        # load metadata
        meta_file = open(self._data_file_path[:self._data_file_path.rfind(".")] + "-meta.txt")
        self._total_lines = int(meta_file.readlines()[0])
        meta_file.close()

        Log.minfo(DataGenerator.MODULE_NAME, "Total lines %d..." % self._total_lines)
        """Total available lines"""

        Log.minfo(DataGenerator.MODULE_NAME, "Generating list of ids %d..." % self._total_lines)
        self._indexes = np.arange(self._total_lines)
        """Range of line indexes"""

        # load train file
        self._train_file_fp = open(self._data_file_path, "r")
        self._train_file_line_size = self._get_line_size()

        self._shuffle = shuffle
        """If randomize the lines when picked for batches"""
        self.on_epoch_end()
        Log.minfo(DataGenerator.MODULE_NAME, "Initializing done")

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(self._total_lines / self._batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Decide if reload the ref dnnq
        if self._total_requested_batches > 0 and self._total_requested_batches % self._ref_dnn_update_batch_freq == 0:
            self._ref_dnn = DNNQ(self._n_nodes, weights_path=self._ref_dnn_latest_weights_path)

        # Generate indexes of the batch
        last_item_for_batch = (index + 1) * self._batch_size if (index + 1) * self._batch_size < len(
            self._indexes) else len(self._indexes) - 1
        indexes = self._indexes[index * self._batch_size: last_item_for_batch]

        # Find list of IDs
        data_ids_temp = indexes  # [self._indexes[k] for k in indexes]

        # Generate data
        x_arr = self._generate_x_arr(data_ids_temp)

        self._total_requested_batches += 1

        if self._to_fit:
            y_arr = self._generate_y_arr(data_ids_temp)
            Log.mdebug(self.MODULE_NAME,
                       "Generated batch%d from %d to %d" % (index, index * self._batch_size, last_item_for_batch))
            return x_arr, y_arr, [None]
        else:
            return x_arr

    def on_epoch_end(self):
        """Updates indexes after each epoch"""

        self._total_requested_batches = 0

        if self._shuffle:
            np.random.shuffle(self._indexes)

    def _generate_x_arr(self, data_ids_temp):
        """Generates data containing batch_size images (shape [batch_size, n])
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """

        # Initialization
        x_arr = np.empty((self._batch_size, self._n_nodes))

        for i, line_number in enumerate(data_ids_temp):
            # Log.minfo(DataGenerator.MODULE_NAME, "Generating x_arr: i=%d id=%d" % (i, line_number))
            line = self._get_data_file_line(line_number)

            components = line.split(" ")
            input_data = []  # the state of all nodes
            # parse input
            for j in range(self._n_nodes):
                # print("line=%d component=%s" % (line_number, components[j]))
                input_data.append(float(components[j]))

            x_arr[i,] = np.asarray(input_data)

        return x_arr

    def _generate_y_arr(self, data_ids_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """

        # Initialization
        y_arr = np.empty((self._batch_size, self._n_nodes))

        for i, line_number in enumerate(data_ids_temp):
            line = self._get_data_file_line(line_number)

            components = line.split(" ")
            input_data = []  # the state of all nodes
            target_data = []  # the expected values for all actions

            # parse input_data
            for j in range(self._n_nodes):
                input_data.append(float(components[j]))

            action = int(components[self._n_nodes])
            slack = float(components[self._n_nodes + 1])
            outcome = components[self._n_nodes + 2]
            reward = 0.0

            # generate reward
            if outcome == "A":
                reward = 100 * slack
            else:
                reward = -100

            # prediction = [[0.0 for i in range(self._n_nodes)]]  # 1x30
            # start_time = time.time()
            prediction = self._ref_dnn.predict(np.asarray(input_data))
            # print("--- %s seconds ---" % (time.time() - start_time))

            # compute the max of the q for next state same action
            max_next_state = 0.0
            if input_data[action] + 1 <= self._k:
                input_data[action] += 1
                prediction_next = self._ref_dnn.predict(np.asarray(input_data))
                max_next_state = prediction_next[0][np.argmax(prediction_next[0])]

            # generate expected value
            for j in range(self._n_nodes):
                if j == action:
                    target_data.append(reward + self._rate_discount * max_next_state)
                else:
                    target_data.append(prediction[0][j])

            y_arr[i,] = np.asarray(target_data)

        return y_arr

    #
    # Data file
    #

    def _get_data_file_line(self, line_number) -> str:
        """Get a line in the data file by seeking"""
        self._train_file_fp.seek(line_number * self._train_file_line_size)
        line = self._train_file_fp.read(self._train_file_line_size).strip()
        return line

    def _get_line_size(self):
        # create fixed width line
        needed_digits = floor(log10(self._n_nodes)) + 1
        return (needed_digits + 1) * (self._n_nodes + 1) + 20 + 2 + 1


def main(argv):
    dnnq = DNNQ(n_nodes=20, batch_size=512, train_txt="log/log-1586854194.txt", fitting=True)
    dnnq.fit()


if __name__ == "__main__":
    main(sys.argv)
