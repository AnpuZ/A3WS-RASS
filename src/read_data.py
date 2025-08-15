# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import scipy.io as sio

class ReadData:

    def __init__(self, data_path, data_flag):
        self.data_path = data_path
        self.data_flag = data_flag

    def read_data(self):

        """
           Load data
        """
        if self.data_flag == 0:
            data = sio.loadmat(self.data_path)
            return self.load_datasets_mat(data)

        elif self.data_flag == 1:
            data = pd.read_csv(self.data_path, header=None)
            return self.load_datasets_csv(data.values)

        elif self.data_flag == 2:
            data = sio.loadmat(self.data_path)
            return self.load_datasets_mat_yang(data)

        elif self.data_flag == 3:
            data = pd.read_csv(self.data_path, header=None)
            return self.load_datasets_csv_draw(data)

        else:
            data = pd.read_excel(self.data_path, header=None)
            return self.load_datasets_csv(data)

    def load_datasets_mat(self, dataset):

        allData = dataset

        Data1 = allData['X']
        target = allData['Y']

        temp_scaler = MinMaxScaler(feature_range=(0, 1))
        Data = temp_scaler.fit_transform(Data1)

        set_label = -np.ones(target.shape[0])
        set_label = np.reshape(set_label, (len(set_label), 1))

        set_label[target == 1] = 1

        p_index = np.where(set_label == 1)[0][:]
        n_index = np.where(set_label != 1)[0][:]
        p_data = Data[p_index, :]
        n_data = Data[n_index, :]

        p_label = set_label[p_index]
        n_label = set_label[n_index]

        return p_data, n_data, p_label, n_label, Data, set_label

    def load_datasets_mat_yang(self, dataset):

        allData = dataset['crx']

        temp_scaler = MinMaxScaler(feature_range=(0, 1))
        Data = temp_scaler.fit_transform(allData[:, :-1])

        target = np.array(allData[:, -1].reshape(Data.shape[0], 1))
        set_label = -np.ones(target.shape[0])
        set_label = np.reshape(set_label, (len(set_label), 1))

        set_label[target == 1] = 1

        return Data, set_label


    def load_datasets_csv(self, dataset):

        dataset = np.array(dataset)

        temp_scaler = MinMaxScaler(feature_range=(0, 1))
        allData = temp_scaler.fit_transform(dataset[:, :-1])

        Data = allData
        target = np.array(dataset[:, -1]).reshape(Data.shape[0], 1)       

        return Data, target

    def load_datasets_csv_draw(self, dataset):

        dataset = np.array(dataset)

        temp_scaler = MinMaxScaler(feature_range=(0, 1))
        allData = temp_scaler.fit_transform(dataset[:, :-1])

        Data = allData
        target = np.array(dataset[:, -1]).reshape(Data.shape[0], 1)

        set_label = -np.ones(target.shape[0])
        set_label = np.reshape(set_label, (len(set_label), 1))

        set_label[target == 1] = 1

        p_index = np.where(set_label == 1)[0][:]
        n_index = np.where(set_label != 1)[0][:]
        p_data = Data[p_index, :]
        n_data = Data[n_index, :]

        p_label = set_label[p_index]
        n_label = set_label[n_index]

        return p_data, n_data, p_label, n_label, Data, set_label, n_index
