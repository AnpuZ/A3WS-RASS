# -*- coding: utf-8 -*-
"""
Sampling
"""

import numpy as np
from Aisvdd import Aisvdd

class Sample:

    def __init__(self, ori_data, all_label, s1, class_k, label_list,eps_ol,delta_bound,gamma_bound):
        self.ori_data = ori_data
        self.all_label = all_label  
        self.s = s1
        self.class_k = class_k
        self.label_list = label_list
        self.eps_ol = eps_ol
        self.delta_bound = delta_bound
        self.gamma_bound = gamma_bound

    def find_all_data_sv(self):
        """
        Find support vector machines for all classes separately
        :return:
        """
        eps_ols = []
        delta_bounds = [] 
        gamma_bounds = []
        for i in range(self.class_k):
            eps_ols.append(self.eps_ol)
            delta_bounds.append(self.delta_bound)
            gamma_bounds.append(self.gamma_bound)   
        #eps_ols[1] = 1e-6 #Set according to the needs of minority classes
        #eps_ols[5] = 1e-10
        #delta_bounds[1] = - 0.1 
        #gamma_bounds[1] = - 0.1
        
        for i in range(self.class_k):
            class_k_index = np.where(self.all_label == self.label_list[i])[0][:]
            data_k = self.ori_data[class_k_index, :]
            a = np.std(data_k)
            s_k = (1 / (data_k.shape[1]*np.std(data_k))) 
            fd_k = Aisvdd(data_k, s_k, gamma_bounds[i], delta_bounds[i], eps_ols[i])
            fd_k.find_sv()
            fd_k._print_res()
            num_label_of_k = np.ones(fd_k.sv.shape[0]) * self.label_list[i]
            data_k_with_label = np.hstack((fd_k.sv, np.reshape(num_label_of_k, (len(num_label_of_k), 1))))
            if i == 0:
                other_data = data_k_with_label
                splicing_data = data_k_with_label
            else:
                splicing_data = np.vstack((data_k_with_label, other_data))
                other_data = splicing_data

        return splicing_data
    



   