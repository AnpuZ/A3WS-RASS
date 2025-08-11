# -*- coding: utf-8 -*-
"""
Heuristic attribute and scale selection algorithm
"""

#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.feature_selection import SelectFromModel
import numpy as np
import relation_matrix as relation_matrix
from collections import Counter
#from imblearn.over_sampling import RandomOverSampler
#import pandas
#import time
import copy


class Select:

    def __init__(self, ms_partial_data, partial_labels, ms_whole_data, scales):
        self.ms_partial_data = ms_partial_data
        self.partial_labels = partial_labels
        self.ms_whole_data = ms_whole_data
        self.scales = scales 

    def get_attribute_importance_theta(self, para_radius):
        only_data = self.ms_partial_data[:, :]
        only_label = self.partial_labels       
        self.temp_label_dict = Counter(only_label)
        self.temp_label_list = list(self.temp_label_dict)
        attribute_left = list(np.arange(0, len(self.scales), 1))        
        attribute_select = []
        scale_select = []
        start = 1
        # Record last round self information
        current_attribute_relation = np.ones((only_data.shape[0], only_data.shape[0]))
        relation_list = relation_matrix.GetRelationMatrix(only_data, only_label, self.scales, para_radius).relation_matrix_1()  
        scale_list = copy.deepcopy(self.scales)
        while start:
            k = 0
            each_attribute_information = []
            each_attribute_scale = []

            while k < len(attribute_left):
                each_scale_information = []
                for i in range(scale_list[k]): 
                    self_information = 0
                    temp_lower_neighbor_dict = dict.fromkeys(self.temp_label_list, 0)
                    temp_upper_neighbor_dict = dict.fromkeys(self.temp_label_list, 0)

                    if len(attribute_select) == 0:                        
                        array_relation = relation_list[k][i]
                    else:
                        array_relation = np.minimum(current_attribute_relation, relation_list[k][i])

                    for j in range(only_data.shape[0]):
                        neighbor_index_j = np.where(array_relation[j, :] == 1)[0][:]
                        neighbor_label_set = set(only_label[neighbor_index_j])
                        if len(neighbor_label_set) == 1:
                            temp_lower_neighbor_dict[list(neighbor_label_set)[0]] += 1
                            temp_upper_neighbor_dict[list(neighbor_label_set)[0]] += 1
                        else:
                            for t in range(len(neighbor_label_set)):
                                temp_upper_neighbor_dict[list(neighbor_label_set)[t]] += 1

                    for j in range(len(self.temp_label_list)):
                        temp_num_class_j = self.temp_label_dict[self.temp_label_list[j]]
                        temp_upper_approx = temp_upper_neighbor_dict[self.temp_label_list[j]]
                        temp_lower_approx = temp_lower_neighbor_dict[self.temp_label_list[j]]

                        if temp_lower_approx == 0:
                            temp_lower_approx = 1 / temp_num_class_j

                        precision = temp_lower_approx / temp_upper_approx
                        self_information = self_information + (-(1 - precision) * np.log(precision))
                    each_scale_information.append(self_information)
                scale_position = np.argmin(each_scale_information)
                each_attribute_information.append(each_scale_information[scale_position])
                each_attribute_scale.append(scale_position)
                k = k + 1
            
            if k == 0:
                start = 0
            else:
                position = np.argmin(each_attribute_information)
                new_min_information = each_attribute_information[position]
                if len(attribute_select) == 0:
                    attribute_select.append(attribute_left[position])
                    scale_select.append(each_attribute_scale[position])
                    old_min_information = new_min_information 
                    current_attribute_relation = np.minimum(current_attribute_relation, relation_list[position][each_attribute_scale[position]])
                    attribute_left.pop(position)
                    scale_list.pop(position)
                    relation_list.pop(position)
                else:
                    similarity = old_min_information - new_min_information
                    if similarity > 0.001:
                        attribute_select.append(attribute_left[position])
                        scale_select.append(each_attribute_scale[position])
                        old_min_information = new_min_information
                        current_attribute_relation = np.minimum(current_attribute_relation, relation_list[position][each_attribute_scale[position]])
                        attribute_left.pop(position)
                        scale_list.pop(position)
                        relation_list.pop(position)
                    else:
                        start = 0
            
        reduced_data = np.empty((self.ms_whole_data.shape[0], 0))
        for i in range(len(self.scales)):
            if i in attribute_select:
                n = attribute_select.index(i)
                index = 0                    
                for j in range(i):
                    index += self.scales[j]
                index += scale_select[n]
                reduced_data = np.append(reduced_data,self.ms_whole_data[:,index].reshape(self.ms_whole_data.shape[0], 1),axis=1)  
                
        return attribute_select, scale_select, reduced_data