# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from read_data import ReadData as rd
from multiscale_whole_data import GenerateMultiscaleWholeData as gmwd
from multiscale_partial_data import GenerateMultiscalePartialData as gmpd
import cross_validation as cr
from classify import Classify
from IS_reduce import Sample
from AS_reduce import Select
import time
from collections import Counter

path = r'..\data\Dry_Bean.csv'
condition_data, labels = rd(path, 1).read_data()
max_values = np.max(condition_data, axis=0)
min_values = np.min(condition_data, axis=0)
scale_flag = 1 #can be set 0 or 1
whole_data_flag = 2 #can be set 1 or 2
scales = gmwd(condition_data, scale_flag, whole_data_flag).generate_scales()
print("scales:", scales)  
RASS_whole_data = gmwd(condition_data, scale_flag, whole_data_flag).multiscale_data(scales) # RASS_whole_data doesn't contain labels 
class_dict = Counter(labels[:, 0])
class_list = []
for item in class_dict:
    class_list.append(item)  
class_k = len(class_list)
sigma = 0.6
k_folds = 10 
temp_train_index, temp_test_index = cr.cross_validation(RASS_whole_data.shape[0], k_folds)
temp_accuracy_comparison = []
temp_f1_score = []
temp_cost_time = []
radius = 0.025
eps_ol = 1e-1 
delta_bound = 0
gamma_bound = 0 #From radius to gamma_bound, their values can be set as needed
run_times = 0 
instance_lengths = []
selection_rates = []
classifying_effects = []

while run_times < 16: #The upper bound of run_times can change as needed 
    print("radius:",radius) #radius can change to be eps_ol, delta_bound or gamma_bound for parameter sensitivity analysis
    start_time1 = time.time()
    find_sv_data = Sample(condition_data, labels, sigma, class_k, class_list,eps_ol,delta_bound,gamma_bound).find_all_data_sv()
    RASS_partial_data, partial_labels = gmpd(find_sv_data, min_values, max_values, whole_data_flag).multiscale_data(scales)
    time1 = time.time() - start_time1
    print("sample_time:",time1)
    instance_length = len(RASS_partial_data[:, 0])
    instance_lengths.append(instance_length)
    temp_accuracy_selected_comparison = []
    temp_f1_selected_score = []
    temp_selected_cost_time = []
    start_time2 = time.time()  
    Selector = Select(RASS_partial_data, partial_labels, RASS_whole_data, scales)
    temp_select_attribute_index, temp_select_scale_index, temp_reduced_data = Selector.get_attribute_importance_theta(radius)
    time2 = time.time() - start_time2
    end_time = time2 + time1
    print("RASS time", end_time)    
    print("selected attributes:", temp_select_attribute_index)
    print("selected scales:", temp_select_scale_index)
    selected_scales_sum = sum(temp_select_scale_index) + len(temp_select_scale_index)
    selection_rate = selected_scales_sum/sum(scales)
    selection_rates.append(selection_rate)   
    for j in range(10):
        train_data = RASS_whole_data[temp_train_index[j]]
        train_label = labels[temp_train_index[j]]
        reduced_train_data = temp_reduced_data[temp_train_index[j]]
        test_data = RASS_whole_data[temp_test_index[j]]
        test_label = labels[temp_test_index[j]]
        reduced_test_data = temp_reduced_data[temp_test_index[j]]
        if run_times == 0:
           temp_accuracy_list_original, temp_f1_score_list_original, temp_cost_time_original = Classify(train_data, train_label, test_data, test_label).classify()
           temp_accuracy_comparison.append(temp_accuracy_list_original)
           temp_f1_score.append(temp_f1_score_list_original)
           temp_cost_time.append(temp_cost_time_original)
        temp_accuracy_list_select, temp_f1_score_list_select, temp_cost_time_select = Classify(reduced_train_data, train_label, reduced_test_data, test_label).classify()
        temp_accuracy_selected_comparison.append(temp_accuracy_list_select)
        temp_f1_selected_score.append(temp_f1_score_list_select)
        temp_selected_cost_time.append(temp_cost_time_select)
    print("RASS classification ability:")
    select_classify_sum = 0
    for k in range(6):
        final_score_comparison = []
        final_f1_score = []
        final_cost_time = []
        final_score_selected_comparison = []
        final_f1_score_selected = []
        final_cost_time_selected = []
        for z in range(10):
            if run_times == 0:
               final_score_comparison.append(temp_accuracy_comparison[z][k])
               final_f1_score.append(temp_f1_score[z][k])
               final_cost_time.append(temp_cost_time[z][k])
            final_score_selected_comparison.append(temp_accuracy_selected_comparison[z][k])
            final_f1_score_selected.append(temp_f1_selected_score[z][k])
            final_cost_time_selected.append(temp_selected_cost_time[z][k])
        if run_times == 0:   
           score_test_comparison = sum(final_score_comparison) / 10
           temp_standard_comparison = np.std(final_score_comparison)
           f1 = sum(final_f1_score)/10
           temp_f1_standard = np.std(final_f1_score)
           average_time = sum(final_cost_time)/10
           print("C", k, "o acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$ ,time:$%0.9lf$" % (score_test_comparison,
                                                    temp_standard_comparison, f1, temp_f1_standard, average_time))
        score_test_selected_comparison = sum(final_score_selected_comparison) / 10
        temp_standard_selected_comparison = np.std(final_score_selected_comparison)
        f1_selected = sum(final_f1_score_selected) / 10
        temp_f1_sel_standard = np.std(final_f1_score_selected)
        average_time_selected = sum(final_cost_time_selected) / 10
        print("C", k, "s acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$ ,time:$%0.9lf$" % (score_test_selected_comparison,
                            temp_standard_selected_comparison, f1_selected, temp_f1_sel_standard, average_time_selected)) 
        select_classify_sum += score_test_selected_comparison + f1_selected 
    classifying_effects.append(select_classify_sum/12)
    radius += 0.025
    run_times += 1       
time2 = time.time() - start_time2
end_time = time2 + time1
print("instance_lengths:",instance_lengths)
print("select_classify_effects:",classifying_effects)
print("selection_rates:",selection_rates)

