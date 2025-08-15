# -*- coding: utf-8 -*-
"""
The generation of multi-scale data
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random
import sys

class GenerateMultiscaleWholeData:

    def __init__(self, data, scale_flag, data_flag):
        self.data = data      
        self.scale_flag = scale_flag
        self.data_flag = data_flag
        
    def generate_scales(self):
        scales = []
        for i in range(0, self.data.shape[1]):
            if self.scale_flag == 0:
                scales.append(random.randint(2, 4))
            else:
                scales.append(3)
        return scales
        
    def multiscale_data(self, scales):
        sys.stdout.flush()    
        if self.data_flag == 0:            
            return self.generate_nominal_multiscaleData(scales)        
        elif self.data_flag == 1:            
            return self.generate_hybrid_multiscaleData(scales)
        elif self.data_flag == 2:            
            return self.generate_numeric_multiscaleData(scales)      
        
    def generate_nominal_multiscaleData(self, scales):
        dataset = np.array(self.data)            
        newDataset = np.empty((dataset.shape[0], 0))
        partitionNumbers = [5, 10, 20, 40]          
        for i in range(dataset.shape[1]):
            scaleNumber = scales[i]         
            minValue = np.min(dataset[:,i])
            maxValue = np.max(dataset[:,i])
            tempColumn = np.zeros((dataset.shape[0],1), dtype=np.int)
            for j in range(scaleNumber):
                intervalLength = (maxValue - minValue) / partitionNumbers[j]
                for k in range(dataset.shape[0]):                        
                    if np.isnan(dataset[k,i]):
                        tempColumn[k] = random.randint(partitionNumbers[j])
                    else:
                        for l in range(partitionNumbers[j]):
                            if dataset[k,i] <= minValue + (l+1)*intervalLength:
                                tempColumn[k] = l
                                break
                            else:
                                continue                                
                newDataset = np.append(newDataset,tempColumn,axis=1)
        return newDataset        
                    
    def generate_hybrid_multiscaleData(self, scales):
        dataset = np.array(self.data)    
        newDataset = np.empty((dataset.shape[0], 0))
        partitionNumbers = [5, 10, 20]
        for i in range(dataset.shape[1]):
            scaleNumber = scales[i]-1                
            minValue = np.min(dataset[:,i])
            maxValue = np.max(dataset[:,i])
            tempColumn = np.zeros((dataset.shape[0],1), dtype=np.int)
            for j in range(scaleNumber):
                intervalLength = (maxValue - minValue) / partitionNumbers[j]
                for k in range(dataset.shape[0]):                        
                    if np.isnan(dataset[k,i]):
                        tempColumn[k] = random.randint(partitionNumbers[j])
                    else:
                        for l in range(partitionNumbers[j]):
                            if dataset[k,i] <= minValue + (l+1)*intervalLength:
                                tempColumn[k] = l
                                break
                            else:
                                continue
                newDataset = np.append(newDataset,tempColumn,axis=1)
            newDataset = np.append(newDataset, dataset[:,i].reshape(-1, 1), axis=1)               
        return newDataset    

    def generate_numeric_multiscaleData(self, scales):
        dataset = np.array(self.data)    
        newDataset = np.empty((dataset.shape[0], 0))
        partitionNumbers = [5, 10, 20]
        for i in range(dataset.shape[1]):
            scaleNumber = scales[i]-1                
            minValue = np.min(dataset[:,i])
            maxValue = np.max(dataset[:,i])
            tempColumn = np.zeros((dataset.shape[0],1), dtype=np.int)
            for j in range(scaleNumber):
                intervalLength = (maxValue - minValue) / partitionNumbers[j]
                for k in range(dataset.shape[0]):                        
                    if np.isnan(dataset[k,i]):
                        tempColumn[k] = random.randint(partitionNumbers[j])
                    else:
                        for l in range(partitionNumbers[j]):
                            if dataset[k,i] <= minValue + (l+1)*intervalLength:
                                tempColumn[k] = l
                                break
                            else:
                                continue
                temp_scaler = MinMaxScaler(feature_range=(0, 1))
                numericalData = temp_scaler.fit_transform(tempColumn.reshape(-1, 1))          
                newDataset = np.append(newDataset,numericalData,axis=1)            
            newDataset = np.append(newDataset, dataset[:,i].reshape(-1, 1), axis=1)   
        return newDataset      
                 