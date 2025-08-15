# -*- coding: utf-8 -*-
"""
The generation of multi-scale data
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import sys

class GenerateMultiscalePartialData:

    def __init__(self, data, minValues, maxValues, data_flag):
        self.data = data
        self.minValues = minValues
        self.maxValues = maxValues        
        self.data_flag = data_flag  
        
    def multiscale_data(self, scales):
        sys.stdout.flush()
        if self.data_flag == 0:            
            return self.generate_partial_nominal_multiscaleData(scales)        
        elif self.data_flag == 1:            
            return self.generate_partial_hybrid_multiscaleData(scales)
        elif self.data_flag == 2:            
            return self.generate_partial_numeric_multiscaleData(scales)      
        
    def generate_partial_hybrid_multiscaleData(self, scales):
        dataset = np.array(self.data)    
        newDataset = np.empty((dataset.shape[0], 0))
        partitionNumbers = [5, 12, 30]
        target = dataset[:, -1]
        for i in range(dataset.shape[1]-1):
            scaleNumber = scales[i]-1                
            minValue = self.minValues[i]
            maxValue = self.maxValues[i]
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
        return newDataset, target    
   
    def generate_partial_numeric_multiscaleData(self, scales): 
        dataset = np.array(self.data)    
        newDataset = np.empty((dataset.shape[0], 0))
        partitionNumbers = [5, 10, 20]
        target = dataset[:, -1]
        for i in range(dataset.shape[1]-1):
            scaleNumber = scales[i]-1                
            minValue = self.minValues[i]
            maxValue = self.maxValues[i]
            tempColumn = np.zeros((dataset.shape[0],1))
            for j in range(scaleNumber):
                intervalLength = (maxValue - minValue) / partitionNumbers[j]
                for k in range(dataset.shape[0]):                        
                    if np.isnan(dataset[k,i]):
                        tempColumn[k] = (random.randint(partitionNumbers[j])+0.0) / (partitionNumbers[j]-1)
                    else:
                        for l in range(partitionNumbers[j]):
                            if dataset[k,i] <= minValue + (l+1)*intervalLength:
                                tempColumn[k] = (l+0.0) / (partitionNumbers[j]-1)
                                break
                            else:
                                continue
                newDataset = np.append(newDataset,tempColumn,axis=1)             
            newDataset = np.append(newDataset, dataset[:,i].reshape(-1, 1), axis=1)    
        return newDataset, target        
                 