# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:08:47 2020

@author: k.sloboda
"""

import pandas as pd
import random as rand
import numpy as np
import copy

from tkinter import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import asyncio
 

class FNN_TSK:
    def __init__(self, file_path, test_data_percent, rules, epochs, cycles, eta, error_threshold):
        self.file_path = file_path
        self.test_data_percent = test_data_percent
        self.rules = rules
        self.epochs = epochs
        self.cycles = cycles
        self.eta = eta
        self.error_threshold = error_threshold
        
        self.init_train_and_test_data()
        
        self.rows = len(self.train_data)
        self.props = len(self.train_data[0]) - 1
        
        # B, C, Sigma - MF parameters
        self.init_B_C_Sigma()
        
        # P - linear weights pkj, k - rules num, j - props num
        self.P = []
        
        # Y_train - predicted output value, D_train - real output value
        self.D_train = [row[-1] for row in self.train_data]
        self.Y_train = []
        self.train_errors = []
        
        
    def init_train_and_test_data(self):
        df = pd.read_csv(self.file_path)
        self.input_data = df.values.tolist()
        
        class1 = [x for x in self.input_data if x[-1] == -1]
        class2 = [x for x in self.input_data if x[-1] == 1]
        
        rand.shuffle(class1)
        rand.shuffle(class2)
        
        test_class1 = class1[:round(class1.__len__() * self.test_data_percent * 0.01)]
        test_class2 = class2[:round(class2.__len__() * self.test_data_percent * 0.01)]
        
        train_class1 = class1[round(class1.__len__() * self.test_data_percent * 0.01):]
        train_class2 = class2[round(class2.__len__() * self.test_data_percent * 0.01):]
        
        self.test_data = list(test_class1 + test_class2)
        rand.shuffle(self.test_data)
        
        self.train_data = list(train_class1 + train_class2)
        rand.shuffle(self.train_data)
        
        print("test_data:", len(test_class1), len(test_class2), 
              "train_data:", len(train_class1), len(train_class2), 
              "shuffle:", len(self.test_data), len(self.train_data), "\n")
        
        
    def init_B_C_Sigma(self):
        props_array = [[row[p] for row in self.train_data] for p in range(self.props)]
        
        props_min_value = [min(props_array[p]) for p in range(self.props)]
        props_max_value = [max(props_array[p]) for p in range(self.props)]
        
        self.B = 1
        self.C = [[rand.uniform(props_min_value[j], props_max_value[j]) 
                   for j in range(self.props)]
                  for k in range(self.rules)]
                  
        self.Sigma = [[np.std(props_array[j]) 
                       for j in range(self.props)]
                      for k in range(self.rules)]
        
    
    def train_model(self):
        for epoch in range(self.epochs):
            print("epoch:", epoch)
            for cycle in range(self.cycles):
                    #print("\nepoch:", epoch, "row:", row, "cycle:", cycle)
                    yield int((epoch * self.cycles + cycle + 1) * 100 / 
                          (self.epochs * self.cycles))
                    
                    self.mf_array = [[[self.mf(self.train_data[row][j], self.B,
                                    self.C[k][j], self.Sigma[k][j]) 
                                 for j in range(self.props)]
                                for k in range(self.rules)]
                                for row in range(self.rows)]
                    
                    w_array = [[np.prod(self.mf_array[row][k])
                               for k in range(self.rules)]
                                for row in range(self.rows)]
                    
                    w_sum = [sum(w_array[row]) for row in range(self.rows)]
                    
                    w_1_array = [[w_array[row][k]/w_sum[row]
                                 for k in range(self.rules)]
                                for row in range(self.rows)]

                    #---------------------------------------------------------
                    
                    A_array = [[w_1_array[i][int(j / (self.props+1))] *
                                self.train_data[i][int(j % (self.props+1)) - 1] 
                                if j % (self.props+1) != 0
                                else w_1_array[i][int(j / (self.props+1))]
                                for j in range(self.rules * (self.props + 1))]
                               for i in range(self.rows)]
                    
                    if cycle == 0:
                        
                        A_pseudo_array = np.linalg.pinv(A_array)
                        
                        self.P = list(np.dot(A_pseudo_array, self.D_train))
                    
                    #---------------------------------------------------------
                    
                    self.Y_train = list(np.dot(A_array, self.P))
                    
                    self.train_error = sum([(self.Y_train[i] - self.D_train[i])**2 
                                            for i in range(self.rows)]) / 2
                    
                    #---------------------------------------------------------
                    
                    # if the error threshold is reached
                    if cycle == self.cycles - 1:  # Before recalculating the coefficients, so that you can add saving the model
                        self.train_errors.append(copy.deepcopy(self.train_error))
                        
                        if (len(self.train_errors) > 2 and
                            abs(self.train_error - self.train_errors[-2]) < self.error_threshold and
                            abs(self.train_error - self.train_errors[-3]) < self.error_threshold):
                            print("train_error:", self.train_errors[-1])
                            return
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.update_params())
                    loop.close() 
                                        
        print("train_error:", self.train_error)


    def mf(self, x, b, c, sigma):
        if sigma == 0:
            return 0
        return 1 / (1 + ((x - c) / sigma)**(2 * b))
        
    
    async def update_params(self):
        await asyncio.gather(self.update_C(),
                             self.update_Sigma())
                    
                    
    async def update_C(self):
        for row in range(self.rows):
            E_c = [[(self.Y_train[row] - self.D_train[row]) * 
                    sum([self.get_p_sum(row, r) *
                         self.dw_dc(j, k, r, row)
                         for r in range(self.rules)])
                    for j in range(self.props)]
                   for k in range(self.rules)]
                       
            self.C = [[self.C[k][j] - self.eta * E_c[k][j]
                       for j in range(self.props)]
                      for k in range(self.rules)]
        
        
    async def update_Sigma(self):
        for row in range(self.rows):
            E_sigma = [[(self.Y_train[row] - self.D_train[row]) * 
                        sum([self.get_p_sum(row, r) *
                             self.dw_dsigma(j, k, r, row)
                             for r in range(self.rules)])
                        for j in range(self.props)]
                       for k in range(self.rules)]
                        
            self.Sigma = [[self.Sigma[k][j] - self.eta * E_sigma[k][j]
                           for j in range(self.props)]
                          for k in range(self.rules)]
    
    
    def get_p_sum(self, row, r):
        return sum([self.P[r * self.props + p] 
                    if p % (self.props + 1) == 0 
                    else (self.P[r * self.props + p] * 
                          self.train_data[row][int(p % (self.props + 1)) - 1])
                    for p in range(self.props + 1)])


    def dw_dc(self, prop, k, r, row):
        dmf_dc = ((2*self.B * ((self.train_data[row][prop] - self.C[k][prop])/self.Sigma[k][prop])**(2*self.B)) /
                  ((self.train_data[row][prop] - self.C[k][prop]) * (1 + ((self.train_data[row][prop] - self.C[k][prop])/self.Sigma[k][prop])**(2*self.B))**2))
        
        mf_array_k_without_j = list(self.mf_array[row][k])
        del mf_array_k_without_j[prop]
        
        last_param_in_numerator = 0
        if k == r:
            mf_array_without_k = list(self.mf_array[row])
            del mf_array_without_k[k]
            last_param_in_numerator = sum([np.prod(mf_array_without_k[i])
                                           for i in range(len(mf_array_without_k))])
        else:
            last_param_in_numerator = np.prod(self.mf_array[row][r])
        
        numerator = (dmf_dc * np.prod(mf_array_k_without_j) * 
                     last_param_in_numerator)
        denominator = (sum([np.prod(self.mf_array[row][i]) 
                            for i in range(len(self.mf_array[row]))]))**2
        return numerator / denominator
                  
    
    def dw_dsigma(self, prop, k, r, row):
        dmf_dsigma = ((2*self.B * ((self.train_data[row][prop] - self.C[k][prop])/self.Sigma[k][prop])**(2*self.B)) /
                      (self.Sigma[k][prop] * (1 + ((self.train_data[row][prop] - self.C[k][prop])/self.Sigma[k][prop])**(2*self.B))**2))
        
        mf_array_k_without_j = list(self.mf_array[row][k])
        del mf_array_k_without_j[prop]
        
        last_param_in_numerator = 0
        if k == r:
            mf_array_without_k = list(self.mf_array[row])
            del mf_array_without_k[k]
            last_param_in_numerator = sum([np.prod(mf_array_without_k[i])
                                           for i in range(len(mf_array_without_k))])
        else:
            last_param_in_numerator = -np.prod(self.mf_array[row][r])
        
        numerator = (dmf_dsigma * np.prod(mf_array_k_without_j) * 
                     last_param_in_numerator)
        denominator = (sum([np.prod(self.mf_array[row][i]) 
                            for i in range(len(self.mf_array[row]))]))**2
        return numerator / denominator


    def test_model(self):
        self.D_test = [row[-1] for row in self.test_data]
        
        rows = self.test_data.__len__()
        
        mf_array = [[[self.mf(self.test_data[row][j], self.B,
                     self.C[k][j], self.Sigma[k][j]) 
                     for j in range(self.props)]
                    for k in range(self.rules)]
                    for row in range(rows)]
                
        w_array = [[np.prod(mf_array[row][k])
                   for k in range(self.rules)]
                    for row in range(rows)]
        
        w_sum = [sum(w_array[row]) for row in range(rows)]
        w_1_array = [[w_array[row][k]/w_sum[row]
                     for k in range(self.rules)]
                    for row in range(rows)]
                
        A_array = [[w_1_array[row][int(j / (self.props+1))] *
                   self.test_data[row][int(j % (self.props+1)) - 1] 
                   if j % (self.props+1) != 0
                   else w_1_array[row][int(j / (self.props+1))]
                   for j in range(self.rules * (self.props + 1))]
                   for row in range(rows)]
               
        self.Y_test = list(np.dot(A_array, self.P))
            
        self.test_error = sum([(self.Y_test[row] - self.D_test[row])**2 
                               for row in range(rows)]) / 2
        print("test_error:", self.test_error)






















    