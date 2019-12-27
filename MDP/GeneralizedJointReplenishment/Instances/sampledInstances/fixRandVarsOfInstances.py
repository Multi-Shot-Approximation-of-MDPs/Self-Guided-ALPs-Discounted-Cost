#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:59:54 2019

@author: parshan
"""

import numpy as np


def dropZeros(x,l):
             if abs(x) < l:
                 x = 0.0
             return x


def sampler():
    
    N = 3000
    
    miorCost        = np.zeros((N,10))
    consumptionRate = np.zeros((N,10))
    upBound         = np.zeros((N,10))
    rndCapacity     = np.zeros((N,10))
    discProb        = np.zeros((N,10))
    
    for i in range(N):
        for j in range(10):
            miorCost[i,j]           = dropZeros(np.random.uniform(low=0,high=60),1e-3) 
            consumptionRate[i,j]    = dropZeros(np.random.uniform(low=0,high=10),1e-3)
            upBound[i,j]            = dropZeros(np.random.uniform(low=0.0,high=1),1e-5)
            rndCapacity[i,j]        = dropZeros(np.random.uniform(low=0.0,high=1),1e-5)
            discProb[i,j]           = dropZeros(np.random.uniform(low=0.0,high=1),1e-5)
            
    np.savetxt('minorCost.csv',miorCost,delimiter=',')        
    np.savetxt('consumptionRate.csv',consumptionRate,delimiter=',')       
    np.savetxt('upBound.csv',upBound,delimiter=',')       
    np.savetxt('rndCapacity.csv',rndCapacity,delimiter=',')       
    np.savetxt('discProb.csv',discProb,delimiter=',') 

    

if __name__== "__main__":
    sampler()