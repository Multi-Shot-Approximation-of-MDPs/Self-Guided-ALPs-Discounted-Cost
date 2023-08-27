# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from BasisFunction.basisFunctions import BasisFunctions
import numpy as np
from numpy import array,maximum,ones,zeros

def eval_basis(state_list:np.ndarray, dist_stat:list):
    #--------------------------------------------------------------------------
    # Eval basis functions on a list of states
    #--------------------------------------------------------------------------
    num_stat            = len(dist_stat)   
    dim                 = len(state_list[0])
    num_bases           = (2*num_stat+1)*dim + 1 - num_stat
    num_samples         = len(state_list)
    evals               = zeros((num_samples, num_bases)) 
    
    # Constant term
    evals[:,0]          = ones(num_samples)
    
    # Linear term
    for _ in range(dim):
        evals[:,1 + _]  = state_list[:,_]
        
    # Bases ased on disposal cost
    for i in range(num_stat):
        for j in range(dim):
            evals[:, 1 + dim + i*dim + j] = maximum(0, sum(state_list[:,_] for _ in range(j+1)) - (j+1)*dist_stat[i])

    # Bases ased on lost sales cost
    for i in range(num_stat):
        for j in range(1,dim):
            evals[:, 1 + dim + num_stat*dim + i*(dim-1) + j -1] = maximum(0, j*dist_stat[i] - sum(state_list[:,_] for _ in range(dim - j - 1,dim)))
        
    return evals

class LNS_BasisFunction(BasisFunctions):
    #--------------------------------------------------------------------------
    # Basis functions  in http://dx.doi.org/10.1287/mnsc.1120.1551
    #--------------------------------------------------------------------------
    def __init__(self,basis_setup):
        #----------------------------------------------------------------------
        # Initialization
        #----------------------------------------------------------------------
        super().__init__(basis_setup)
        
        # Demand statistics used in LNS bases
        self.bandwidth              = basis_setup['basis_func_conf']['bandwidth']
        self.batch_size             = basis_setup['basis_func_conf']['batch_size']
        

    def sample(self, add_constant_basis = False):
        return None,None

    def set_param(self,param):
        pass

    def eval_basis(self,state,basis_param=None):     
        return eval_basis(array([state]),self.bandwidth).flatten()

    def eval_basis_list(self,state_list,basis_param=None):     
        return eval_basis(state_list,self.bandwidth)

    def expected_basis(self,state_list,basis_param):        
        return np.mean(eval_basis(state_list,self.bandwidth), axis = 0) 
    
    def get_VFA(self,state,coef = None):
    
        if coef is None:
            return (eval_basis(array([state]),self.bandwidth).flatten()) @ self.opt_coef
        else:
            return (eval_basis(array([state]),self.bandwidth).flatten()) @ coef
        
    
    def get_expected_VFA(self,state_list):
        state_list = array(state_list)
        return np.mean(eval_basis(state_list,self.bandwidth),axis=0) @ self.opt_coef
    
    
    
    