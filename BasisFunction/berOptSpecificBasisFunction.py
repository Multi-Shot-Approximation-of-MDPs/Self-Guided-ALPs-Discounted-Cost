# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from BasisFunction.basisFunctions import BasisFunctions
from numba import jit,prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


import numpy as np
from numpy import array


@jit(nopython=True,nogil=True)
def get_VFA(state,strike_price,opt_coef):
    #--------------------------------------------------------------------------
    # Get VFA based on the bases in http://dx.doi.org/10.1287/mnsc.1120.1551
    #--------------------------------------------------------------------------
    y = state[len(state)-1] 
    return  array([1 - y] + [max(max([state[_] for _ in range(len(state-1)) ]) - strike_price,0)*(1-y)] +  [(1-y)*state[_] for _ in range(len(state-1))])@opt_coef

@jit(nopython=True,nogil=True,parallel=False,fastmath=True)
def eval_basis(state_list,strike_price,opt_coef):
    #--------------------------------------------------------------------------
    # Get expected VFA based on several state
    #--------------------------------------------------------------------------
    dim         = len(state_list[0])
    num_samples = len(state_list)
    VFA         = 0
    
    for i in range(num_samples):
        y   = state_list[i][dim-1]
        VFA += array([1 - y] + [max(max([state_list[i][_] for _ in range(dim-1) ]) - strike_price,0)*(1-y)] +  [(1-y)*state_list[i][_] for _ in range(dim-1)])
    return (VFA@opt_coef)/len(state_list)


@jit(nopython=True,nogil=True,fastmath=True,parallel=True)
def eval_basis(state_list:np.ndarray,max_state_list:np.ndarray,strike_price,num_basis):
    #--------------------------------------------------------------------------
    # Eval basis functions on a list of states
    #--------------------------------------------------------------------------
    dim                 = len(state_list[0])
    num_samples         = len(state_list)
    evals               = np.zeros((num_samples,dim+1)) 
    evals[:,0]          = np.ones(num_samples) - state_list[:,dim-1]
    evals[:,1]          = np.multiply(np.maximum(max_state_list  - strike_price,
                                          np.zeros(num_samples)),evals[:,0] )
    
    for _ in prange(dim-1):
        evals[:,_+2]    = np.multiply(state_list[:,_],evals[:,0] )

    return evals


@jit(nopython=True,nogil=True,fastmath=True,parallel=True)
def mean_evals_on_inner_samples(state_matrix,num_basis_func,strike_price,num_ini_samples,num_inner_samples):

    VFA   = np.empty((num_ini_samples,num_basis_func))
    dim   = len(state_matrix[0,0,:])
    
    for i in prange(num_ini_samples):
        state_list          = state_matrix[i,:,:]
        sum_evals           = np.zeros(dim+1) 

        for j in prange(num_inner_samples):
            y                   = 1 - state_list[j,dim-1]   
            sum_evals[0]       += y
            sum_evals[1]       += max(max(state_list[j,:]) - strike_price, 0) * y
            for _ in prange(dim-1):
                sum_evals[_+2] += state_list[j,_] * y

        VFA[i,:] = sum_evals/num_inner_samples
    
    return VFA


@jit(nopython=True,nogil=True,fastmath=True,parallel=True)
def evals_on_inner_samples(state_matrix,num_basis_func,strike_price,num_ini_samples,num_inner_samples):

    VFA   = np.empty((num_ini_samples,num_inner_samples,num_basis_func))
    dim   = len(state_matrix[0,0,:])
    for i in prange(num_ini_samples):
        
        state_list          = state_matrix[i,:,:]
        for j in prange(num_inner_samples):
            y               = 1 - state_list[j,dim-1]
            VFA[i,j,0]      = y
            VFA[i,j,1]      = max(max(state_list[j,:]) - strike_price,0) * y
    
            for _ in prange(dim-1):
                VFA[i,j,_+2] = state_list[j,_] * y
    
    return VFA



@jit(nopython=True,nogil=True,parallel=True,fastmath=True)
def numba_mean(matrix:np.ndarray,num_init_states:int,num_basis:int): 
    #--------------------------------------------------------------------------
    # Jited version of taking mean of a matrix
    #--------------------------------------------------------------------------
    VFA = np.empty((num_init_states,num_basis))
    for i in prange(num_init_states):
        for j in range(num_basis):
            VFA[i,j] = np.mean(matrix[i,:,j])
        
    return VFA


class BerOptBasisFunction(BasisFunctions):
    #--------------------------------------------------------------------------
    # Basis functions  in http://dx.doi.org/10.1287/mnsc.1120.1551
    #--------------------------------------------------------------------------
    def __init__(self,basis_setup):
        #----------------------------------------------------------------------
        # Initialization
        #----------------------------------------------------------------------
        super().__init__(basis_setup)
        self.strike_price       = basis_setup['mdp_conf']['strike_price']
        self.num_basis_func     = basis_setup['mdp_conf']['num_asset']+2
        self.batch_size         = basis_setup['mdp_conf']['num_asset']+2
        self.bandwidth          = basis_setup['basis_func_conf']['bandwidth'] # its here just for saving results
        assert self.batch_size == self.max_basis_num, 'batch_size or/and max_basis_num are off for BerOptBasisFunction'

        self.max_basis_num      = self.batch_size
        self.preprocess_batch   = self.num_basis_func 
        
        
    def sample(self, add_constant_basis = False):
        return None,None


    def set_param(self,param):
        pass

    
    def compute_expected_basis_func(self,state_matrix,num_init_states,num_inner_samples,num_times_basis_added=None):
        return mean_evals_on_inner_samples(state_matrix,self.num_basis_func,self.strike_price,np.shape(state_matrix)[0],num_inner_samples)
        

    def eval_basis_func_on_inner_samples(self,state_matrix,num_init_states,num_inner_samples,num_times_basis_added=None):
        return evals_on_inner_samples(state_matrix,self.num_basis_func,self.strike_price,np.shape(state_matrix)[0],num_inner_samples)
        

    def list_expected_basis(self,state_matrix,basis_param=None):
        #----------------------------------------------------------------------
        # Compute expecd value od statse
        #----------------------------------------------------------------------
        (num_init_states,num_inner_samples,dim)     = np.shape(state_matrix)
        state_matrix                                = np.reshape(state_matrix, (num_init_states*num_inner_samples,dim))
        list_of_mean_basis                          = self.eval_basis(state_matrix,None)  
        list_of_mean_basis                          = np.reshape(list_of_mean_basis, (num_init_states,num_inner_samples,self.num_basis_func))
        list_of_mean_basis                          = numba_mean(list_of_mean_basis,num_init_states,self.num_basis_func)        
        return list_of_mean_basis
        
    
    def eval_basis(self,state_list,num_times_basis_added=None,all_bases=True):     
        return eval_basis(state_list,np.max(state_list,axis=1),self.strike_price,self.num_basis_func)
   
    

