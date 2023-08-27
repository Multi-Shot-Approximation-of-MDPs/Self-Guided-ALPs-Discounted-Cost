# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from BasisFunction.basisFunctions import BasisFunctions
from scipy.stats import multivariate_normal
from math import pi
from numpy import mean,cos,zeros,multiply,eye,add,concatenate
from numpy.random import uniform,seed
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
import numpy as np


@jit(nopython=True,nogil=True)
def get_VFA(theta_list,intercept_list,opt_coef,state):
    #----------------------------------------------------------------------
    # This function computes VFA at a state for a given set of optimal 
    # weights for these bases obtained from an ALP model and a set of
    # sampled random basis parameters.
    #----------------------------------------------------------------------
    
    return cos(theta_list@state+intercept_list)@opt_coef


@jit(nopython=True,nogil=True,parallel=False,fastmath=True)
def get_expected_VFA(theta_list,intercept_list,opt_coef,state_list):
    #----------------------------------------------------------------------
    # This function computes VFA at a list of states for a given set of 
    # optimal weights for these bases obtained from an ALP model and a set 
    # of sampled random basis parameters.
    #----------------------------------------------------------------------
    VFA = cos(theta_list@state_list[0]+intercept_list)
    num_samples = len(state_list)
    for _ in range(1,num_samples):
        VFA += cos(theta_list@state_list[_]+intercept_list)
    return (VFA@opt_coef)/len(state_list)


class FourierBasis(BasisFunctions):

    def __init__(self, basis_setup): 
        #----------------------------------------------------------------------
        # This is constructor for Fourier basis functions object
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        # Call supper class constructor
        super().__init__(basis_setup)
        
        #----------------------------------------------------------------------
        # Bandwidth of sampling distribution
        self.bandwidth          = basis_setup['basis_func_conf']['bandwidth']
        
        #----------------------------------------------------------------------
        # Sampling batch size
        self.batch_size         = basis_setup['basis_func_conf']['batch_size']
    
        #----------------------------------------------------------------------
        # Initialize weights of basis functions
        self.intercept_list     = None
        self.theta_list         = None

        
    def set_param(self,param):
        #----------------------------------------------------------------------
        # This function sets params of random bases using based on an input
        #----------------------------------------------------------------------
        self.intercept_list,self.theta_list = param
        

    def sample(self, add_constant_basis = False):        
        #----------------------------------------------------------------------
        # This function samples a batch of random bases. If add_constant_basis,
        # then it constructs a random basis that evaluates to one at all states
        #----------------------------------------------------------------------
              
        #----------------------------------------------------------------------
        # phi(s;(intercept,theta)) = cos(s.theta + intercept) where:
        #     1) intercept ~ unif[-pi,pi]
        #     2) intercept ~ MVN(0,bandwidth*I)
        #----------------------------------------------------------------------
        seed_ = self.basis_func_random_state+self.num_basis_func
        seed(seed_)
        intercept = uniform(low=-pi,high=pi,size = self.batch_size)
        theta = []
        
        #----------------------------------------------------------------------
        # Sample a batch of parameters
        #----------------------------------------------------------------------
        for  _ in range(self.batch_size):
            seed_ = self.basis_func_random_state+self.num_basis_func + _
            seed(seed_)
            bandwidth = np.random.choice(self.bandwidth)
            theta.append(multivariate_normal( mean = zeros(shape=self.dim_state),
                                              cov  = multiply(2*bandwidth, eye(self.dim_state))).rvs(size=1,random_state=seed_) )
        
        theta = np.asarray(theta)
        if add_constant_basis:
            intercept[0] = 0
            theta[0]     = zeros(self.dim_state)

        return  intercept, theta
          

    def eval_basis(self,state,basis_param):  
        #----------------------------------------------------------------------
        # This function returns the evaluation of multiple random basis 
        # functions on a particular state.
        #----------------------------------------------------------------------
        intercept_list, theta_list = basis_param
        return cos(theta_list@state+intercept_list)
        
    
    def expected_basis(self,state_list,basis_param):
        #----------------------------------------------------------------------
        # This function returns the expected value of random bases on a batch 
        # of states.
        #----------------------------------------------------------------------
        intercept_list, theta_list = basis_param
        return mean(cos(add(state_list@theta_list.T,intercept_list)),0)



    def concat(self, basis_coef_1, basis_coef_2):
        #----------------------------------------------------------------------
        # Concat parameters of random bases with parameters 
        # basis_coef_1 nd basis_coef_2
        #----------------------------------------------------------------------
        intercept_list_1,theta_list_1 = basis_coef_1
        intercept_list_2,theta_list_2 = basis_coef_2
    
        return ((concatenate((intercept_list_1,intercept_list_2)),
                concatenate((theta_list_1,theta_list_2))))
    

    def get_VFA(self,state,coef = None):
        #----------------------------------------------------------------------
        # Please see the corresponding jited functions in preamble
        #----------------------------------------------------------------------
        if coef is None:
            return get_VFA(self.theta_list,self.intercept_list,self.opt_coef, state)
        else:
            return get_VFA(self.theta_list,self.intercept_list, coef, state)
        

    def get_expected_VFA(self,state_list):
        #----------------------------------------------------------------------
        # Please see the corresponding jited functions in preamble
        #----------------------------------------------------------------------
        return get_expected_VFA(self.theta_list, self.intercept_list, self.opt_coef, state_list)
 
