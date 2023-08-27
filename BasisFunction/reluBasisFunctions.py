"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from BasisFunction.basisFunctions import BasisFunctions
from scipy.stats import multivariate_normal,norm
from math import pi
import pickle
from numpy import mean,maximum,zeros,multiply,eye,transpose,array,zeros_like,add,concatenate,apply_along_axis
from numpy.random import uniform,seed
import numpy as np
import nengo


"""
    Class of random  Fourier bases
"""
class ReLUBasis(BasisFunctions):
    
    """
       Given basis functions configuration (BF_Setup), initialize parameters of
       basis functions.
    """
    def __init__(self, basis_setup): 
        #----------------------------------------------------------------------
        # Call supper class constructor
        super().__init__(basis_setup)
        
        #----------------------------------------------------------------------
        # Bandwidth of sampling distribution
        self.bandwidth              = basis_setup['basis_func_conf']['bandwidth']
        self.batch_size             = basis_setup['basis_func_conf']['batch_size']
        
        #----------------------------------------------------------------------
        # Initialize weights of basis functions
        self.intercept_list     = []
        self.theta_list         = []
        self.counter            = 0
        self.const              = self.bandwidth*((self.dim_state)**.5)
        
        
    def normalize_state(self,state):
        return  state / self.const
    
    def set_param(self,param):
        self.intercept_list,self.theta_list = param
    
    def get_param(self):
        
        return (self.intercept_list,self.theta_list)
    
    def sample_spherical(self,num_samples, dim):
        seed_   = self.basis_func_random_state + self.num_basis_func + self.counter
        X       = nengo.dists.UniformHypersphere(surface=True).sample(num_samples, dim+1,rng=np.random.RandomState(seed_)) #((self.dim_state)**.5)
        X_0     = X[:,0]
        X_1     = X[:,1:dim+1] 
        return X_0, X_1
        

    """
        Getter for samples of random bases.
    """     
    def sample(self, add_constant_basis = False):

        self.counter +=1
        intercept, theta = self.sample_spherical(self.batch_size,self.dim_state)
        if add_constant_basis:
            intercept[0] = 1.0
            theta[0]     = zeros(self.dim_state)
        
        return  intercept, theta
          

    """
        Please see the supper class.
    """
    def eval_basis(self,state,basis_param):  
        state = self.normalize_state(state)
        intercept_list, theta_list = basis_param
        return maximum(theta_list@state + intercept_list,0.0)


    """
       Compute the expected value of random bases on a batch of states.
       *** Remark: parameters of random bases are fixed.
    """
    def expected_basis(self,state_list,basis_param):
        intercept_list, theta_list = basis_param
        
        state_list = apply_along_axis(self.normalize_state, 1, state_list)
        
        return mean(maximum(add(state_list@theta_list.T,intercept_list),0.0),0)

    def eval_basis_list(self,state_list,basis_param):
        
        state_list = apply_along_axis(self.normalize_state, 1, state_list)
        intercept_list, theta_list = basis_param
        return maximum(add(state_list@theta_list.T, intercept_list),0.0)

    def concat(self, basis_coef_1, basis_coef_2):
        intercept_list_1,theta_list_1 = basis_coef_1
        intercept_list_2,theta_list_2 = basis_coef_2
        
        if intercept_list_1 == []:
            return basis_coef_2
        else:
            return ((concatenate((intercept_list_1,intercept_list_2)),
                concatenate((theta_list_1,theta_list_2))))
    
    """
        Please see the supper class.
    """    
    def get_VFA(self,state,coef = None):
        
        state = self.normalize_state(state)
        
        if coef is None:
            return maximum(self.theta_list@state + self.intercept_list,0.0)@self.opt_coef
        
        else:
            return maximum(self.theta_list@state + self.intercept_list,0.0)@coef

    def get_expected_VFA(self,state_list):
        state_list = apply_along_axis(self.normalize_state, 1, state_list)
        return mean(maximum(add(state_list@ self.theta_list.T, self.intercept_list),0.0),0)@self.opt_coef
    
 