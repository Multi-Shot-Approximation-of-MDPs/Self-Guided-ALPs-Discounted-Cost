"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from BasisFunction.basisFunctions import BasisFunctions
from scipy.stats import randint
import numpy as np
from numpy import sign, mean, add
from numpy.random import seed,uniform

def jited_sign(state,random_unit_vect,threshold_list):
    return sign(random_unit_vect @ state - threshold_list)


def mean_sign(state_list,random_unit_vect,threshold_list):
    return mean(sign(add(state_list @ random_unit_vect, -threshold_list)))
 

def get_VFA(state,random_unit_vect,threshold_list,opt_coef):
    #----------------------------------------------------------------------
    # This function computes VFA at a state for a given set of optimal 
    # weights for these bases obtained from an ALP model and a set of
    # sampled random basis parameters.
    #----------------------------------------------------------------------

    return sign(random_unit_vect @ state - threshold_list)@opt_coef

def get_expected_VFA(state_list,random_unit_vect,threshold_list,opt_coef):
    VFA = sign(random_unit_vect@state_list[0] - threshold_list)
    num_samples = len(state_list)
    for _ in range(1,num_samples):
        VFA += sign(random_unit_vect @ state_list[_] - threshold_list)
    return (VFA@opt_coef)/len(state_list)
        

"""
    Class of random stump bases
"""
class StumpBasis(BasisFunctions):
        
    """
        Given basis functions configuration (basis_setup), initialize parameters of
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
        self.counter                = 0
        self.threshold_list         = None
        self.index_list             = None
        self.random_unit_vect       = None


        self.basis_stat             = [[] for _ in range(self.dim_state)]


    def set_param(self,basis_param):
        self.index_list, self.threshold_list    = basis_param
        self.random_unit_vect                   = np.asarray([np.eye(1,self.dim_state,self.index_list[i]).flatten()
                                                      for i in range(self.num_basis_func)],dtype=float)
        
        

        

    def sample(self,add_constant_basis = False):
        self.counter    += 1
        seed_           =  self.basis_func_random_state + self.counter*self.batch_size
        index           =  []
        threshold       =  []
        
        for i in range(self.batch_size):
            seed(seed_ + i)
            coordinate      = randint(0,self.dim_state).rvs(random_state=seed_ + i)
            thr             = uniform(low=0.0,high=self.bandwidth[coordinate])
            index.append(coordinate)
            threshold.append(thr)

        return np.array(index), np.array(threshold)
    
    
    def eval_basis(self,state,basis_param):
        index_list, threshold_list          = basis_param
        random_unit_vect                    = np.asarray([np.eye(1,self.dim_state,index_list[i]).flatten()
                                                                  for i in range(len(index_list))], dtype=float)
        
        return jited_sign(state,random_unit_vect,threshold_list)   
       
        
    def expected_basis(self,state_list,basis_param):
        index_list, threshold_list          = basis_param
        random_unit_vect                    = np.asarray([np.eye(1,self.dim_state,index_list[i]).flatten()
                                                                  for i in range(len(index_list))],dtype=float).T

        return mean_sign(state_list, random_unit_vect, threshold_list)


    def concat(self, basis_coef_1, basis_coef_2):
        index_list_1, threshold_list_1 = basis_coef_1
        index_list_2, threshold_list_2 = basis_coef_2
    
        if threshold_list_1 == []:
            return basis_coef_2
        else:
            return ((np.concatenate((index_list_1,index_list_2)),
                         np.concatenate((threshold_list_1,threshold_list_2))))
        
        
    def get_VFA(self,state,coef = None):
        if coef is None:
            return get_VFA(state, self.random_unit_vect, self.threshold_list, self.opt_coef)
        else:
            return get_VFA(state, self.random_unit_vect, self.threshold_list, coef)


    def get_expected_VFA(self,state_list):
        return get_expected_VFA(state_list, self.random_unit_vect, self.threshold_list, self.opt_coef)



