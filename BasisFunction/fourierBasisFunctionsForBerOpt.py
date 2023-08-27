"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from BasisFunction.basisFunctions import BasisFunctions
from scipy.stats import multivariate_normal
from math import pi
from numpy import mean,empty,cos,zeros,multiply,eye,transpose,array,zeros_like,add,concatenate,matmul
from numpy.random import uniform,seed
from numba import jit,prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaPerformanceWarning
import warnings
import numpy as np
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)



@jit(nopython=True,nogil=True,parallel=True,fastmath=True)
def numba_mean(matrix:np.ndarray,num_init_states:int,num_basis:int): 
    VFA = empty((num_init_states,num_basis))
    for i in prange(num_init_states):
        for j in range(num_basis):
            VFA[i,j] = mean(matrix[i,:,j])
        
    return VFA
    
    
@jit(nopython=True,nogil=True,parallel=False,fastmath=True)
def list_basis_eval(theta_list:np.ndarray,intercept_list:np.ndarray,num_samples:int,price_list:np.ndarray,knocked_out:np.ndarray,num_basis:int):
    VFA   = empty((num_samples,num_basis))
    for _ in range(num_samples):
        VFA[_,:] = cos(theta_list @ price_list[_]+intercept_list)*knocked_out[_]
    return VFA 


@jit(nopython=True,nogil=True,parallel=True,fastmath=True)
def inner_sample_evals(theta_list:np.ndarray,intercept_list:np.ndarray,price_list:np.ndarray,knocked_out:np.ndarray,num_ini_samples:int,num_inner_samples:int,num_basis:int):
    
    VFA = np.empty((num_ini_samples,num_basis))
    
    for i in prange(num_ini_samples):  
        VFA[i,:] = cos(theta_list @ price_list[i,0,:] + intercept_list)*knocked_out[i,0]
        for j in range(1,num_inner_samples):
            VFA[i,:]   += cos(theta_list @ price_list[i,j,:]+intercept_list)*knocked_out[i,j]
          
        VFA[i,:] = VFA[i,:]/num_inner_samples
    return VFA


"""
    Class of random  Fourier bases
"""
class FourierBasisForBerOpt(BasisFunctions):
    

    def __init__(self, basis_setup): 
        #----------------------------------------------------------------------
        # Call supper class constructor
        super().__init__(basis_setup)
        
        #----------------------------------------------------------------------
        # Bandwidth of sampling distribution
        self.bandwidth      = basis_setup['basis_func_conf']['bandwidth']
        self.batch_size     = basis_setup['basis_func_conf']['batch_size']
        self.num_cpu_core   = basis_setup['basis_func_conf']['num_cpu_core']
        
        #----------------------------------------------------------------------
        # Initialize weights of basis functions
        self.intercept     = None
        self.theta         = None
        
        
        
        self.preprocess_batch = basis_setup['basis_func_conf']['preprocess_batch']
        
        
        #----------------------------------------------------------------------
        # Setup random bases sampling distribution

    
        
    def set_param(self,param):
        self.intercept,self.theta = param
        

    def list_basis_no_pre_cond(self,state_matrix,basis_param=None):

        intercept_list, theta_list                  = basis_param
        num_basis                                   = len(intercept_list)
        (num_states,dim)                            = np.shape(state_matrix)
        list_of_mean_basis                          = list_basis_eval(theta_list,
                                                                      intercept_list,
                                                                      num_states,
                                                                      state_matrix[:,0:dim-1],
                                                                      1.0 - state_matrix[:,dim-1],
                                                                      num_basis)
        
       
        
        return list_of_mean_basis
    
    
    
    
    def form_orthogonal_bases(self, state_list,path,stage,to_load=False):
        #--------------------------------------------------------------------------
        # Sample bases with pre-processing using PCA
        #--------------------------------------------------------------------------
        if not to_load:
            intercept       = []
            theta           = []

            for _ in range(self.preprocess_batch):
                seed_       = self.basis_func_random_state +  _
                seed(seed_)
                bandwidth   = np.random.choice(self.bandwidth)
                intercept.append(uniform(low=-pi,high=pi))   
                theta.append(multivariate_normal( mean = zeros(shape=self.dim_state-1),
                                                  cov  = multiply(2*bandwidth, eye(self.dim_state-1))).rvs(size=1,random_state=seed_) )
            
            #--------------------------------------------------------------------------
            # Perform PCA to orthogonalize random bases as a pre-processing step
            self.theta              = np.asarray(theta)
            self.intercept          = np.asarray(intercept)       
            basis_evals             = self.list_basis_no_pre_cond(state_list,(self.intercept, self.theta))
            eig_vals, eig_vect      = np.linalg.eig(basis_evals.T@basis_evals)
            eig_vals                = eig_vals.real
            idx                     = list(np.argsort(-np.abs(eig_vals)))
            eig_vect                = eig_vect[:,idx]
            thrivial_component      = np.zeros(self.preprocess_batch)
            thrivial_component[0]   = 1
            self.intercept[0]       = 0
            self.theta[0,:]         = zeros(self.dim_state-1)
            self.pre_cond           = np.append(np.array([thrivial_component]).T, eig_vect.real,axis=1)
            self.pre_cond           = self.pre_cond[:,:-1]
            
            
            np.savez_compressed(path + 'theta_stage_'+str(stage),self.theta,allow_pickle=False)
            np.savez_compressed(path + 'intercept_stage_'+str(stage),self.intercept,allow_pickle=False)
            np.savez_compressed(path + 'pre_cond_stage_'+str(stage),self.pre_cond,allow_pickle=False)
            
        else:
            self.theta      = np.load(path + 'theta_stage_'+str(stage)+ '.npz')
            self.theta      = self.theta.f.arr_0 
            self.intercept  = np.load(path + 'intercept_stage_'+str(stage) + '.npz')
            self.intercept  = self.intercept.f.arr_0
            self.pre_cond   = np.load(path + 'pre_cond_stage_'+str(stage)+ '.npz')
            self.pre_cond   = self.pre_cond.f.arr_0

            
            

    """
        Please see the supper class.
    """
    def eval_basis(self,state,num_times_basis_added,all_bases=False):
        if not all_bases:
            pre_cond                        = self.pre_cond[:, num_times_basis_added*self.batch_size:
                                                               (num_times_basis_added+1)*self.batch_size]

        else:
            pre_cond                        = self.pre_cond[:, 0: (num_times_basis_added+1)*self.batch_size]

        (num_init_states,dim)                       = np.shape(state)
        
        
        list_of_mean_basis                          = list_basis_eval(self.theta,
                                                                      self.intercept,
                                                                      num_init_states,
                                                                      state[:,0:dim-1],
                                                                      1.0 - state[:,dim-1],
                                                                      self.preprocess_batch
                                                                      )
        
        # list_of_mean_basis[0,:] = np.ones(len(list_of_mean_basis[0,:]))
        
        
        list_of_mean_basis                          = list_of_mean_basis@pre_cond 
            
        return list_of_mean_basis



    
    
    
    
    def compute_expected_basis_func(self,state_matrix,num_init_states,num_inner_samples,discount,stage,path,is_train):
        
        
        # if is_train:
        #     name = 'discounted_expected_VFA_train_batch_'+str(self.preprocess_batch) + '_stage_'+ str(stage)
        # else:
        #     name = 'discounted_expected_VFA_test_batch_' +str(self.preprocess_batch) + '_stage_'+ str(stage)
            
        
    
        basis_evals     = inner_sample_evals(theta_list         = self.theta,
                                            intercept_list      = self.intercept,
                                            price_list          = state_matrix[:,:,0:self.dim_state-1],
                                            knocked_out         = 1.0 - state_matrix[:,:, self.dim_state-1],  
                                            num_ini_samples     = num_init_states,
                                            num_inner_samples   = num_inner_samples,
                                            num_basis           = self.preprocess_batch
                                            )
    
        mean_evals     = (basis_evals@self.pre_cond)*discount 
        #np.savez_compressed(path+name, mean_evals,allow_pickle=False)
        
        return mean_evals
        
        
    def load_expected_basis(self,  num_times_basis_added, file,stage,is_train=True):
        #--------------------------------------------------------------------------
        # Please see the jited function above class.
        #--------------------------------------------------------------------------
        mean_evals      = file['expected_basis_func_'+str(stage)]
        if is_train:
            mean_evals      = mean_evals[:,num_times_basis_added*self.batch_size:
                                              (num_times_basis_added+1)*self.batch_size]  
        else:
            mean_evals      = mean_evals[:,0:(num_times_basis_added+1)*self.batch_size] 
               
        return mean_evals
            

    def concat(self, basis_coef_1, basis_coef_2):
        intercept_list_1,theta_list_1 = basis_coef_1
        intercept_list_2,theta_list_2 = basis_coef_2
    
        return ((concatenate((intercept_list_1,intercept_list_2)),
                concatenate((theta_list_1,theta_list_2))))
    
 
    def get_VFA(self,state,coef = None):
        pass
        # if coef is None:
        #     return get_VFA(self.theta,self.intercept,self.opt_coef,state)
        # else:
        #     return get_VFA(self.theta,self.intercept, coef,state)


    def get_expected_VFA(self,state_list):
        pass
        # return get_expected_VFA(self.theta,self.intercept,self.opt_coef,state_list)
 