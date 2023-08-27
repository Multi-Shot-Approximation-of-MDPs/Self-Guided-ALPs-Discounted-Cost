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
from numpy import mean,maximum,zeros,multiply,eye,transpose,array,zeros_like,add,concatenate,apply_along_axis,empty
from numpy.random import uniform,seed
import numpy as np
import nengo


from numba import jit,prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)



    
@jit(nopython=True,nogil=True,parallel=False,fastmath=True)
def list_basis_eval(theta_list:np.ndarray,intercept_list:np.ndarray,num_samples:int,price_list:np.ndarray,knocked_out:np.ndarray,num_basis:int):
    VFA   = empty((num_samples,num_basis))
    for _ in range(num_samples):
        VFA[_,:] = maximum(theta_list @  price_list[_] +  intercept_list,0.0) * knocked_out[_]
    return VFA 

@jit(nopython=True,nogil=True,parallel=True,fastmath=True)
def inner_sample_evals(theta_list:np.ndarray,intercept_list:np.ndarray,price_list:np.ndarray,knocked_out:np.ndarray,num_ini_samples:int,num_inner_samples:int,num_basis:int):
    
    VFA = np.empty((num_ini_samples,num_basis))
    
    for i in prange(num_ini_samples):  
        VFA[i,:] = maximum(theta_list @  price_list[i,0,:] +  intercept_list,0.0) * knocked_out[i,0]
        
        for j in range(1,num_inner_samples):
            VFA[i,:]   += maximum(theta_list @  price_list[i,j,:] +  intercept_list,0.0) * knocked_out[i,j]
          
        VFA[i,:] = VFA[i,:]/num_inner_samples
    return VFA


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
        self.const              = None #self.bandwidth*((self.dim_state)**.5)
    
        self.preprocess_batch = basis_setup['basis_func_conf']['preprocess_batch']
        
    
    def set_normalizing_const(self,const):
        self.const = const
        
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
    
    def eval_basis_list(self,state_list,basis_param):
        
        knocked_out = 1.0 - state_list[:, self.dim_state-1]
        price_list  = state_list[:, 0:self.dim_state-1]
        price_list  = apply_along_axis(self.normalize_state, 1, price_list)

        intercept_list, theta_list = basis_param
        return (maximum(add(price_list @ theta_list.T, intercept_list),0.0).T * knocked_out).T
    
    def form_orthogonal_bases(self, state_list,path,stage,to_load=False):
        #--------------------------------------------------------------------------
        # Sample bases with pre-processing using PCA
        #--------------------------------------------------------------------------
        if not to_load:
            intercept, theta        = self.sample_spherical(self.preprocess_batch,self.dim_state-1)
            
            #--------------------------------------------------------------------------
            # Perform PCA to orthogonalize random bases as a pre-processing step
            self.theta              = np.asarray(theta)
            self.intercept          = np.asarray(intercept)
            basis_evals             = self.eval_basis_list(state_list,(self.intercept, self.theta))            
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
            
            
            np.savez_compressed(path + 'relu_theta_stage_'+str(stage),self.theta,allow_pickle=False)
            np.savez_compressed(path + 'relu_intercept_stage_'+str(stage),self.intercept,allow_pickle=False)
            np.savez_compressed(path + 'relu_pre_cond_stage_'+str(stage),self.pre_cond,allow_pickle=False)
             
        else:
            self.theta      = np.load(path + 'relu_theta_stage_'+str(stage)+ '.npz')
            self.theta      = self.theta.f.arr_0 
            self.intercept  = np.load(path + 'relu_intercept_stage_'+str(stage) + '.npz')
            self.intercept  = self.intercept.f.arr_0
            self.pre_cond   = np.load(path + 'relu_pre_cond_stage_'+str(stage)+ '.npz')
            self.pre_cond   = self.pre_cond.f.arr_0    
    
    
    
    
    def compute_expected_basis_func(self,state_matrix,num_init_states,num_inner_samples,discount,stage,path,is_train):
        
        basis_evals     = inner_sample_evals(theta_list         = self.theta,
                                            intercept_list      = self.intercept,
                                            price_list          = state_matrix[:,:,0:self.dim_state-1],
                                            knocked_out         = 1.0 - state_matrix[:,:, self.dim_state-1],  
                                            num_ini_samples     = num_init_states,
                                            num_inner_samples   = num_inner_samples,
                                            num_basis           = self.preprocess_batch
                                            )
    
        mean_evals     = (basis_evals@self.pre_cond)*discount 
        
        return mean_evals    
    
    
    
    
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

        list_of_mean_basis                          = list_of_mean_basis@pre_cond 
            
        return list_of_mean_basis    
    
    
    
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
    # def eval_basis(self,state,basis_param):  
    #     state = self.normalize_state(state)
    #     intercept_list, theta_list = basis_param
    #     return maximum(theta_list@state + intercept_list,0.0)


    """
       Compute the expected value of random bases on a batch of states.
       *** Remark: parameters of random bases are fixed.
    """
    def expected_basis(self,state_list,basis_param):
        intercept_list, theta_list = basis_param
        
        state_list = apply_along_axis(self.normalize_state, 1, state_list)
        
        return mean(maximum(add(state_list@theta_list.T,intercept_list),0.0),0)



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
    

    


