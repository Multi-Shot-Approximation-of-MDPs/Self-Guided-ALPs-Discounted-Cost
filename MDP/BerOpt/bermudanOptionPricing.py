# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from MDP.mdp import MarkovDecisionProcess
import numpy as np
from numpy import zeros
from numpy.linalg import cholesky
from numpy.random import standard_normal,seed
from numba import jit,prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@jit(nopython=True,nogil=True,fastmath=True)
def multi_dim_geo_brown_motion_simulator(dim:int,So:np.ndarray, mu:np.ndarray, sigma:np.ndarray, cholesky_cov:np.ndarray, delta_t:float, N:int,num_path:int,normal_dist_samples:np.ndarray):
    #--------------------------------------------------------------------------
    # This is an implementation of multidimensional geometric Brownian motion 
    # that is inspired from https://towardsdatascience.com/how-to-simulate-financial-portfolios-with-python-d0dc4b52a278
    #--------------------------------------------------------------------------
    num_samples         = np.shape(normal_dist_samples)[0]
    sample_paths        = np.zeros((num_samples,N,dim))
    for k in range(num_samples):
        S               = zeros(shape=(N, dim))
        S[0, :]         = So[k,:]
        diffusion       = cholesky_cov@normal_dist_samples[k,:,:]
        drift           = (mu - 0.5 * sigma**2)*delta_t
        path_matrix     = np.exp(np.expand_dims(drift,axis=1) + diffusion*(delta_t**.5))
    
        for i in range(1, N):
            S[i, :]     = S[i-1,:]*path_matrix[:,i-1]
           
        sample_paths[k,:,:] = S
    return sample_paths


@jit(nopython=True,nogil=True,parallel=True)
def one_step_multi_dim_geo_brown_motion_simulator(dim:int, init_price_list:np.ndarray, len_init_price_list:int, diffusion_path:np.ndarray, num_path:int):
    #--------------------------------------------------------------------------
    # ??/
    #--------------------------------------------------------------------------
    ones_step_sample    = np.empty((len_init_price_list,num_path,dim))
    for k in prange(len_init_price_list):
        ones_step_sample[k,:,:] = init_price_list[k]*diffusion_path
    
    return ones_step_sample

@jit(nopython=True,nogil=True,parallel=True)
def get_knock_out(a:np.ndarray,c:np.ndarray,b:float):
    #--------------------------------------------------------------------------
    # This is a util function to check if the price is knocked out or not
    #--------------------------------------------------------------------------
    or_ = np.empty(np.shape(a)[0])
    for i in prange(np.shape(a)[0]):
        or_[i] = np.max(a[i,:])>=b or c[i]
    return or_
    
@jit(nopython=True,nogil=True)
def pay_off(state:np.ndarray,strike_price:float):
    #--------------------------------------------------------------------------
    # Options payoff function at a given price (state)
    #--------------------------------------------------------------------------
    q = len(state)-1
    return  max(max(state[0:q]) - strike_price, 0.0)*(1.0 - state[q])

@jit(nopython=True,nogil=True)
def pay_off_list(state_list:np.ndarray,strike_price:float):
    #--------------------------------------------------------------------------
    # Options payoff function at a list of prices (states)
    #--------------------------------------------------------------------------
    (num_states,dim)    = np.shape(state_list)
    pay_off             = np.zeros(num_states)
    for i in range(num_states):
        state           = state_list[i,:]
        pay_off[i]      = max(max(state[0:dim-1]) - strike_price, 0.0)*(1.0 - state[dim-1])
    return pay_off


class BermudanOption(MarkovDecisionProcess):

    #--------------------------------------------------------------------------
    # Bermudan Options Pricing based on Section 4 of http://dx.doi.org/10.1287/mnsc.1120.1551    
    #--------------------------------------------------------------------------
   
    def __init__(self, mdp_setup):
        #----------------------------------------------------------------------
        # Constructor of the class that specifies an instance's parameters
        #----------------------------------------------------------------------
        mdp_setup['mdp_conf']['random_seed']    = None
        super().__init__(mdp_setup)
        self.num_asset:int                      = mdp_setup['mdp_conf']['num_asset']
        self.interest_rate:float                = mdp_setup['mdp_conf']['interest_rate']
        self.knock_out_price:float              = mdp_setup['mdp_conf']['knock_out_price']
        self.strike_price:float                 = mdp_setup['mdp_conf']['strike_price']
        self.discount:float                     = mdp_setup['mdp_conf']['discount']
        self.init_price                         = mdp_setup['mdp_conf']['init_price']
        self.cor_matrix                         = mdp_setup['mdp_conf']['cor_matrix']
        self.time_horizon:float                 = mdp_setup['mdp_conf']['time_horizon']
        self.volatility                         = mdp_setup['mdp_conf']['volatility']
        self.num_stages                         = mdp_setup['mdp_conf']['num_stages']
        self.inner_sample_size                  = mdp_setup['mdp_conf']['inner_sample_size']
        self.inner_sample_seed                  = mdp_setup['mdp_conf']['inner_sample_seed'] 
        self.num_cpu_core                       = mdp_setup['misc_conf']['num_cpu_core']
        self.num_VFA_sample_path:int            = mdp_setup['mdp_conf']['num_CFA_sample_path']
        self.num_pol_eval_sample_path:int       = mdp_setup['mdp_conf']['num_pol_eval_sample_path']
        self.set_brown_motion_cov_matrix()
        self.VFA_random_seed                    = mdp_setup['mdp_conf']['CFA_random_seed'] 
        self.pol_random_seed                    = mdp_setup['mdp_conf']['pol_random_seed'] 
        self.state_relevance_type               = mdp_setup['mdp_conf']['state_relevance_type']  
        
        #----------------------------------------------------------------------
        # Fix inner samples
        if not self.inner_sample_size is None:
            seed(self.inner_sample_seed)
            self.diffusion                      = standard_normal((self.inner_sample_size,self.num_asset))@self.cholesky_cov_matrix 
            delta_t                             = self.time_horizon/self.num_stages
            drift                               = np.expand_dims((self.interest_rate - 0.5 * self.volatility**2)*delta_t,axis=1).T
            self.diffusion                      = np.exp(drift  + self.diffusion*delta_t**.5)

    def set_brown_motion_cov_matrix(self):
        #----------------------------------------------------------------------
        # Set up covariance matrix of Brownian motion
        #----------------------------------------------------------------------
        self.cov_matrix          = np.tensordot(self.volatility, self.volatility, 0) * self.cor_matrix 
        self.cholesky_cov_matrix = cholesky(self.cov_matrix)


    def get_batch_samples_state_relevance(self,num_samples=None):
        #----------------------------------------------------------------------
        # Generate samples form the state relevance distribution
        #----------------------------------------------------------------------
        state_relevance_state   = [self.state_relevance[i].rvs(size=num_samples,random_state= i) for i in range(self.dim_state)] 
        samples                 = [np.array([state_relevance_state[idx][_] for idx in range(self.dim_state)]) for _ in range(num_samples)]
        return samples


    def get_sample_path(self,num_path,random_seed,method='init'):
        #----------------------------------------------------------------------
        # Generate sample paths for training and testing
        #----------------------------------------------------------------------
        delta_t                 = self.time_horizon/self.num_stages
        seed(random_seed) 
        normal_dist_samples     = standard_normal((num_path,self.num_asset,self.num_stages-1))

        #----------------------------------------------------------------------
        # Simulate paths from initial price
        if method == 'init':
            init_price_list     = np.array([self.init_price for _ in range(num_path)])
            path                = multi_dim_geo_brown_motion_simulator( self.num_asset,
                                                                        init_price_list,
                                                                        self.interest_rate,
                                                                        self.volatility,
                                                                        self.cholesky_cov_matrix,
                                                                        delta_t,
                                                                        self.num_stages,
                                                                        num_path,
                                                                        normal_dist_samples)
                    
            path                = np.reshape(path,newshape=(num_path,self.num_stages,self.num_asset))
            knocked_out         = np.zeros((num_path,self.num_stages,1),dtype=np.bool)
            
            for t in range(self.num_stages):
                if t==0:
                    knocked_out[:,t,0] = np.max(path[:,t,:],axis=1) >= self.knock_out_price
                else:
                    knocked_out[:,t,0] = np.logical_or(knocked_out[:,t-1,0], np.max(path[:,t,:],axis=1) >= self.knock_out_price)
                
            path                = np.append(path, knocked_out, axis=2)
            return np.asarray(path)
        
        #----------------------------------------------------------------------
        # Simulate paths from a price that is distributed according to the 
        # empirical distribution of prices 
        if method == 'lognormal':
            init_price_list     = np.array([self.init_price for _ in range(num_path)])
            path                = multi_dim_geo_brown_motion_simulator( self.num_asset,
                                                                        init_price_list,
                                                                        self.interest_rate,
                                                                        self.volatility,
                                                                        self.cholesky_cov_matrix,
                                                                        delta_t,
                                                                        self.num_stages,
                                                                        num_path,
                                                                        normal_dist_samples)
                    
            path                = np.reshape(path,newshape=(num_path,self.num_stages,self.num_asset))
            path                = path[:,0:int(self.num_stages/3),:]  
            init_price_list     = np.median(path,axis = 1)
            init_price_list[0,:]=self.init_price
            path                = multi_dim_geo_brown_motion_simulator( self.num_asset,
                                                                        init_price_list,
                                                                        self.interest_rate,
                                                                        self.volatility,
                                                                        self.cholesky_cov_matrix,
                                                                        delta_t,
                                                                        self.num_stages,
                                                                        num_path,
                                                                        normal_dist_samples)
                    
            path                = np.reshape(path,newshape=(num_path,self.num_stages,self.num_asset))
            knocked_out         = np.zeros((num_path,self.num_stages,1),dtype=np.bool)
            
            for t in range(self.num_stages):
                if t==0:
                    knocked_out[:,t,0] = np.max(path[:,t,:],axis=1) >= self.knock_out_price
                else:
                    knocked_out[:,t,0] = np.logical_or(knocked_out[:,t-1,0], np.max(path[:,t,:],axis=1) >= self.knock_out_price)
                
                              
            path                = np.append(path, knocked_out, axis=2)
            return np.asarray(path)

    def get_inner_samples(self,state_list,is_pol_sim=False):
        #----------------------------------------------------------------------
        # Generate inner samples given states
        #----------------------------------------------------------------------
        if not is_pol_sim:
            inner_samples   = one_step_multi_dim_geo_brown_motion_simulator(self.num_asset,
                                                                  np.array(state_list[:,0:np.shape(state_list)[1]-1]),
                                                                  np.shape(state_list)[0],
                                                                  self.diffusion ,
                                                                  self.inner_sample_size)
        else:
            inner_samples   = one_step_multi_dim_geo_brown_motion_simulator(self.num_asset,
                                                                  np.array(state_list[:,0:np.shape(state_list)[1]-1]),
                                                                  np.shape(state_list)[0],
                                                                  self.diffusion,
                                                                  self.inner_sample_size)

        inner_samples   = np.reshape(inner_samples, (len(state_list)*self.inner_sample_size,self.num_asset)) 
        is_knocked_out  = np.array([get_knock_out(inner_samples,np.repeat(state_list[:,self.dim_state-1]== 1, repeats=self.inner_sample_size),self.knock_out_price) ]).T
        inner_samples   = np.append(inner_samples, is_knocked_out,axis=1)
        inner_samples   = np.reshape(inner_samples, (len(state_list),self.inner_sample_size,self.dim_state)) 
        return inner_samples

    
    def get_reward_of_path(self, path):
        #----------------------------------------------------------------------
        # MDP reward function
        #----------------------------------------------------------------------
        num_sample_path     = np.shape(path)[0]
        path_               = np.reshape(path, newshape=(num_sample_path*self.num_stages,self.dim_state))
        rewards             =  np.reshape(pay_off_list(path_,self.strike_price),
                                          newshape=(num_sample_path,self.num_stages))

        return rewards
            
    
    def get_state(self, stage):
        #----------------------------------------------------------------------
        # Get state (prices) at a given stage
        #----------------------------------------------------------------------
        return self.path[:,stage,:]

    def get_path(self, path_index):
        #----------------------------------------------------------------------
        # Get a price path via its index
        #----------------------------------------------------------------------
        return self.path[path_index,:,:]
    

    def get_reward(self,state_list):
        #----------------------------------------------------------------------
        # Get reward of list of prices (states)
        #----------------------------------------------------------------------
        return pay_off_list(state_list,self.strike_price)
    
        
    def get_immediate_reward(self,stage):
        #----------------------------------------------------------------------
        # Get reward at a particular stage
        #----------------------------------------------------------------------
        return self.immediate_reward[:,stage]
    


            
    
  