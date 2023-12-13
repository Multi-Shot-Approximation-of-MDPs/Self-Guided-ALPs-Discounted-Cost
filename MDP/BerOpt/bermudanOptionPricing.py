
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
from numpy import linspace,zeros
from numpy.linalg import cholesky
from numpy.random import standard_normal,seed,uniform,lognormal
from numba import jit,prange

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
from utils import make_text_bold

from multiprocessing import Pool
from functools import partial
from scipy.stats import norm,multivariate_normal


import time

"""
    Based on:       https://towardsdatascience.com/how-to-simulate-financial-portfolios-with-python-d0dc4b52a278
"""

@jit(nopython=True,nogil=True,fastmath=True)
def multi_dim_geo_brown_motion_simulator(dim:int,So:np.ndarray, mu:np.ndarray, sigma:np.ndarray, cholesky_cov:np.ndarray, delta_t:float, N:int,num_path:int,normal_dist_samples:np.ndarray):
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
    # ??
    #--------------------------------------------------------------------------
    ones_step_sample    = np.empty((len_init_price_list,num_path,dim))
    for k in prange(len_init_price_list):
        ones_step_sample[k,:,:] = init_price_list[k]*diffusion_path[k]
    
    return ones_step_sample

 

@jit(nopython=True,nogil=True,parallel=True)
def get_knock_out(a:np.ndarray,c:np.ndarray,b:float):
    or_ = np.empty(np.shape(a)[0])
    for i in prange(np.shape(a)[0]):
        or_[i] = np.max(a[i,:])>=b or c[i]
    return or_
    

# # @jit(nopython=True,nogil=True,parallel=False,fastmath=True)
# def one_step_multi_dim_geo_brown_motion_simulator(dim:int, init_price_list:np.ndarray,knock_out:np.ndarray, knock_out_price:float, len_init_price_list:int, diffusion_path:np.ndarray, num_path:int):
#     ones_step_sample    = np.empty((num_path, len_init_price_list,dim))
#     for k in prange(len_init_price_list):
#         ones_step_sample[:,k,0:dim-1]   = init_price_list[k]*diffusion_path
        
#         print('\n',np.shape(init_price_list[k]*diffusion_path), knock_out[k] or np.max(init_price_list[k]*diffusion_path) >= knock_out_price)
#         exit()
        
#         # ones_step_sample[:,k,  dim-1]   = 
        
#     return ones_step_sample


@jit(nopython=True,nogil=True)
def pay_off_inner_samples(state_matrix:np.ndarray,strike_price:float):
    #--------------------------------------------------------------------------
    # Options payoff function at a list of prices (states)
    #--------------------------------------------------------------------------
    
    (num_outer_sample,num_inner_sample,dim_state) = np.shape(state_matrix)
    pay_off = np.zeros((num_outer_sample,num_inner_sample))
    
    for i in range(num_outer_sample):
        pay_off[i,:] = pay_off_list(state_list=state_matrix[i,:,:],strike_price=strike_price)

    return pay_off


@jit(nopython=True,nogil=True)
def pay_off(state:np.ndarray,strike_price:float):
    q = len(state)-1
    return  max(max(state[0:q]) - strike_price, 0.0)*(1.0 - state[q])


@jit(nopython=True,nogil=True)
def pay_off_list(state_list:np.ndarray,strike_price:float):
    
    (num_states,dim)    = np.shape(state_list)
    pay_off             = np.zeros(num_states)
    
    for i in range(num_states):
        state           = state_list[i,:]
        pay_off[i]      = max(max(state[0:dim-1]) - strike_price, 0.0)*(1.0 - state[dim-1])
        
    return pay_off






class BermudanOption(MarkovDecisionProcess):

    def __init__(self, mdp_setup):
        # print(mdp_setup)
        
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
        
        
    def fix_inner_samples(self,inner_sample_size,outer_sample_size,inner_sample_seed):  
        
        self.diffusion = []
        for j in range(outer_sample_size):
            seed(inner_sample_seed+j)
            diffusion       = standard_normal((inner_sample_size, self.num_asset))@self.cholesky_cov_matrix 
            delta_t         = self.time_horizon/self.num_stages
            drift           = np.expand_dims((self.interest_rate - 0.5 * self.volatility**2)*delta_t,axis=1).T
            diffusion       = np.exp(drift + diffusion*delta_t**.5)
            self.diffusion.append(diffusion)
        
    
    def set_brown_motion_cov_matrix(self):
         self.cov_matrix                         = np.tensordot(self.volatility, self.volatility, 0) * self.cor_matrix 
         self.cholesky_cov_matrix                = cholesky(self.cov_matrix)


    def get_batch_samples_state_relevance(self,num_samples=None):
        state_relevance_state   = [self.state_relevance[i].rvs(size=num_samples,random_state= i) for i in range(self.dim_state)] 
        samples                 = [np.array([state_relevance_state[idx][_] for idx in range(self.dim_state)]) for _ in range(num_samples)]
        return samples


    def get_sample_path(self,num_path,random_seed,method='init'):
        
        delta_t             = self.time_horizon/self.num_stages
        seed(random_seed) 
        normal_dist_samples = standard_normal((num_path,self.num_asset,self.num_stages))


        if method == 'init':
    
            init_price_list     = np.array([self.init_price for _ in range(num_path)])
            
            path                = multi_dim_geo_brown_motion_simulator( self.num_asset,
                                                                        init_price_list,
                                                                        self.interest_rate,
                                                                        self.volatility,
                                                                        self.cholesky_cov_matrix,
                                                                        delta_t,
                                                                        self.num_stages+1,
                                                                        num_path,
                                                                        normal_dist_samples)
                    
            path                = np.reshape(path,newshape=(num_path,self.num_stages+1,self.num_asset))
            knocked_out         = np.zeros((num_path,self.num_stages+1,1),dtype=np.bool_)
            
            
            for t in range(self.num_stages+1):
                if t==0:
                    knocked_out[:,t,0] = np.max(path[:,t,:],axis=1) >= self.knock_out_price
                else:
                    knocked_out[:,t,0] = np.logical_or(knocked_out[:,t-1,0], np.max(path[:,t,:],axis=1) >= self.knock_out_price)
                
                              
            path                = np.append(path, knocked_out, axis=2)
            return np.asarray(path)
        
        
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
                                                                        self.num_stages+1,
                                                                        num_path,
                                                                        normal_dist_samples)
                    
            path                = np.reshape(path,newshape=(num_path,self.num_stages+1,self.num_asset))
            knocked_out         = np.zeros((num_path,self.num_stages+1,1),dtype=np.bool_)
            
            
            for t in range(self.num_stages+1):
                if t==0:
                    knocked_out[:,t,0] = np.max(path[:,t,:],axis=1) >= self.knock_out_price
                else:
                    knocked_out[:,t,0] = np.logical_or(knocked_out[:,t-1,0], np.max(path[:,t,:],axis=1) >= self.knock_out_price)
                
                              
            path                = np.append(path, knocked_out, axis=2)
            return np.asarray(path)


    def get_state_act_for_upper_bound(self,num_samples):
        
        seed(321) 
        noise               = standard_normal( (num_samples,self.num_asset))#*10
        noise[0]            = 0.0

        init_price_list     = np.array([self.init_price + noise[_] for _ in range(num_samples)])

        
        # knocked_out         = np.zeros((num_samples,1),dtype=np.bool_)
        # knocked_out[:,0]    = np.max(init_price_list,axis=1) >= self.knock_out_price
        
        # init_state          = np.append(init_price_list, knocked_out, axis=1)
        

        return init_price_list
    
    
    def get_reward_of_path(self, path):
        num_sample_path     = np.shape(path)[0]
        path_               = np.reshape(path, newshape=(num_sample_path*(self.num_stages+1),self.dim_state))
        rewards             =  np.reshape(pay_off_list(path_,self.strike_price),
                                          newshape=(num_sample_path,self.num_stages+1))

        return rewards
            
            
    def get_state(self, stage):
        return self.path[:,stage,:]

    def get_path(self, path_index):
        return self.path[path_index,:,:]
    

    def get_reward(self,state_list):
        return pay_off_list(state_list,self.strike_price)
    
        
    def get_immediate_reward(self,stage):
        return self.immediate_reward[:,stage]
    
    
    def is_state_feasible(self,prices,stage,init_MCMC_state):
        assert len(prices) == self.num_asset
        
        if max(prices) >=self.knock_out_price: 
            return False
        
        if min(prices) <=0.0: 
            return False  
        
        # if max(prices) < self.strike_price: 
        #     return False
        
        
        
        # mins = np.min(init_MCMC_state[:,stage,0:self.num_asset],axis=0)#*.8
        # maxs = np.max(init_MCMC_state[:,stage,0:self.num_asset],axis=0)#*1.2

        # print(stage,mins)

        # if not all(prices<=maxs): 
        #     return False
        
        # if not all(prices>=mins): 
        #     return False
        
        
        return True
        
    

    def get_inner_samples(self,state_list):
        
        inner_samples   = one_step_multi_dim_geo_brown_motion_simulator(
                                                              dim = self.num_asset,
                                                              init_price_list = np.array(state_list[:,0:np.shape(state_list)[1]-1]),
                                                              len_init_price_list = np.shape(state_list)[0],
                                                              diffusion_path = self.diffusion ,
                                                              num_path = self.inner_sample_size)
        
        

        
        # else:
        #     inner_samples   = one_step_multi_dim_geo_brown_motion_simulator(self.num_asset,
        #                                                           np.array(state_list[:,0:np.shape(state_list)[1]-1]),
        #                                                           np.shape(state_list)[0],
        #                                                           self.diffusion,
        #                                                           self.inner_sample_size)

        inner_samples   = np.reshape(inner_samples, (len(state_list)*self.inner_sample_size,self.num_asset)) 
        is_knocked_out  = np.array([get_knock_out(inner_samples,np.repeat(state_list[:,self.dim_state-1]== 1, repeats=self.inner_sample_size),self.knock_out_price) ]).T
        inner_samples   = np.append(inner_samples, is_knocked_out,axis=1)
        inner_samples   = np.reshape(inner_samples, (len(state_list),self.inner_sample_size,self.dim_state)) 
        return inner_samples

            
    
    def get_reward_of_inner_samples(self, inner_samples):
        #----------------------------------------------------------------------
        # MDP reward function
        #----------------------------------------------------------------------
                                

        return pay_off_inner_samples(inner_samples,self.strike_price)