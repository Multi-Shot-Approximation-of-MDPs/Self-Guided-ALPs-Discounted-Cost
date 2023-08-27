# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""

import numpy as np
import time
from utils import mean_confidence_interval


class InformationRelaxation:

    def __init__(self,instance_conf,mdp) -> None:
        self.instance_conf                      = instance_conf
        self.mdp                                = mdp
        self.num_stages                         = instance_conf['mdp_conf']['num_stages'] 
        self.discount                           = instance_conf['mdp_conf']['discount']
        self.num_inner_sample                   = instance_conf['mdp_conf']['inner_sample_size']
        self.basis_func                         = None
        self.num_sample_path                    = None
        self.sample_path                        = None
        self.sample_path_reward                 = None
        self.num_basis_func                     = None
        self.basis_func_coef_matrix             = None
    

    def set_basis_func_coef(self,num_basis_func,basis_func_coef_matrix,basis_func,num_times_basis_added):
        self.num_basis_func         = num_basis_func
        self.basis_func_coef_matrix = basis_func_coef_matrix
        self.basis_func             = basis_func
        self.num_times_basis_added  = num_times_basis_added 


    def set_sample_path(self,num_sample_path,sample_path,sample_path_reward):
        self.num_sample_path        = num_sample_path
        self.sample_path            = sample_path
        self.sample_path_reward     = sample_path_reward

    
    def get_dual_bound_no_penalty(self):
        """ 
            Compute dual value function to get the upper bound using a deterministic DP with no dual penalty.
        """       
        dual_value_function = np.empty(shape = (self.num_sample_path,self.num_stages+2))
        for t in range(self.num_stages+1,-1,-1):
            if t == self.num_stages+1:                
                dual_value_function[:,t]    = np.zeros(self.num_sample_path)            
            else:
                dual_value_function[:,t]    = np.maximum(self.sample_path_reward[:,t], self.discount*dual_value_function[:,t+1])
          
        return mean_confidence_interval(dual_value_function[:,0])
        

    def get_dual_bound_from_CFA(self):
        """
        -----------
        Description
        -----------
        This function computes the information relaxation upper bound on the optimal reward
        using the CFA computed by the LSM algorithm. In particular, this function uses the 
        dual penalties defined in (15) of Nadarajah et al. 2017. This penalty is defined as follows:
            
            - p_{t}^i:                                  price on trajectory i at time t
            - p_{t+1}^{ji}:                             price at time t+1 of the j-th inner sample generated given time-t price p_{t}^i
            - z_t(p_{t}^i, p_{t+1}^{ji}, 'continue'):   dual penalty at time t for 'continue' action and prices p_{t}^i and p_{t+1}^{ji}
            - z_t(p_{t}^i, p_{t+1}^{ji}, 'stop'):       dual penalty at time t for 'stop' action and prices p_{t}^i and p_{t+1}^{ji}
            - C_t(p_t):                                 CFA at time t for some price p_t
            - Dual bound is defined as:
                
                - z_t(p_{t}^i, p_{t+1}^{ji}, 'stop')        = 0
                - z_t(p_{t}^i, p_{t+1}^{ji}, 'continue')    = discount*(max{r(p_{t+1}^i), C_{t+1}(p_{t+1}^i)} - [Σ_j max{r(p_{t+1}^{ji}), C_{t+1}(p_{t+1}^{ji}})]/J)          
        """
        
        dual_penalty = np.empty(shape=(self.num_sample_path,self.num_stages+1))

        for t in range(self.num_stages,-1,-1):
            
  
            if t == self.num_stages:
                # At t = T, dual_penalties = 0 based on the definition of z_t(p_{t}^i, p_{t+1}^{ji}, 'continue')
                dual_penalty[:,t]               = np.zeros(self.num_sample_path)
            
            else:
                # Computing term:    max{r(p_{t+1}^i), C_{t+1}(p_{t+1}^i)}
                next_state_list                 = self.sample_path[:,t+1,:]
                next_CFA_values                 = self.basis_func[t+1].eval_basis(next_state_list, self.num_times_basis_added)@self.basis_func_coef_matrix[:,t+1]
                next_reward_list                = self.sample_path_reward[:,t+1]
                dual_penalty_term_1             = np.maximum(next_reward_list,next_CFA_values)
                
                
                # Computing term:    [Σ_j max{r(p_{t+1}^{ji}), C_{t+1}(p_{t+1}^{ji}})]/J          
                state_list                      = self.sample_path[:,t,:]
                inner_next_state_list           = self.mdp.get_inner_samples(state_list)
                inner_next_state_reward_list    = self.mdp.get_reward_of_inner_samples(inner_next_state_list)
                

                inner_next_state_basis_eval     = self.basis_func[t+1].eval_basis_func_on_inner_samples(
                                                                                state_matrix          = inner_next_state_list,
                                                                                num_init_states       = self.num_sample_path,
                                                                                num_inner_samples     = self.num_inner_sample,
                                                                                num_times_basis_added = self.num_times_basis_added)
                
                inner_CFV_value                 = inner_next_state_basis_eval @ self.basis_func_coef_matrix[:,t+1]
                max_reward_CFA                  = np.maximum(inner_next_state_reward_list,inner_CFV_value)
                dual_penalty_term_2             = np.mean(max_reward_CFA, axis=1)
                
                # Computing term:   z_t(p_{t}^i, p_{t+1}^{ji}, 'continue')
                dual_penalty[:,t]               = self.discount*(dual_penalty_term_1 - dual_penalty_term_2)

            
 
        """ 
            Compute dual value function to get the upper bound using a deterministic DP.
            See (12) in Nadarajah et al. 2017.
        """       
        dual_value_function = np.empty(shape = (self.num_sample_path,self.num_stages+2))
        for t in range(self.num_stages+1,-1,-1):
            if t == self.num_stages+1:                
                dual_value_function[:,t]    = np.zeros(self.num_sample_path)            
            else:
                dual_value_function[:,t]    = np.maximum(self.sample_path_reward[:,t], self.discount*dual_value_function[:,t+1] - dual_penalty[:,t])
        
        
        return mean_confidence_interval(dual_value_function[:,0])


        

    def get_dual_bound_from_VFA(self,continue_function):
        """
            This function computes the information relaxation upper bound on the optimal reward
            using the CFA from LSM algorithm.
            
            Description:
            
                - z_t( ):           time-t penalty
                - p_{t+1}^i:        price on trajectory i at time i+1
                - p_{t+1}^{j|i}:    price at time t+1 on the inner sample j given price p_{t}^i
                - We use the following formula to compute dual penalties:
                
                    z_t(p_t^i) = V_{t+1}(p_{t+1}^i)  - (1/J)* Σ_j  V_{t+1}(p_{t+1}^{j|i}})

        """

        dual_penalties  = np.empty(shape=(self.num_sample_path,self.num_stages+1))

        
        for t in range(self.num_stages,-1,-1):

            if t == self.num_stages:
                dual_penalties[:,t]             = np.zeros(self.num_sample_path)
            
            elif t == self.num_stages-1:
                next_state_basis_eval           = self.sample_path_reward[:,t+1]
                
                
                state_list                      = self.sample_path[:,t,:]
                inner_next_state_list           = self.mdp.get_inner_samples(state_list)
                inner_next_state_list_reward    = self.mdp.get_reward_of_inner_samples(inner_next_state_list) 
                expected_next_state_basis_eval  = np.mean(inner_next_state_list_reward,axis=1)
                
                dual_penalties[:,t]             = self.discount*(next_state_basis_eval - expected_next_state_basis_eval) 
            
            
            else:
                
                next_state_list                 = self.sample_path[:,t+1,:]
                next_state_basis_eval           = self.basis_func[t+1].eval_basis(next_state_list, self.num_times_basis_added,all_bases=True)
                
                # state_list                      = self.sample_path[:,t,:]
                # inner_next_state_list           = self.mdp.get_inner_samples(state_list)
                # expected_next_state_basis_eval  = self.basis_func[t+1].compute_expected_basis_func(
                #                                                             state_matrix          = inner_next_state_list,
                #                                                             num_init_states       = self.num_sample_path,
                #                                                             num_inner_samples     = self.num_inner_sample,
                #                                                             num_times_basis_added = self.num_times_basis_added)
                        
                # expected_next_state_basis_eval  = self.discount*(expected_next_state_basis_eval @ self.basis_func_coef_matrix[:,t+1] )

                dual_penalties[:,t]             = self.discount*(next_state_basis_eval) @ self.basis_func_coef_matrix[:,t+1] - continue_function[:,t]

 



                
        """ Compute dual value function """       
        dual_value_function = np.empty(shape = (self.num_sample_path,self.num_stages+2))
        for t in range(self.num_stages+1,-1,-1):
            if t == self.num_stages+1:                
                dual_value_function[:,t]    = np.zeros(self.num_sample_path)            
            else:
                dual_value_function[:,t]    = np.maximum(self.sample_path_reward[:,t], self.discount*dual_value_function[:,t+1] - dual_penalties[:,t])
        
            
        
        return mean_confidence_interval(dual_value_function[:,0])
