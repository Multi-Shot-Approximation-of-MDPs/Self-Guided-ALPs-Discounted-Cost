# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from multiprocessing import Pool,Manager
from utils import mean_confidence_interval
import numpy as np
from numpy import array


class GreedyPolicy:
    
    def __init__(self,mdp,greedy_pol_conf):
        #----------------------------------------------------------------------
        # Initializing the components of the greedy policy optimization problem
        #----------------------------------------------------------------------
        self.basis_func                 = None
        self.mdp                        = mdp
        self.conf                       = greedy_pol_conf
        self.traj_len                   = self.conf['len_traj']
        self.num_traj                   = self.conf['num_traj']
        self.num_cpu_core               = self.conf['num_cpu_core']
        self.action_selector_name       = self.conf['action_selector_name']
        self.noise_list                 = self.mdp.list_demand_obs[range(1000)]
        
        #--------------------------------------------------------------------------
        # Fix rand noises
        
        if hasattr(self.mdp,'sample_fix_batch_mdp_noise'):
            self.mdp.sample_fix_batch_mdp_noise()

        if hasattr(self.mdp,'get_batch_init_state'):          
            self.init_states_list           = self.mdp.get_batch_init_state(self.num_traj)
        
        if hasattr(self.mdp,'get_batch_sample_path'):
            self.sample_path_list           = self.mdp.get_batch_sample_path(self.traj_len,self.num_traj) 
        
        
        self.observed_state_opt_action  = None      
        
        if 'state_round_decimal' in greedy_pol_conf:          
            self.state_round_decimal = greedy_pol_conf['state_round_decimal']
        else:
            self.state_round_decimal = 0
        
        
        
        if self.action_selector_name  == 'discretization':
            self.action_selector        = self.discretization
            self.all_action             = self.mdp.get_discrete_actions()
            self.state_round_decimal    = self.conf['state_round_decimal']
            
        else:
            raise ('Greedy policy method [' +self.action_selector_name+'] is not implemented.' )
    
    def set_basis_func(self,basis_func):
        #----------------------------------------------------------------------
        # Setter for specification of basis function object used in  greedy 
        # policy optimization
        #----------------------------------------------------------------------
        self.basis_func = basis_func
   
    

    def cost_on_single_trajectory(self,init_state,sample_path_noise):
        #----------------------------------------------------------------------
        # This function computes the total discounted cost of simulating a 
        # policy from an state given a trajectory of MPD noise
        #----------------------------------------------------------------------    

        #----------------------------------------------------------------------
        # Define some auxiliary variables and abbreviated functions
        tot_discounted_cost             = 0.0
        cur_state                       = init_state
        get_cost                        = self.mdp.get_cost_given_noise
        get_next_state                  = self.mdp.get_next_state_given_noise
        discount                        = self.mdp.discount
        action_selector                 = self.action_selector
        visited_states                  = [init_state]
        visited_action                  = []
        
        #----------------------------------------------------------------------
        # Iterate over stages of the given trajectory
        for time in range(self.traj_len): 
            opt_act                 = action_selector(cur_state)
            visited_action.append(opt_act)   
            tot_discounted_cost    += (discount**time)*get_cost(cur_state,opt_act,sample_path_noise[time])
            cur_state               = get_next_state(cur_state,opt_act,sample_path_noise[time]) 
            visited_states.append(cur_state)
        
        visited_action.append(action_selector(cur_state))  
        return tot_discounted_cost,visited_states,visited_action
    
    
    def expected_cost(self, get_visited_action=False):
        #----------------------------------------------------------------------
        # Estimates the expected cost of a policy defined by a VFA
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        # This dictionary stores an optimal action from a state to avoid 
        # resolving the greedy optimization problem.
        self.observed_state_opt_action = Manager().dict()
        
        #--------------------------------------------------------------------------
        # For multiple sample-path, run a pool and compute cost policy on 
        # different paths.
        inputs                          = [(self.init_states_list[_],self.sample_path_list[_]) for _ in range(self.num_traj)]
        pool                            = Pool(self.num_cpu_core)
        X                               = pool.starmap(self.cost_on_single_trajectory,inputs)
        pool.close()
        pool.join()
        cost_list        = [X[_][0] for _ in range(self.num_traj)]
        visited_states   = [np.array(X[_][1][j]) for _ in range(self.num_traj) for j in range(len(X[_][1]))]
        
        cost_mean, cost_lb,cost_ub,cost_se = mean_confidence_interval(cost_list)
        
        if get_visited_action== False:
            return cost_mean, cost_lb,cost_ub,cost_se,visited_states

        else:
            visited_action   = [np.array(X[_][2][j]) for _ in range(self.num_traj) for j in range(len(X[_][2]))]
            return cost_mean, cost_lb,cost_ub,cost_se,visited_states,visited_action
            
            
        
        #--------------------------------------------------------------------------
        # Calculate the mean and standard error of computed costs.
        
 
    
    def greedy_policy_objective(self,cur_state,cur_action):
        #----------------------------------------------------------------------
        # Objective function of the greedy optimization problem.
        #----------------------------------------------------------------------
        next_state_list           = self.mdp.get_batch_next_state(cur_state,cur_action,self.noise_list)
        immediate_cost            = self.mdp.get_expected_cost(cur_state,cur_action,self.noise_list)
        future_cost               = self.mdp.discount*self.basis_func.get_expected_VFA(next_state_list)
        return immediate_cost + future_cost
    

    def discretization(self,cur_state):
        #----------------------------------------------------------------------
        # Solves greedy optimization problems via enumeration. Only works for 
        # small-state MDPs.
        #----------------------------------------------------------------------
        cur_state =  array(cur_state).round(self.state_round_decimal)
        pickeled_state = cur_state.tobytes()
        if pickeled_state in self.observed_state_opt_action:
            opt_act = self.observed_state_opt_action[pickeled_state]
        else:
            values  = [self.greedy_policy_objective(cur_state,action) for action in self.all_action]
            opt_act = self.all_action[np.argmin(values)]
            
            self.observed_state_opt_action[pickeled_state] = opt_act
            # print(opt_act,len(np.unique(values)),end='\r')
        return opt_act
        

 