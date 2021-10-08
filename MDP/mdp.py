# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
class MarkovDecisionProcess:
    
    def __init__(self, mdp_setup):
        #----------------------------------------------------------------------
        # MDP setting
        #----------------------------------------------------------------------
        self.mdp_name                       = mdp_setup['mdp_conf']['mdp_name']  
        self.dim_state                      = mdp_setup['mdp_conf']['dim_state']     
        self.dim_act                        = mdp_setup['mdp_conf']['dim_act']      
        self.discount                       = mdp_setup['mdp_conf']['discount']      
        self.random_seed                    = mdp_setup['mdp_conf']['random_seed']  
        self.instance_number                = mdp_setup['mdp_conf']['instance_number']
  
    def get_batch_init_state(self,num_traj):
        #----------------------------------------------------------------------
        # Generates samples from initial state distribution
        #----------------------------------------------------------------------
        pass

    def get_feasible_actions(self, cur_state):
        #----------------------------------------------------------------------
        # Given a state, generate all feasible actions can be taken. For most
        # MDPs, this function is not useless
        #----------------------------------------------------------------------
        pass

    def get_batch_next_state(self, cur_state, cur_action):
        #----------------------------------------------------------------------
        # Given state and action, generate a batch of sampled next state.
        #----------------------------------------------------------------------
        pass
    
    def get_expected_cost(self, cur_state, cur_action, listExoInfo=None):
        #----------------------------------------------------------------------
        #  Given state and action, it generates the expected immediate cost.
        #----------------------------------------------------------------------
        pass
  
    def get_state_act_for_ALP_constr(self):
        #----------------------------------------------------------------------
        # Sample state-action pairs, used often in constraint sampling
        #----------------------------------------------------------------------
        pass
    
    def set_batch_MDP_randomness(self):
        #----------------------------------------------------------------------
        # Sample & fix a set of realized sampled from the MDP exogenous 
        # random variable.
        #----------------------------------------------------------------------
        pass
      
    def read_batch_MDP_randomness(self,file_name):
        #----------------------------------------------------------------------
        # From a file, load sampled exogenous samples, e.g., demand, price, ...
        #----------------------------------------------------------------------
        pass
     
    def get_expected_policy_cost_from_VFA(self,BF):
        #----------------------------------------------------------------------
        # Given an object from basis function class, compute the expected cost 
        # of greedy policy w.r.t. the VFA defined by the input basis 
        # function object. 
        #----------------------------------------------------------------------
        pass    
    
    def get_cost_given_noise(self, cur_state, cur_action, noise):
        #----------------------------------------------------------------------
        # Compute expected cost of being in a state-action pair
        #----------------------------------------------------------------------
        pass

    def get_next_state_given_noise(self, cur_state, cur_action, noise):
        #----------------------------------------------------------------------
        # Get a batch of next states from a state-action pair
        #----------------------------------------------------------------------
        pass

    
    
    
    
    
    
    