"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""

"""
    Class framework for general MDPs
"""
class MarkovDecisionProcess:
    
    """
        Constructing an MDP object
    """
    def __init__(self, mdp_setup):
    #--------------------------------------------------------------------------
      # MDP setting
      self.mdp_name                       = mdp_setup['mdp_conf']['mdp_name']  
      self.dim_state                      = mdp_setup['mdp_conf']['dim_state']    # Dimension of state
      self.dim_act                        = mdp_setup['mdp_conf']['dim_act']      # Dimension of action
      self.discount                       = mdp_setup['mdp_conf']['discount']     # Discount  factor
      self.random_seed                    = mdp_setup['mdp_conf']['random_seed']  
      self.instance_number                = mdp_setup['mdp_conf']['instance_number']
  
    """ 
        Generates a batch of samples from randomness in the MDP
    """
    def get_batch_init_state(self,num_traj):
        pass

    """ 
        Given a state, generate all feasible actions can be taken.
    """
    def get_feasible_actions(self, cur_state):
        pass

    """ 
        Given state and action, generate a batch of sampled next state.
    """
    def get_batch_next_state(self, cur_state, cur_action):
        pass
    
    """ 
        Given state and action, it generates the expected immediate cost.
    """
    def get_expected_cost(self, cur_state, cur_action, listExoInfo=None):
        pass

    """ 
        Generate state-action pairs.
    """    
    def get_state_act_for_ALP_constr(self):
        pass
        
    """ 
        Sample & fix a set of realized sampled from the MDP exogenous random variable.
    """        
    def set_batch_MDP_randomness(self):
        pass

    """ 
        From a file, load sampled exogenous samples, e.g., demand, price, ...
    """       
    def read_batch_MDP_randomness(self,file_name):
        pass

    """ 
        Given an object from basis function class, compute the expected cost of
        greedy policy w.r.t. the VFA defined by the input basis function object.        
    """     
    def get_expected_policy_cost_from_VFA(self,BF):
        pass    
    
    def get_cost_given_noise(self, cur_state, cur_action, noise):
        pass

    def get_next_state_given_noise(self, cur_state, cur_action, noise):
        pass

    
    
    
    
    
    
    