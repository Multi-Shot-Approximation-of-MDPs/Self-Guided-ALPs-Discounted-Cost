# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from MDP.mdp import MarkovDecisionProcess
import numpy as np
from scipy.stats import truncnorm
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@jit(nopython=True,nogil=True,fastmath=True)
def get_cost_given_noise(c_p,c_h,c_b,c_d,c_l,max_backlog,life_time,cur_state, cur_action, noise):  
    #----------------------------------------------------------------------
    # Compute expected cost of being in a state-action pair
    #----------------------------------------------------------------------
    sum_inventories = 0.0
    tot_cost        = 0.0
    for _ in range(1,life_time):        
        sum_inventories +=cur_state[_]

    not_met_demand = noise - sum_inventories -cur_state[0]
    tot_cost += c_h*max(0.0, sum_inventories - max(0.0,noise-cur_state[0]) ) +\
                c_b*max(0.0, not_met_demand) +\
                c_d*max(0.0, cur_state[0]-noise) +\
                c_l*max(0.0, max_backlog +not_met_demand)
    
    return c_p*cur_action + tot_cost


@jit(nopython=True,nogil=True,fastmath=True)
def get_expected_cost(c_p,c_h,c_b,c_d,c_l,max_backlog,life_time,cur_state, cur_action, noise_list):  
    #----------------------------------------------------------------------
    # Given state and action, it returns the expected immediate cost
    #----------------------------------------------------------------------
    sum_inventories = 0.0
    tot_cost        = 0.0
    for _ in range(1,life_time):        
        sum_inventories +=cur_state[_]

    for _ in  range(len(noise_list)):
        noise = noise_list[_]
        not_met_demand = noise - sum_inventories -cur_state[0]
        tot_cost += c_h*max(0.0, sum_inventories - max(0.0,noise-cur_state[0]) ) +\
                    c_b*max(0.0, not_met_demand) +\
                    c_d*max(0.0, cur_state[0]-noise) +\
                    c_l*max(0.0, max_backlog +not_met_demand)
    
    return c_p*cur_action + (tot_cost/len(noise_list))


@jit(nopython=True,nogil=True)
def get_batch_next_state(dim_state:int,max_backlog:float,life_time:int,lead_time:int,cur_state,cur_action,noise_list):
    #----------------------------------------------------------------------
    # Given state and action, generate a batch of sampled next state.
    #----------------------------------------------------------------------
    next_state_list = np.zeros((len(noise_list),dim_state))
    total_on_hand   = 0
    for _ in range(2,life_time):
        total_on_hand+=cur_state[_]
    
    for _ in range(len(noise_list)):
        next_state_list[_,0] = max((cur_state[1] - max(0,noise_list[_] - cur_state[0])), max_backlog -total_on_hand) 
        
        for shift_left in range(1,lead_time+life_time-2):
            next_state_list[_,shift_left] = cur_state[shift_left+1]
        
        next_state_list[_,dim_state-1] = cur_action

    return next_state_list


@jit(nopython=True,nogil=True)
def get_next_state_given_noise(dim_state:int,max_backlog:float,life_time:int,lead_time:int,cur_state,cur_action,noise):
    #----------------------------------------------------------------------
    # Get a batch of next states from a state-action pair
    #----------------------------------------------------------------------
    next_state_list = np.zeros(dim_state)
    total_on_hand   = 0
    for _ in range(2,life_time):
        total_on_hand+=cur_state[_]
    
    next_state_list[0] = max((cur_state[1] - max(0,noise - cur_state[0])), max_backlog -total_on_hand) 
    for shift_left in range(1,lead_time+life_time-2):
        next_state_list[shift_left] = cur_state[shift_left+1]
    
    next_state_list[dim_state-1] = cur_action
    return next_state_list


class PerishableInventoryPartialBacklogLeadTime(MarkovDecisionProcess):
    
    def __init__(self,mdp_setup):
        #----------------------------------------------------------------------
        # Constructor of class
        #----------------------------------------------------------------------
        super().__init__(mdp_setup)
        
        self.c_p: float                     = mdp_setup['mdp_conf']['purchase_cost']                    # Purchasing cost
        self.c_b: float                     = mdp_setup['mdp_conf']['backlogg_cost']                    # Backlogging cost     
        self.c_d: float                     = mdp_setup['mdp_conf']['disposal_cost']                    # Disposal cost
        self.c_l: float                     = mdp_setup['mdp_conf']['lostsale_cost']                    # Lost sales cost
        self.c_h: float                     = mdp_setup['mdp_conf']['holding_cost']                     # Holding cost
        self.max_order: float               = mdp_setup['mdp_conf']['max_order']                        # Max order level
        self.max_backlog: float             = mdp_setup['mdp_conf']['max_backlog']                      # Max backlogging level
        self.lead_time: int                 = mdp_setup['mdp_conf']['lead_time']                        # Lead time
        self.life_time:int                  = mdp_setup['mdp_conf']['life_time']                        # Merchandise lifetime
        self.dist_mean: float               = mdp_setup['mdp_noise_conf']['dist_mean']                  # Mean of sampling distribution
        self.dist_std: float                = mdp_setup['mdp_noise_conf']['dist_std']                   # STD of sampling distribution
        self.dist_min: float                = mdp_setup['mdp_noise_conf']['dist_min']                   # Min of sampling distribution
        self.dist_max: float                = mdp_setup['mdp_noise_conf']['dist_max']                   # Max of sampling distribution
        self.num_sample_noise:int           = mdp_setup['mdp_noise_conf']['num_sample_noise']
        self.action_discrete_param:float    = mdp_setup['greedy_pol_conf']['action_discrete_param']
        self.init_state_sampler             = mdp_setup['greedy_pol_conf']['init_state_sampler']        # Initial distribution used for policy simulation
        self.state_sampler                  = mdp_setup['mdp_sampler_conf']['state_sampler']            # Random variable for sampling states
        self.act_sampler                    = mdp_setup['mdp_sampler_conf']['act_sampler']              # Random variable for sampling actions
        self.state_relevance                = mdp_setup['mdp_sampler_conf']['state_relevance']          # State relevance distribution
        self.constr_batch_size: int         = mdp_setup['constr_conf']['constr_gen_batch_size']         # State relevance distribution
        self.num_cpu_core: int              = mdp_setup['misc_conf']['num_cpu_core']                    # Number of CPU cores
        self.list_demand_obs                = None          
        self.demand_sampler                 = truncnorm(a  =(self.dist_min - self.dist_mean)/self.dist_std,
                                                        b  =(self.dist_max - self.dist_mean)/self.dist_std,
                                                        loc=self.dist_mean,  scale=self.dist_std)
    
    
    def get_cost_given_noise(self,cur_state, cur_action, noise):   
        return get_cost_given_noise(self.c_p,self.c_h,self.c_b,self.c_d,self.c_l,self.max_backlog, self.life_time,np.asarray(cur_state,float), cur_action, noise)
    
    def get_next_state_given_noise(self, cur_state, cur_action, noise):
        return  get_next_state_given_noise(self.dim_state,self.max_backlog,self.life_time,self.lead_time,np.asarray(cur_state,float),cur_action,noise)
    
    def get_expected_cost(self,cur_state, cur_action, noise_list=None):
        if noise_list is None:
            return get_expected_cost(self.c_p,self.c_h,self.c_b,self.c_d,self.c_l,self.max_backlog, self.life_time,np.asarray(cur_state,float), cur_action, self.list_demand_obs)
        else:
            return get_expected_cost(self.c_p,self.c_h,self.c_b,self.c_d,self.c_l,self.max_backlog, self.life_time,np.asarray(cur_state,float), cur_action, noise_list)
            
    def get_batch_next_state(self, cur_state, cur_action, noise_list = None):
        if noise_list is not None:
             return get_batch_next_state(self.dim_state,self.max_backlog,self.life_time,self.lead_time,np.asarray(cur_state,float),cur_action,noise_list)
        return get_batch_next_state(self.dim_state,self.max_backlog,self.life_time,self.lead_time,np.asarray(cur_state,float),cur_action,self.list_demand_obs)

    def get_batch_mdp_noise(self,num_samples=None):
        return self.demand_sampler.rvs(size = self.num_sample_noise,random_state = self.random_seed) if num_samples is None else  self.demand_sampler.rvs(size = num_samples)  

    def sample_fix_batch_mdp_noise(self):
        self.list_demand_obs = self.get_batch_mdp_noise()
        return self.list_demand_obs       

    def get_batch_next_state_given_next_states_from_diff_action(self,other_next_state,new_action):
        #----------------------------------------------------------------------
        # Knowing the next states from a state and an action, returns the next
        # states from the same states from a different action. This function
        # highly depends on the structure of this problem.
        #----------------------------------------------------------------------
        other_next_state[:,self.dim_state-1] = new_action
        return other_next_state

    def get_state_act_for_ALP_constr(self, num_samples = None,random_seed=1):
        #----------------------------------------------------------------------
        # Sample state-action pairs, used often in constraint sampling
        #----------------------------------------------------------------------
        coordinate_state_list           = [self.state_sampler[i].rvs(size=self.constr_batch_size,random_state=self.random_seed + (i+1)*random_seed) for i in range(self.dim_state)]
        if num_samples is None:
            state_list                  = [np.array([coordinate_state_list[idx][_] for idx in range(self.dim_state)]) for _ in range(self.constr_batch_size)]
            act_List                    = self.act_sampler.rvs(size=self.constr_batch_size,random_state=self.random_seed + (self.dim_state+1)*random_seed)
            
        else:
            state_list                  = [np.array([coordinate_state_list[idx][_] for idx in range(self.dim_state)]) for _ in range(num_samples)]
            act_List                    = self.act_sampler.rvs(size=num_samples,random_state=self.random_seed + (self.dim_state+1)*random_seed)
        
        return state_list,act_List

    def get_batch_samples_state_relevance(self,num_samples=None):
        #----------------------------------------------------------------------
        # Generate samples from state relevance distribution
        #----------------------------------------------------------------------
        if num_samples == None:
            num_samples = self.num_sample_noise
        state_relevance_state = [self.state_relevance[i].rvs(size=num_samples,random_state=self.random_seed + i) for i in range(self.dim_state)] +[np.array([-10,0]),np.array([-10,10]),np.array([10,10]),np.array([10,0])]
        return [np.array([state_relevance_state[idx][_] for idx in range(self.dim_state)]) for _ in range(num_samples)]

    def get_batch_sample_path(self,path_len,num_traj):
        #----------------------------------------------------------------------
        # Generate samples paths of demand realizations
        #----------------------------------------------------------------------
        return [np.asarray(self.demand_sampler.rvs( size= path_len,random_state=self.random_seed + _),dtype=float)     for _ in range(num_traj)]
    
    def get_discrete_actions(self):
        #----------------------------------------------------------------------
        # Discritize action space
        #----------------------------------------------------------------------
        return np.arange(start=0.0, stop =self.max_order+self.action_discrete_param, step =self.action_discrete_param,dtype=float)
    
    def get_batch_init_state(self,num_traj):
        #----------------------------------------------------------------------
        # Sample states from the initial distribution
        #----------------------------------------------------------------------
        coordinate_init_state_list = [self.init_state_sampler[i].rvs(size=num_traj,random_state=self.random_seed + i) for i in range(self.dim_state)]
        init_state_list = [np.array([coordinate_init_state_list[idx][_] for idx in range(self.dim_state)]) for _ in range(num_traj)]
        return init_state_list
    
    def is_state_action_feasible(self,state,action):
        #----------------------------------------------------------------------
        # Returens true only if the input state-action pair is feasbile
        #----------------------------------------------------------------------
        if action < 0 or action > self.max_order:
            return False
        if any(state > self.max_order) or state[0]<self.max_backlog:
            return False
        for i in range(1,self.dim_state):
            if state[i]<0:
                return False
        
        return True
 
    
 
    