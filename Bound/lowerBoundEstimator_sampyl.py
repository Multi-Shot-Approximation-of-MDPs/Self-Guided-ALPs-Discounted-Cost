# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from multiprocessing import Pool
import numpy as np
from math import log,pi,gamma
from sampyl.samplers.metropolis import Metropolis
from utils import mean_confidence_interval
LB_RANDOM_STATE=333


class LowerBound:
    
    #--------------------------------------------------------------------------
    # Please see the Online Supplement of Pakiman t al. 2021. This function 
    # computes a lower bound on the optimal cost ba sampling approach.
    # Lambda is a fixed parameter defined in Lemma EC.3 of Lin, Qihang, 
    # Selvaprabu Nadarajah, and Negar Soheili (LNS). "Revisiting approximate 
    # linear programming:  Constraint-violation learning with applications to 
    # inventory control and energy storage." Management Science (2019).
    #--------------------------------------------------------------------------
    def __init__(self,mdp,lower_bound_conf):
        #----------------------------------------------------------------------
        # Constructor for specification of VFA, parameters, etc.
        #---------------------------------------------------------------------- 
        
        self.basis_func                 = None
        self.mdp                        = mdp
        self.conf                       = lower_bound_conf
        self.lower_bound_algo_name:str  = lower_bound_conf['lower_bound_algo_name']
        self.init_states_list           = self.mdp.get_batch_init_state(self.mdp.num_sample_noise)
        self.num_cpu_core               = self.conf['num_cpu_core']
        
        if self.lower_bound_algo_name == 'LNS':
            self.get_lower_bound                = self.LNS_lower_bound
            self.num_MC_init_states             = lower_bound_conf['num_MC_init_states']
            self.expected_VFA_on_initial_state  = None
            self.lambda_                        = None
        else:
            raise('Lower bounding method [' + self.lower_bound_algo_name+ '] is not implemented.')


    def set_basis_func(self,basis_func):
        #----------------------------------------------------------------------
        # Fix VFA used in lower bound estimation based on the constraint 
        # violation learning approach.
        #----------------------------------------------------------------------
        self.basis_func = basis_func
        self.expected_VFA_on_initial_state = self.basis_func.get_expected_VFA(self.init_states_list)
        self.lambda_ = self.get_lambda()


    def get_LNS_constant(self):
        #----------------------------------------------------------------------
        # Lipschitz constant of function Y; Please see **nOnline Supplement of 
        # Pakiman et al. 2021.
        # ** Given weights of a VFA, we compute 1-norm of its weights.
        #----------------------------------------------------------------------
        """ KEY ASSUMPTION: First Basis is Constant """
        norm_VFA_coefs          = np.linalg.norm(self.basis_func.opt_coef,1) 
        Lipschitz_cost_func     = ((2*self.mdp.discount**2)*self.mdp.c_p + self.mdp.c_h + self.mdp.c_b + self.mdp.c_d + self.mdp.c_l)*self.mdp.max_order
        Lipschitz_dual_func     = (4*norm_VFA_coefs + Lipschitz_cost_func)/(1-self.mdp.discount)
    
        #----------------------------------------------------------------------
        # Page 20 of Revisiting approximate linear programming: 
        # Constraint-violation learning with applications to 
        # inventory control and energy storage
        LNS_constant =  log(1/self.conf['volume_state_action'])  -\
                        Lipschitz_dual_func*(self.conf['radius_ball_in_state_action'] + self.conf['diameter_state_action']) +\
                        self.conf['dim_state_act']*log(self.conf['radius_ball_in_state_action']) -\
                        -log(gamma((self.conf['dim_state_act']/2) + 1) / (pi**(self.conf['dim_state_act']/2)))

        return LNS_constant        

    def get_lambda(self):
        #----------------------------------------------------------------------
        # A heuristic way of to choose lambda in constraint violation learning 
        # approach
        #----------------------------------------------------------------------
        return abs(1/(self.get_LNS_constant() + self.mdp.dim_state + self.mdp.dim_act)) 


    def saddle_func(self,state,action):
        #----------------------------------------------------------------------
        # If a state-action pair is infeasible to a PIC instance, then assign 
        # infinity value, e.g., no violation in the ALP constraint! This ensures,
        # with small probability, we may sample this infeasible pair.
        # Check if a state-action pair is feasible to a PIC instance.
        #----------------------------------------------------------------------
        expected_cost   = self.mdp.get_expected_cost(state,action)
        expected_VFA    = self.basis_func.get_expected_VFA(self.mdp.get_batch_next_state(state,action))
        val             = (expected_cost + self.mdp.discount*expected_VFA - self.basis_func.get_VFA(state))/(1-self.mdp.discount)
        return val 
       
    def MC_sampler_given_init_state(self,MC_init_state):
        #----------------------------------------------------------------------
        # Run Metropolis-Hastings from an initial state
        #----------------------------------------------------------------------
        def log_prob(state,action):
            if self.mdp.is_state_action_feasible (state,action):
                return -(self.saddle_func(state,action)/self.lambda_)  
            else:
                return -np.infty  
        
        return Metropolis(logp = log_prob, start = MC_init_state,random_seed=LB_RANDOM_STATE).sample(num=650,burn=500,progress_bar=False)
    
    
    def LNS_lower_bound(self,state_high_val):
        #----------------------------------------------------------------------
        # Lower estimator
        #----------------------------------------------------------------------
        rand_state_list,rand_action_list     = self.mdp.get_state_act_for_ALP_constr(self.num_MC_init_states)
        initial_state_action_pairs           = [{'state':state_high_val[_], 'action':rand_action_list[_]} for _ in range(self.num_MC_init_states)]
        
        pool    = Pool(self.num_cpu_core)
        chain   = pool.map(self.MC_sampler_given_init_state,initial_state_action_pairs)       
        pool.close()
        pool.join()
        
        pool    = Pool(self.num_cpu_core)
        lower_bound_evals = pool.starmap(self.saddle_func,[(chain[i].state[j],chain[i].action[j]) for i in range(self.num_MC_init_states) for j in range(len(chain[i].state))])
        pool.close()
        pool.join()
        
        #----------------------------------------------------------------------
        # The average value of saddleFunction plus some constants that are provided.
        # below gives a lower bound estimate. Also, compute the standard error.
        lower_bound_mean, lower_bound_lb,lower_bound_ub,lower_bound_se = mean_confidence_interval(lower_bound_evals)
        
        
        lower_bound_mean    += self.conf['dim_state_act']*self.lambda_*log(self.lambda_)  + self.lambda_*self.get_LNS_constant() + self.expected_VFA_on_initial_state
        lower_bound_lb      += self.conf['dim_state_act']*self.lambda_*log(self.lambda_)  + self.lambda_*self.get_LNS_constant() + self.expected_VFA_on_initial_state
        lower_bound_ub      += self.conf['dim_state_act']*self.lambda_*log(self.lambda_)  + self.lambda_*self.get_LNS_constant() + self.expected_VFA_on_initial_state
    
        return lower_bound_mean, lower_bound_lb,lower_bound_ub,lower_bound_se





