# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
import numpy as np
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from numba import jit
from utils import mean_confidence_interval,make_text_bold
import time
from utils import output_handler_option_pricing
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
     
                      
@jit 
def lstsq_jit(X,y):
    #--------------------------------------------------------------------------
    # Jited least seqaures
    #--------------------------------------------------------------------------
    return np.linalg.lstsq(a=X,b=y)[0]

class LeastSquaresMonteCarlo():
    #--------------------------------------------------------------------------
    # Least-Squares Monte Carlo: Longstaff & Schwartz
    #--------------------------------------------------------------------------
    def __init__(self,instance_conf):
        #----------------------------------------------------------------------
        # Initialization
        #----------------------------------------------------------------------
        self.instance_conf                      = instance_conf
        self.mdp                                = instance_conf['mdp_conf']['mdp'](instance_conf)
        self.basis_func                         = instance_conf['basis_func_conf']['basis_func'](instance_conf)
        self.num_CFA_sample_path:int            = instance_conf['mdp_conf']['num_CFA_sample_path']
        self.num_pol_eval_sample_path:int       = instance_conf['mdp_conf']['num_pol_eval_sample_path']
        self.num_stages                         = self.mdp.num_stages
        self.num_basis_func                     = self.basis_func.num_basis_func
        self.discount                           = instance_conf['mdp_conf']['discount']
        self.basis_func_coef_matrix             = np.empty(shape=(self.num_basis_func,self.num_stages))
        self.CFA_random_seed                    = instance_conf['mdp_conf']['CFA_random_seed'] 
        self.pol_random_seed                    = instance_conf['mdp_conf']['pol_random_seed'] 
        self.output_handler                     = output_handler_option_pricing(instance_conf)


    def print_algorithm_instance_info(self):
        #----------------------------------------------------------------------
        # Print to users some info
        #----------------------------------------------------------------------
        print('\n')
        print('='*99)
        print('Instance number:             \t'     + make_text_bold(self.mdp.instance_number))
        print('Algorithm name:              \t'     + make_text_bold('LSM'))
        print('Basis function type:         \t'     + make_text_bold(self.basis_func.basis_func_type))
        print('State relevance:             \t'     + make_text_bold(self.mdp.state_relevance_type))
        print('Random seed of CFA paths:    \t'     + make_text_bold(str(self.CFA_random_seed)))
        print('Random seed of pol sim:      \t'     + make_text_bold(str(self.pol_random_seed)))
        print('='*99)
        print('| {:>9s} | {:>8s} | {:>8s} | {:>9s} | {:>9s} | {:>15s} | {:>8s} | {:>8s} |'.format(
              'Path GenT', '# Basis','Time','CFA  T', 'Train LB', 'Test LB', 'LB RT','TOT RT') )
        print('-'*99)  
    
    
    def generate_sample_paths(self):
        #----------------------------------------------------------------------
        # Generate and store sample paths
        #----------------------------------------------------------------------
        self.CFA_sample_paths                   = self.mdp.get_sample_path(self.num_CFA_sample_path,self.CFA_random_seed, self.mdp.state_relevance_type)
        self.CFA_paths_rewards                  = self.mdp.get_reward_of_path(self.CFA_sample_paths)
        self.pol_sim_sample_paths               = self.mdp.get_sample_path(self.num_pol_eval_sample_path,self.pol_random_seed)
        self.pol_sim_paths_rewards              = self.mdp.get_reward_of_path(self.pol_sim_sample_paths)


    def LSMN_fit_CFA(self):
        #----------------------------------------------------------------------
        # Fit Continuation Function Approximation  (CFA)
        #----------------------------------------------------------------------
        tot_runtime                 = time.time()    
        start                       = time.time()
        self.print_algorithm_instance_info()
        self.generate_sample_paths()
        path_gen_RT                 = time.time() - start
        CFA_RT,LB_RT                = 0,0
        when_print_results          = 25 
        
        #----------------------------------------------------------------------
        # For loop over all time steps
        CFA_values = np.empty(shape=(self.num_CFA_sample_path,self.num_stages))
        for t in range(self.num_stages-1,-1,-1):
            
            print('| {:>9.2f} | {:>8d} | {:>8d} | {:>9s} | {:>9s} | {:>15s} | {:>8s} | {:>8.1f} |'.format(path_gen_RT,self.num_basis_func,t,'','','','',(time.time()-tot_runtime)/60),end='\r')
            
            #----------------------------------------------------------------------
            # Fit CFA
            if t == self.num_stages-1:                
                start                               = time.time()
                CFA_values[:,t]                     = np.zeros(len(self.CFA_sample_paths[:,t,0])) #self.CFA_paths_rewards[:,t]*self.discount
                CFA_RT                             += time.time() - start
                print('| {:>9.2f} | {:>8d} | {:>8d} | {:>9.1f} | {:>9s} | {:>15s} | {:>8s} | {:>8.1f} |'.format(path_gen_RT,self.num_basis_func,t,CFA_RT,'','','',(time.time()-tot_runtime)/60),end='\r')
            
            elif t == self.num_stages-2:   
                CFA_values[:,t]                     = self.CFA_paths_rewards[:,t+1]*self.discount
            
            else:
                start                               = time.time()
                state_list                          = self.CFA_sample_paths[:,t,:]
                feature_matrix                      = self.basis_func.eval_basis(state_list)
                self.basis_func_coef_matrix[:,t]    = lstsq_jit(feature_matrix,CFA_values[:,t+1])
                CFA_values[:,t]                     = np.maximum(self.CFA_paths_rewards[:,t],feature_matrix@self.basis_func_coef_matrix[:,t]*self.discount)
                CFA_RT                             += time.time() - start
                
                if t%when_print_results==0:
                    print('| {:>9.2f} | {:>8d} | {:>8d} | {:>9.1f} | {:>9s} | {:>15s} | {:>8s} | {:>8.1f} |'.format(path_gen_RT,self.num_basis_func,t,CFA_RT,'','','',(time.time()-tot_runtime)/60),end='\r')
                    
        
        #----------------------------------------------------------------------
        # Compute lower bound confidence interval
        train_LB_stat = self.get_policy_from_continue_func(CFA_values,self.CFA_sample_paths,self.CFA_paths_rewards )                 
        print('| {:>9.2f} | {:>8d} | {:>8d} | {:>9.1f} | {:>9.2f} | {:>15s} | {:>8s} | {:>8.1f} |'.format(path_gen_RT,self.num_basis_func,t,CFA_RT,train_LB_stat[0][0],'','',(time.time()-tot_runtime)/60),end='\r')
        start       = time.time()
        LB_stat     = self.simulate_CVFA_policy()         
        LB_RT      += time.time() - start
        
        print('| {:>9.2f} | {:>8d} | {:>8d} | {:>9.1f} | {:>9.2f} | {:>15.2f} | {:>8.1f} | {:>8.1f} |'.format(path_gen_RT,self.num_basis_func,t,CFA_RT,train_LB_stat[0][0],LB_stat[0],LB_RT,(time.time()-tot_runtime)/60),end='\n')
        self.output_handler.append_to_outputs(algorithm_name            = 'LSM',
                                  basis_seed                            = self.basis_func.basis_func_random_state,
                                  num_basis_func                        = self.num_basis_func,
                                  num_constr                            = self.num_CFA_sample_path,
                                  ALP_con_runtime                       = np.nan,
                                  FALP_obj                              = np.nan,
                                  ALP_slv_runtime                       = np.nan,
                                  train_LB_mean                         = train_LB_stat[0][0],
                                  train_LB_SE                           = train_LB_stat[0][3],
                                  test_LB_mean                          = LB_stat[0],
                                  test_LB_SE                            = LB_stat[3], 
                                  test_LB_runtime                       = (time.time()-start)/60,  
                                  total_runtime                         = (time.time()-tot_runtime)/60)        

        print('-'*99)
        return True  


    def simulate_CVFA_policy(self):
        #----------------------------------------------------------------------
        # Construct policy from a value function and perform inner sampling
        #----------------------------------------------------------------------
        continue_value_list   = np.zeros((len(self.pol_sim_paths_rewards),self.num_stages))
        reward           = []
        eliminated_paths = []
        stopping_time = np.zeros(len(self.pol_sim_sample_paths))
        for t in range(self.num_stages):
            feature_matrix      = self.basis_func.eval_basis(self.pol_sim_sample_paths[:,t,:])
        
            if t == self.num_stages-1:
                continue_value      = np.zeros_like(self.pol_sim_paths_rewards[:,t] )
            else:
                continue_value      = feature_matrix@self.basis_func_coef_matrix[:,t]
                
            immediate_reward    = self.pol_sim_paths_rewards[:,t] 
            stopping_time       = np.less_equal(continue_value,immediate_reward)
            path_to_stop        = np.setdiff1d(np.nonzero(stopping_time)[0],eliminated_paths)
            
            if len(path_to_stop)>0:
            
                reward.extend([self.pol_sim_paths_rewards[_,t]*(self.discount**(t)) for _ in path_to_stop])
                eliminated_paths.extend(path_to_stop)

            continue_value_list[:,t] = continue_value
            
        last_stage_stop =np.setdiff1d(range(len(self.pol_sim_sample_paths)),eliminated_paths)
        T = self.num_stages
        reward.extend([self.pol_sim_paths_rewards[_,T-1]*(self.discount**(T-1)) for _ in last_stage_stop])   
    
        return mean_confidence_interval(reward) 

    def get_policy_from_continue_func(self,continue_func, paths_state, paths_rewards):
        #----------------------------------------------------------------------
        # Construct policy from a value function and perform inner sampling
        #----------------------------------------------------------------------
        reward              = []
        eliminated_paths    = []
        stopping_time       = np.zeros(len(paths_state))
        pol_visited_state   = [[] for _ in range(self.num_stages)]
            
        for t in range(self.num_stages):
            
            immediate_reward            = paths_rewards[:,t] 
            continue_value              = continue_func[:,t]
            state_list                  = paths_state[:,t]
            
            stopping_time               = np.less_equal(continue_value, immediate_reward)
            path_to_stop                = np.setdiff1d(np.nonzero(stopping_time)[0], eliminated_paths)
            pol_visited_state[t]        = [state_list[_] for _ in np.setdiff1d(range(len(state_list)),eliminated_paths)]
            
            if len(path_to_stop)>0:
                reward.extend([paths_rewards[_,t]*(self.discount**(t)) for _ in path_to_stop])
                eliminated_paths.extend(path_to_stop)
            
        last_stage_stop     = np.setdiff1d(range(len(paths_state)),eliminated_paths)
        T                   = self.num_stages
        reward.extend([paths_rewards[_,T-1]*(self.discount**(T-1)) for _ in last_stage_stop])   
        return mean_confidence_interval(reward),pol_visited_state
    
    
    
    
    