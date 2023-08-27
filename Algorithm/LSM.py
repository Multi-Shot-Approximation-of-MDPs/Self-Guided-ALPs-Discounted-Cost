# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
import numpy as np
from numpy.linalg import lstsq
import time
import warnings

from utils import mean_confidence_interval,make_text_bold,output_handler_option_pricing
from BasisFunction.berOptSpecificBasisFunction import BerOptBasisFunction
from Bound.informationRelaxation import InformationRelaxation
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@jit 
def lstsq_jit(X,y):
    #--------------------------------------------------------------------------
    # Jited least seqaures
    #--------------------------------------------------------------------------
    return lstsq(a=X,b=y)[0]


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
        self.num_stages                         = self.mdp.num_stages
        self.discount                           = instance_conf['mdp_conf']['discount']
    
        self.train_random_seed                  = instance_conf['LSM_conf']['train_random_seed'] 
        self.test_random_seed                   = instance_conf['LSM_conf']['test_random_seed'] 

        self.CFA_basis_func                     = [self.instance_conf['basis_func_conf']['basis_func'](self.instance_conf) for _ in range(self.num_stages+1)]
        self.CFA_basis_coef                     = None
        self.CFA_num_basis_func                 = self.CFA_basis_func[0].num_basis_func
        self.CFA_num_train_sample_path:int      = instance_conf['LSM_conf']['CFA_num_train_sample_path']
        self.CFA_num_test_sample_path:int       = instance_conf['LSM_conf']['CFA_num_test_sample_path']
        
        self.optimizer                          = lstsq_jit

        self.output_handler                     = output_handler_option_pricing(instance_conf)
        self.print_len                          = 128

        self.IR_inner_sample_seed               =  instance_conf['IR_conf']['inner_sample_seed']
    
    def optimize(X,y):
        return lstsq(a=X,b=y)[0]
        
        
    def print_algorithm_instance_info(self,algo_name:str):
        #----------------------------------------------------------------------
        # Print to users some info
        #----------------------------------------------------------------------
        print('\n')
        print('='*self.print_len)
        print('Objective type:              \t'     + make_text_bold('Reward Max'))
        print('MDP name:                    \t'     + make_text_bold(self.mdp.mdp_name))
        print('Algorithm name:              \t'     + make_text_bold(algo_name))
        print('Instance number:             \t'     + make_text_bold(self.mdp.instance_number))
        print('Basis function type:         \t'     + make_text_bold(self.CFA_basis_func[0].basis_func_type))
        print('State relevance:             \t'     + make_text_bold(self.mdp.state_relevance_type))
        print('Underlying random seed:      \t'     + make_text_bold(str(self.CFA_basis_func[0].basis_func_random_state)))
        print('='*self.print_len)
        
        if algo_name == 'LSMN':
            print('| {:>8s} | {:>8s} | {:>20s} | {:>20s} | {:>20s} | {:>20s} | {:>10s} |'.format(
                  '# Basis','Stage', 'Train LB', 'Test LB', 'UB (No Penalty)', 'UB (CFA Penalty)', 'Opt Gap') )
        elif algo_name == 'LSML':
            print('| {:>8s} | {:>8s} | {:>20s} | {:>20s} | {:>20s} | {:>20s} | {:>10s} |'.format(
                  '# Basis','Stage', 'Train LB', 'Test LB', 'UB (No Penalty)', 'UB (CFA Penalty)', 'Opt Gap') )
        print('-'*self.print_len)  

    
    def generate_sample_paths(self):

        self.train_sample_paths      = self.mdp.get_sample_path(self.CFA_num_train_sample_path,self.train_random_seed,self.mdp.state_relevance_type)
        self.train_paths_rewards     = self.mdp.get_reward_of_path(self.train_sample_paths)
        self.test_sample_paths       = self.mdp.get_sample_path(self.CFA_num_test_sample_path,self.test_random_seed)
        self.test_paths_rewards      = self.mdp.get_reward_of_path(self.test_sample_paths)

    
    
    
    def LSMN_fit_CFA(self):
        """
            LSMN algorithm for fitting continuation function approximation(CFA)
            Algorithm 1 in Nadarajah et al. 2017.
            
            Compute CFA value at time t using 
                1. dynamic program: C(s_t) = discount * E[ max_{a}[ r(s_{t+1},a) + C(s_{t+1})] | s_t]
                2. CFA:             C(s_t) ≈ \beta_t \Phi (s_t)
                3. SAA:             E[ max_{a}[ r(s_{t+1},a) + C(s_{t+1})] | s_t] ≈ max_{a}[ r(s^i_{t+1},a) + C(s^i_{t+1})]
        """

        CFA_values                      = np.empty(shape=(self.CFA_num_train_sample_path,self.num_stages))
        self.CFA_basis_coef             = np.empty(shape=(self.CFA_num_basis_func ,self.num_stages+1))
       
        # Backward induction to compute CFA
        # t=0,1,2,...,num_stages
        for t in range(self.num_stages,-1,-1):

            print('| {:>8d} | {:>8d} |'.format(self.CFA_num_basis_func,t),end='\r')
            if t == self.num_stages:  
                # Terminal CFA = 0, so CFA_basis_coef=0
                self.CFA_basis_coef[:,t]    = np.zeros(self.CFA_num_basis_func)
            
            else:
                next_state_list             = self.train_sample_paths[:,t+1,:]
                next_state_feature_matrix   = self.CFA_basis_func[t+1].eval_basis(next_state_list)
                CFA_values[:,t]             = self.discount*np.maximum(self.train_paths_rewards[:,t+1], next_state_feature_matrix @ self.CFA_basis_coef[:,t+1])
                state_list                  = self.train_sample_paths[:,t,:]
                feature_matrix              = self.CFA_basis_func[t].eval_basis(state_list)
                self.CFA_basis_coef[:,t]    = self.optimizer(feature_matrix, CFA_values[:,t])
            
        return True 
    


    def get_policy_from_CFA(self, mode:str):
        """         Policy simulation from CFA          """
        
        if mode == 'train':
            paths_state     = self.train_sample_paths
            paths_rewards   = self.train_paths_rewards
            num_paths       = self.CFA_num_train_sample_path
            
        elif mode == 'test':
            paths_state     = self.test_sample_paths
            paths_rewards   = self.test_paths_rewards
            num_paths       = self.CFA_num_test_sample_path
            
        else:
            Exception('Undefined mode!')
        
        reward              = []
        eliminated_paths    = []
        stopping_time       = np.zeros(num_paths)
        pol_visited_state   = [[] for _ in range(self.num_stages+1)]
        
        for t in range(self.num_stages+1):
            immediate_reward            = paths_rewards[:,t] 
            state_list                  = paths_state[:,t,:]
            continue_value              = self.CFA_basis_func[t].eval_basis(state_list) @ self.CFA_basis_coef[:,t]
            stopping_time               = np.less_equal(continue_value, immediate_reward)
            path_to_stop                = np.setdiff1d(np.nonzero(stopping_time)[0], eliminated_paths) #Paths should be eliminated and are not eliminated yet
            pol_visited_state[t]        = [state_list[_] for _ in np.setdiff1d(range(len(state_list)),eliminated_paths)]
            
            if len(path_to_stop) > 0:
                reward.extend([paths_rewards[_,t]*(self.discount**(t)) for _ in path_to_stop])
                eliminated_paths.extend(path_to_stop)
        
        last_stage_stop = np.setdiff1d(range(len(paths_state)),eliminated_paths) 
        assert len(last_stage_stop) ==0  
        return mean_confidence_interval(reward),pol_visited_state    
    


    def LSMN(self): 
        """
            LSMN Full Algorithm
                - Sampling basis functions or using fixed basis functions
                - Fitting CFA
                - Simulating greedy policy to compute lower bound on the optimal reward
                - Computing dual bounds to compute upper bound on the optimal reward
        """
        
        tot_runtime = time.time()            
        self.print_algorithm_instance_info('LSMN')
        
        """     Generate sample paths for training CFA, testing greedy policy, and computing bound      """
        start = time.time()
        self.generate_sample_paths()
        path_gen_RT = time.time() - start
        
        print('| {:>8d} |'.format(self.CFA_num_basis_func),end='\r')   
        
        
        
        
        
        """ 
            For loop is active if basis functions defining VFA are set to random bases.
            Otherwise, max_basis_num is equal to batch_size, and thus there is no loop.
        """

            
        """ [1] Fit CFA """

        start = time.time()
        self.LSMN_fit_CFA()
        CVFA_fitting_runtime = time.time() - start
        
        """ [2] Simulate greedy policy """
        start = time.time()
        train_LB_stat,_               = self.get_policy_from_CFA(mode='train') 
        print('| {:>8d} | {:>8d} | {:>20.4f} |'.format(self.CFA_num_basis_func,0,train_LB_stat[0]),end='\r')   
        
        # Switch to testing phase
        test_LB_stat,_                = self.get_policy_from_CFA(mode='test') 
        lower_bound_runtime         = time.time() - start
        
        print('| {:>8d} | {:>8d} | {:>20.4f} | {:>20.4f} |'.format(self.CFA_num_basis_func,0,train_LB_stat[0],test_LB_stat[0]),end='\r')   
            
   
    
        """ [3] Compute Infromation Relaxation Upper Bounds """
        

        start = time.time()
        self.mdp.fix_inner_samples(self.mdp.inner_sample_size,self.CFA_num_test_sample_path,self.IR_inner_sample_seed)
        
        self.IR = InformationRelaxation(self.instance_conf,self.mdp)
        
        
        self.IR.set_sample_path(  num_sample_path       = self.CFA_num_test_sample_path,
                                  sample_path           = self.test_sample_paths,
                                  sample_path_reward    = self.test_paths_rewards)
                    
        self.IR.set_basis_func_coef(self.CFA_num_basis_func,
                                    self.CFA_basis_coef,
                                    self.CFA_basis_func,
                                    0)
        
        # Compute dual bound with no penalty
        dual_bound_no_penalty_stat = self.IR.get_dual_bound_no_penalty()
        
        print('| {:>8d} | {:>8d} | {:>20.4f} | {:>20.4f} | {:>20.4f} |'.format(self.CFA_num_basis_func,0,train_LB_stat[0],
                                                                               test_LB_stat[0],dual_bound_no_penalty_stat[0]),end='\r')   
       
        dual_bound_with_penalty_stat  = self.IR.get_dual_bound_from_CFA()
        opt_gap                       = 100*(dual_bound_with_penalty_stat[0] - test_LB_stat[0])/dual_bound_with_penalty_stat[0]
                                                                                         
        print('| {:>8d} | {:>8d} | {:>20.4f} | {:>20.4f} | {:>20.4f} | {:>20.4f} | {:>10.2f} |'.format(self.CFA_num_basis_func,0,train_LB_stat[0],
                                                                               test_LB_stat[0],dual_bound_no_penalty_stat[0],
                                                                               dual_bound_with_penalty_stat[0],
                                                                               opt_gap),end='\n')
        
        upp_bound_runtime = time.time() - start


        # """ [4] Store results """
        self.output_handler.append_to_outputs(
                          algorithm_name                        = 'LSM',
                          state_relevance_type                  = self.mdp.state_relevance_type,
                          basis_func_type                       = self.CFA_basis_func[0].basis_func_type,
                          basis_seed                            = self.CFA_basis_func[0].basis_func_random_state,
                          basis_bandwidth_str                   = ''.join(str(x) for x in self.CFA_basis_func[0].bandwidth),
                          abs_val_upp_bound                     = str(float('inf')),
                          max_basis_num                         = self.CFA_basis_func[0].max_basis_num,
                          num_basis_func                        = self.CFA_num_basis_func,
                          num_train_samples                     = self.CFA_num_train_sample_path,
                          num_test_samples                      = self.CFA_num_test_sample_path,
                          num_inner_samples                     = self.mdp.inner_sample_size,
                          train_LB_mean                         = train_LB_stat[0],
                          train_LB_SE                           = train_LB_stat[3],
                          test_LB_mean                          = test_LB_stat[0],
                          test_LB_SE                            = test_LB_stat[3], 
                          dual_bound_no_penalty_mean            = dual_bound_no_penalty_stat[0],
                          dual_bound_no_penalty_se              = dual_bound_no_penalty_stat[3],
                          dual_bound_with_penalty_mean          = dual_bound_with_penalty_stat[0],
                          dual_bound_with_penalty_se            = dual_bound_with_penalty_stat[3],
                          best_upper_bound                      = dual_bound_with_penalty_stat[0],
                          opt_gap                               = opt_gap,
                          path_gen_runtime                      = path_gen_RT,
                          upp_bound_runtime                     = upp_bound_runtime,
                          lower_bound_runtime                   = lower_bound_runtime,
                          CVFA_fitting_runtime                  = CVFA_fitting_runtime,
                          total_runtime                         = (time.time()-tot_runtime)
                          ) 

            
        print('-'*self.print_len)
        print('{:>55s}: {:>15.3f} (s)'.format('Runtime of simulating train, test, and inner samples',path_gen_RT))
        print('{:>55s}: {:>15.3f} (s)'.format('Total runtime', time.time() - tot_runtime))
        print('{:>55s}: {:>15.3f} (%)'.format('Terminal optimality gap', opt_gap))
        print('-'*self.print_len)


