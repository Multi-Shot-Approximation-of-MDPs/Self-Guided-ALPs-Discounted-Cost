# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from tqdm import tqdm
import numpy as np
import gurobipy as gb
from utils import output_handler_option_pricing
import time
from utils import mean_confidence_interval
from utils import make_text_bold
from Wrapper.gurobiWrapperBerOpt import gurobi_LP_wrapper
import gc


class SelfGuidedALPs:
    
    def __init__(self,instance_conf):
        #----------------------------------------------------------------------
        # Initialization
        #----------------------------------------------------------------------
        self.mdp                                = instance_conf['mdp_conf']['mdp'](instance_conf)
        self.num_stages                         = self.mdp.num_stages
        self.basis_func                         = [instance_conf['basis_func_conf']['basis_func'](instance_conf) for _ in range(self.num_stages-1)]
        
        for t in range(self.num_stages-1):
            self.basis_func[t].basis_func_random_state = self.basis_func[t].basis_func_random_state + t*self.basis_func[t].preprocess_batch
        
        self.num_VFA_sample_path:int            = instance_conf['mdp_conf']['num_CFA_sample_path']
        self.num_pol_eval_sample_path:int       = instance_conf['mdp_conf']['num_pol_eval_sample_path']
        self.num_basis_func                     = self.basis_func[0].num_basis_func    
        self.discount                           = instance_conf['mdp_conf']['discount']
        self.basis_func_coef_matrix             = None     
        self.VFA_values                         = np.empty(shape=(self.num_VFA_sample_path,self.num_stages))
        self.VFA_random_seed                    = instance_conf['mdp_conf']['CFA_random_seed'] 
        self.pol_random_seed                    = instance_conf['mdp_conf']['pol_random_seed'] 
        self.FALP                               = gb.Model()
        self.FALP_var                           = None
        self.num_cpu_core                       =  instance_conf['mdp_conf']['num_cpu_core']
        self.FALP.setParam('OutputFlag',False)
        self.FALP.setParam('LogFile','Output/groubi_log_file.log')
        self.output_handler                     = output_handler_option_pricing(instance_conf)
        self.ALP_solver                         = gurobi_LP_wrapper(instance_conf['solver_conf'])
        self.print_len                          = 169
        
        
    def print_algorithm_instance_info(self,algo_name,state_relevance_name):
        #----------------------------------------------------------------------
        # Used to show what algorithm is running to users
        #----------------------------------------------------------------------
        print('\n')
        print('='*self.print_len)
        print('Instance number:             \t'   + make_text_bold(self.mdp.instance_number))
        print('Algorithm name:              \t'   + make_text_bold(algo_name))
        print('Basis function type:         \t'   + make_text_bold(self.basis_func[0].basis_func_type))
        print('State relevance:             \t'   + make_text_bold(state_relevance_name))
        print('Random basis seed:           \t'   + make_text_bold(str(self.basis_func[0].basis_func_random_state)))
        print('='*self.print_len)
        if not algo_name == 'Computing Expected VFAs':
            print('| {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} |'.format(
                  'Path GenT', '# Basis','Time','ALP ConT', 'ALP SlvT','ALP OBJ', 'Train LB', 'Test LB', 'Test LBT','UB','UBT','Train Gap','Test Gap','TOT RT') )
    
            print('-'*self.print_len)  
            
    
    def generate_sample_paths(self):
        #----------------------------------------------------------------------
        # Generate sample paths and fix them throughout
        #----------------------------------------------------------------------
        self.VFA_sample_paths           = self.mdp.get_sample_path(self.num_VFA_sample_path,self.VFA_random_seed,self.mdp.state_relevance_type)
        self.VFA_paths_rewards          = self.mdp.get_reward_of_path(self.VFA_sample_paths)
        self.pol_sim_sample_paths       = self.mdp.get_sample_path(self.num_pol_eval_sample_path,self.pol_random_seed)
        self.pol_sim_paths_rewards      = self.mdp.get_reward_of_path(self.pol_sim_sample_paths)
        
    def get_policy_from_continue_func(self,continue_func, paths_state, paths_rewards):
        #----------------------------------------------------------------------
        # Construct policy from a continuation function
        #----------------------------------------------------------------------
        reward              = []
        eliminated_paths    = []
        stopping_time       = np.zeros(len(paths_state))
        pol_visited_state   = [[] for _ in range(self.num_stages)]
            
        for t in range(self.num_stages):
            #------------------------------------------------------------------
            # Compare the value of stopping and continuing
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


    def simulation_VFA_policy_compute_expectations(self,basis_func_coef_matrix,use_train_paths=False):
        #----------------------------------------------------------------------
        # Construct policy from a value function and perform inner sampling
        #----------------------------------------------------------------------
        reward              = []
        eliminated_paths    = []
        pol_visited_state   = [[] for _ in range(self.num_stages)]
        
        if not use_train_paths:
            pol_sim_paths_rewards = self.pol_sim_paths_rewards
            pol_sim_sample_paths  = self.pol_sim_sample_paths
        else:
            pol_sim_paths_rewards = self.VFA_paths_rewards
            pol_sim_sample_paths  = self.VFA_sample_paths
            
        stopping_time       = np.zeros(len(pol_sim_sample_paths))
        for t in range(self.num_stages):
            #------------------------------------------------------------------
            # Compare the value of stopping and continuing
            immediate_reward                    = pol_sim_paths_rewards[:,t] 
            state_list                          = pol_sim_sample_paths[:,t,:]            
            if t == self.num_stages-1:
                continue_value = np.zeros_like(immediate_reward) 
            elif t == self.num_stages-2:
                continue_value = self.pol_sim_paths_rewards[:,t+1]*self.discount
            else:
                #--------------------------------------------------------------
                # Performing inner sampling to estimate expected value
                inner_next_state_samples                                = self.mdp.get_inner_samples(state_list) 
                inner_next_state_samples                                = np.reshape(inner_next_state_samples, newshape=(len(state_list),self.mdp.inner_sample_size, self.mdp.dim_state))
                discounted_expected_next_state_feature_matrix_new_basis = self.basis_func[t+1].list_expected_basis(inner_next_state_samples)*self.discount
                continue_value                                          = discounted_expected_next_state_feature_matrix_new_basis@basis_func_coef_matrix[:,t+1]
        
            stopping_time                       = np.less_equal(continue_value, immediate_reward)
            path_to_stop                        = np.setdiff1d(np.nonzero(stopping_time)[0], eliminated_paths)
            pol_visited_state[t]                = [state_list[_] for _ in np.setdiff1d(range(len(state_list)), eliminated_paths)]
            
            if len(path_to_stop)>0:
                reward.extend([pol_sim_paths_rewards[_,t]*(self.discount**(t)) for _ in path_to_stop])
                eliminated_paths.extend(path_to_stop)
                
        last_stage_stop     = np.setdiff1d(range(len(pol_sim_sample_paths)),eliminated_paths)
        T                   = self.num_stages
        reward.extend([pol_sim_paths_rewards[_,T-1]*(self.discount**(T-1)) for _ in last_stage_stop])   
        return mean_confidence_interval(reward),pol_visited_state



    def FALP_non_adaptive_fit_fix_basis(self):
        #----------------------------------------------------------------------
        # ALP with fixed basis functions used in https://towardsdatascience.com/how-to-simulate-financial-portfolios-with-python-d0dc4b52a278
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        # Show the algorithm's name to a user and initialize parameters
        self.print_algorithm_instance_info('FALP Non-Adaptive', self.mdp.state_relevance_type)
        tot_time                                        = time.time()
        path_gen_RT                                     = time.time()
        self.generate_sample_paths()
        path_gen_RT                                     = time.time() - path_gen_RT
        cur_feature_matrix                              = [None for _  in range(self.num_stages)] 
        discounted_expected_next_state_feature_matrix   = [None for _  in range(self.num_stages)]
        num_basis                                       = self.basis_func[0].num_basis_func
        start                                           = time.time()
        
        
        #----------------------------------------------------------------------
        # [1] Construc ALP Components      
        for t in range(self.num_stages-1,-1,-1):
            print('| {:>9.2f} | {:>9d} | {:>9d} |'.format(path_gen_RT,num_basis,t),end='\r')
            state_list              = np.array(self.VFA_sample_paths[:,t,:])
            cur_reward              = self.VFA_paths_rewards[:,t]
            if  t <= self.num_stages-2:
                new_cols_in_cur_feature_matrix = self.basis_func[t].eval_basis(state_list)
        
            if t == self.num_stages-1:
                self.ALP_solver.set_terminal_val(cur_reward*self.discount)
                print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9.2f} |'.format(
                        path_gen_RT,num_basis,t,time.time()-start,'','','','','',(time.time()-tot_time)/60),end='\r')   
        
            elif t == self.num_stages-2:
                self.ALP_solver.set_up_variables(self.basis_func[0].batch_size,t)
                cur_feature_matrix[t]                                   = new_cols_in_cur_feature_matrix
                self.ALP_solver.add_FALP_constr(new_cols_in_cur_feature_matrix,None,cur_reward,t,warm_start=None)
                
            else:
                self.ALP_solver.set_up_variables(self.basis_func[0].batch_size,t)
                cur_feature_matrix[t]                                   = new_cols_in_cur_feature_matrix
                inner_next_state_samples                                = self.mdp.get_inner_samples(state_list) 
                inner_next_state_samples                                = np.reshape(inner_next_state_samples, newshape=(len(state_list),self.mdp.inner_sample_size, self.mdp.dim_state))
                discounted_expected_next_state_feature_matrix_new_basis = self.basis_func[t+1].list_expected_basis(inner_next_state_samples)*self.discount
                discounted_expected_next_state_feature_matrix[t]        = discounted_expected_next_state_feature_matrix_new_basis
                
                self.ALP_solver.add_FALP_constr(new_cols_in_cur_feature_matrix,discounted_expected_next_state_feature_matrix_new_basis,cur_reward,t,warm_start=None)
                print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9.2f} |'.format(
                        path_gen_RT,num_basis,t,time.time()-start,'','','','','',(time.time()-tot_time)/60),end='\r')   
              
            gc.collect()
            del state_list
            del cur_reward
            if  t <= self.num_stages-2:
                del new_cols_in_cur_feature_matrix
                
                
        FALP_obj = np.mean(self.basis_func[0].eval_basis(np.array(self.VFA_sample_paths[:,0,:])), axis=0)  
        if num_basis == self.basis_func[0].batch_size:
            self.ALP_solver.set_objective(FALP_obj,False, None)
        else:
            self.ALP_solver.set_objective(FALP_obj,False, self.basis_func_coef_matrix)
        
        
        #----------------------------------------------------------------------
        # [2] Solve ALP
        FALP_con_RT                         = time.time()-start
        start                               = time.time()
        self.ALP_solver.prepare()
        self.ALP_solver.optimize(0)
        self.basis_func_coef_matrix         = self.ALP_solver.get_optimal_solution()
        FALP_opt_obj_val                    = self.ALP_solver.get_optimal_value()
        FALP_slv_RT                         = time.time()-start
        print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9.1f} | {:>9.1f} | {:>9s} | {:>9s} | {:>9s} | {:>9.2f} |'.format(
                    path_gen_RT,num_basis,t,FALP_con_RT,FALP_slv_RT,FALP_opt_obj_val,'','','',(time.time()-tot_time)/60),end='\r')
         
        
        #----------------------------------------------------------------------
        # [3] Use ALP solution to form continuation function
        continue_function                   = [None for _  in range(self.num_stages)] 
        for t in range(self.num_stages):  
            if t == self.num_stages-1:
                continue_function[t]        = np.zeros_like(continue_function[0])
            elif  t == self.num_stages-2:
                continue_function[t]        = self.VFA_paths_rewards[:,t+1]*self.discount
            else:
                continue_function[t]        = discounted_expected_next_state_feature_matrix[t] @ self.basis_func_coef_matrix[:,t+1]
       
        #----------------------------------------------------------------------
        # [4] Compute pay off on train set and store results
        continue_function                   = np.array(continue_function).T
        train_LB_stat,pol_visited_state     = self.get_policy_from_continue_func(continue_function, self.VFA_sample_paths, self.VFA_paths_rewards)
        print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9.1f} | {:>9.1f} | {:>9.1f} | {:>9s} | {:>9s} | {:>9.2f} |'.format(
                    path_gen_RT,num_basis,t,FALP_con_RT,FALP_slv_RT,FALP_opt_obj_val,train_LB_stat[0],'','',(time.time()-tot_time)/60),end='\r') 
        
        self.output_handler.append_to_outputs(algorithm_name        = 'FALP_non_adaptive',
                                          basis_seed                = self.basis_func[0].basis_func_random_state,
                                          num_basis_func            = num_basis,
                                          num_constr                = self.num_VFA_sample_path,
                                          FALP_obj                  = FALP_opt_obj_val,
                                          ALP_con_runtime           = FALP_con_RT/60,
                                          ALP_slv_runtime           = FALP_slv_RT/60,
                                          train_LB_mean             = train_LB_stat[0],
                                          train_LB_SE               = train_LB_stat[3],
                                          test_LB_mean              = np.nan,
                                          test_LB_SE                = np.nan, 
                                          test_LB_runtime           = np.nan,
                                          total_runtime             = (time.time()-tot_time)/60
                                          )
            
        #----------------------------------------------------------------------
        # [5] Test Policy Performance on the test set and store results
        start                           = time.time()
        test_LB_stat,_                  = self.simulation_VFA_policy_compute_expectations(self.basis_func_coef_matrix,False)
        test_LB_RT                      = time.time()-start
        
        print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9.1f} | {:>9.1f} | {:>9.1f} | {:>9.1f} | {:>9.1f} | {:>9.2f} |'.format(
                    path_gen_RT,num_basis,t,FALP_con_RT,FALP_slv_RT,FALP_opt_obj_val,train_LB_stat[0],test_LB_stat[0],test_LB_RT,(time.time()-tot_time)/60),end='\n')     

        self.output_handler.append_to_outputs(algorithm_name            = 'FALP_non_adaptive',
                                              basis_seed                = self.basis_func[0].basis_func_random_state,
                                              num_basis_func            = num_basis,
                                              num_constr                = self.num_VFA_sample_path,
                                              FALP_obj                  = FALP_opt_obj_val,
                                              ALP_con_runtime           = FALP_con_RT/60,
                                              ALP_slv_runtime           = FALP_slv_RT/60,
                                              train_LB_mean             = train_LB_stat[0],
                                              train_LB_SE               = train_LB_stat[3],
                                              test_LB_mean              = test_LB_stat[0],
                                              test_LB_SE                = test_LB_stat[3], 
                                              test_LB_runtime           = test_LB_RT/60,
                                              total_runtime             = (time.time()-tot_time)/60
                                              )
        
        self.ALP_solver.re_initialize_solver()
        print('-'*self.print_len)   
        
        
    def compute_expected_basis_func(self,seed_):
        #----------------------------------------------------------------------
        # Compute expected values of basis functions
        #----------------------------------------------------------------------
        self.print_algorithm_instance_info('Computing Expected VFAs', self.mdp.state_relevance_type)
        self.generate_sample_paths()
        path = 'Output/'+self.mdp.mdp_name+'/instance_'+self.mdp.instance_number+'/seed_'+str(seed_)+'/'
        train_expected_basis_func = {}
        
        #----------------------------------------------------------------------
        # Compute components in each stage
        for t in tqdm(range(self.num_stages-2,-1,-1),ncols=self.print_len,leave=True,desc='Train'):
            state_list = np.array(self.VFA_sample_paths[:,t,:])       
            self.basis_func[t].form_orthogonal_bases(state_list,
                                                     path+'/basis_params/',
                                                     t,
                                                     False)   
            
            if t < self.num_stages-2:
                inner_next_state_samples        = self.mdp.get_inner_samples(state_list) 
                evals                           = self.basis_func[t+1].compute_expected_basis_func(inner_next_state_samples,
                                                                    self.num_VFA_sample_path,
                                                                    self.mdp.inner_sample_size,
                                                                    self.discount,
                                                                    t+1,
                                                                    path +'/train/',
                                                                    True)
                
                train_expected_basis_func.update({'expected_basis_func_'+str(t+1) : evals})
        
        np.savez_compressed(path+'discounted_expected_VFA_train_batch_'+str(self.basis_func[0].preprocess_batch) ,train_expected_basis_func)
        
        #----------------------------------------------------------------------
        # Compute components in each stage
        test_expected_basis_func = {}
        for t in tqdm(range(self.num_stages-3,-1,-1),ncols=self.print_len,leave=True,desc='Test '):
            state_list                      = np.array(self.pol_sim_sample_paths[:,t,:])
            inner_next_state_samples        = self.mdp.get_inner_samples(state_list) 
            evals                           = self.basis_func[t+1].compute_expected_basis_func(inner_next_state_samples,
                                                              self.num_pol_eval_sample_path,
                                                              self.mdp.inner_sample_size,
                                                              self.discount,
                                                              t+1,
                                                              path+'/test/',
                                                              False)
            
            test_expected_basis_func.update({'expected_basis_func_'+str(t+1) : evals})
                
        np.savez_compressed(path+'discounted_expected_VFA_test_batch_'+str(self.basis_func[0].preprocess_batch) ,test_expected_basis_func)


    def FALP_non_adaptive_fit_VFA(self):
        #----------------------------------------------------------------------
        # FALP implementation
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        # Show the algorithm's name to a user and initialize parameters
        self.print_algorithm_instance_info('FALP Non-Adaptive', self.mdp.state_relevance_type)
        tot_time                                        = time.time()
        path_gen_RT                                     = time.time()
        self.generate_sample_paths()
        path_gen_RT                                     = time.time() - path_gen_RT
        basis_range                                     = np.arange(self.basis_func[0].batch_size, self.basis_func[0].max_basis_num+1,self.basis_func[0].batch_size)
        cur_feature_matrix                              = [None for _  in range(self.num_stages)] 
        discounted_expected_next_state_feature_matrix   = [None for _  in range(self.num_stages)] 
        continue_function                               = [None for _  in range(self.num_stages)] 


        #----------------------------------------------------------------------
        # [1] Load ALP components from file  
        num_times_basis_added               = -1
        for num_basis in basis_range:
            path            = 'Output/'+self.mdp.mdp_name+'/instance_'+self.mdp.instance_number+'/seed_'+str(int(self.VFA_random_seed/10))
            file_name       = path+ '/discounted_expected_VFA_train_batch_' +str(self.basis_func[0].preprocess_batch) 
            expct_VFA       = np.load(file_name+'.npz',allow_pickle=True)
            expct_VFA       = expct_VFA.f.arr_0
            expct_VFA       = expct_VFA.item()
            start                               = time.time()
            num_times_basis_added               += 1 
            print('| {:>9.2f} | {:>9d} |'.format(path_gen_RT,num_basis),end='\r')
 
            #------------------------------------------------------------------
            # [2] Construct ALP via a Gurobi model      
            #------------------------------------------------------------------
            for t in range(self.num_stages-1,-1,-1):
                print('| {:>9.2f} | {:>9d} | {:>9d} |'.format(path_gen_RT,num_basis,t),end='\r')
                state_list              = np.array(self.VFA_sample_paths[:,t,:])
                cur_reward              = self.VFA_paths_rewards[:,t] 
                
                if  t <= self.num_stages-2:
                    if num_times_basis_added == 0:
                        self.basis_func[t].form_orthogonal_bases(state_list,path+'/basis_params/',stage=t,to_load=True)   
                    new_cols_in_cur_feature_matrix = self.basis_func[t].eval_basis(state_list,num_times_basis_added,False)
                    
                if t == self.num_stages-1:
                    self.ALP_solver.set_terminal_val(cur_reward*(self.discount))
                    
                    print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9.2f} |'.format(
                            path_gen_RT,num_basis,t,time.time()-start,'','','','','','','','','',(time.time()-tot_time)/60),end='\r')   
                    
                elif t == self.num_stages-2:
                    if num_times_basis_added == 0:
                        self.ALP_solver.set_up_variables(self.basis_func[0].batch_size,t)
                    else:
                        self.ALP_solver.add_new_variables(self.basis_func[0].batch_size,t)
                    
                    cur_feature_matrix[t] = new_cols_in_cur_feature_matrix if cur_feature_matrix[t] is None \
                                                   else np.concatenate((cur_feature_matrix[t],new_cols_in_cur_feature_matrix),axis=1)
                     
                    self.ALP_solver.add_FALP_constr(new_cols_in_cur_feature_matrix,None,cur_reward,t,warm_start=None)            



                else:
                    if num_times_basis_added == 0:
                        self.ALP_solver.set_up_variables(self.basis_func[0].batch_size, t)
                    else:
                        self.ALP_solver.add_new_variables(self.basis_func[0].batch_size,t)
                    
                    cur_feature_matrix[t]                                   = new_cols_in_cur_feature_matrix if cur_feature_matrix[t] is None \
                                                                                    else np.concatenate((cur_feature_matrix[t],new_cols_in_cur_feature_matrix),axis=1)

                    discounted_expected_next_state_feature_matrix_new_basis = self.basis_func[t+1].load_expected_basis(num_times_basis_added,expct_VFA, t+1)
                    discounted_expected_next_state_feature_matrix[t]        = discounted_expected_next_state_feature_matrix_new_basis \
                                                                                    if discounted_expected_next_state_feature_matrix[t] is None\
                                                                                        else np.concatenate((discounted_expected_next_state_feature_matrix[t],discounted_expected_next_state_feature_matrix_new_basis),axis=1)
                                                                                        
                    self.ALP_solver.add_FALP_constr(new_cols_in_cur_feature_matrix,discounted_expected_next_state_feature_matrix_new_basis,cur_reward,t,warm_start=None)
                    
                    print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9.2f} |'.format(
                            path_gen_RT,num_basis,t,time.time()-start,'','','','','','','','','',(time.time()-tot_time)/60),end='\r')   
                  
                gc.collect()
                del state_list
                del cur_reward
                if  t <= self.num_stages-2:
                    del new_cols_in_cur_feature_matrix
              
            #------------------------------------------------------------------
            # Set objective function   
            FALP_obj = np.mean(self.basis_func[0].eval_basis(np.array(self.VFA_sample_paths[:,0,:]), num_times_basis_added,all_bases=True), axis=0)  
 
            if num_basis == self.basis_func[0].batch_size:
                self.ALP_solver.set_objective(FALP_obj,False,None)
            else:
                self.ALP_solver.set_objective(FALP_obj,False,self.basis_func_coef_matrix)
            
            
            #------------------------------------------------------------------
            # [3] Solve ALP 
            #------------------------------------------------------------------
            FALP_con_RT                         = time.time()-start
            start                               = time.time()
            self.ALP_solver.prepare()
            self.ALP_solver.optimize(num_times_basis_added)
            self.basis_func_coef_matrix         = self.ALP_solver.get_optimal_solution()
            FALP_opt_obj_val                    = self.ALP_solver.get_optimal_value()
            FALP_slv_RT                         = time.time()-start
            
            print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9.2f} |'.format(
                        path_gen_RT,num_basis,t,FALP_con_RT,FALP_slv_RT,FALP_opt_obj_val,'','','','','','','',(time.time()-tot_time)/60),end='\r')
              
            #------------------------------------------------------------------
            # [4] Policy Performance on Train Set
            #------------------------------------------------------------------
            continue_function                   = [None for _  in range(self.num_stages)] 
            for t in range(self.num_stages):  
                if t == self.num_stages-1:
                    continue_function[t]        = np.zeros_like(continue_function[0])
                elif  t == self.num_stages-2:
                    continue_function[t]        = self.VFA_paths_rewards[:,t+1]*self.discount
                else:
                    continue_function[t]        = discounted_expected_next_state_feature_matrix[t] @ self.basis_func_coef_matrix[:,t+1]
                    
            continue_function                   = np.array(continue_function).T
            train_LB_stat,pol_visited_state     = self.get_policy_from_continue_func(continue_function, self.VFA_sample_paths, self.VFA_paths_rewards)
            
            print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9s} | {:>9s} | {:>9s} | {:>9s}| {:>9s} | {:>9s} | {:>9.2f} |'.format(
                        path_gen_RT,num_basis,t,FALP_con_RT,FALP_slv_RT,FALP_opt_obj_val,train_LB_stat[0],'','','','','','',(time.time()-tot_time)/60),end='\n') 
            
            #------------------------------------------------------------------
            # [5]  Policy Performance on Test Set via CFA
            #------------------------------------------------------------------    
            start
            test_continue_function              = [None for _  in range(self.num_stages)] 
            file_name                           = path+ '/discounted_expected_VFA_test_batch_' +str(self.basis_func[0].preprocess_batch) 
            expct_VFA                           = np.load(file_name+'.npz',allow_pickle=True)
            expct_VFA                           = expct_VFA.f.arr_0
            expct_VFA                           = expct_VFA.item()
            
            for t in range(self.num_stages):
                if t == self.num_stages-1:
                    test_continue_function[t]   = np.zeros(self.num_pol_eval_sample_path)
                elif  t == self.num_stages-2:
                    test_continue_function[t]   = self.pol_sim_paths_rewards[:,t+1]*self.discount
                else:
                    test_discounted_expected_next_state_feature_matrix = self.basis_func[t+1].load_expected_basis(num_times_basis_added, expct_VFA,t+1,is_train=False)
                    
                    test_continue_function[t]   = test_discounted_expected_next_state_feature_matrix @ self.basis_func_coef_matrix[:,t+1]

            test_continue_function              = np.array(test_continue_function).T
            test_LB_stat,_                      = self.get_policy_from_continue_func(test_continue_function, self.pol_sim_sample_paths, self.pol_sim_paths_rewards)
            test_LB_RT                          = time.time()-start

            #------------------------------------------------------------------
            # [6] Show results and store them on disk
            #------------------------------------------------------------------ 
            print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9.2f} |'.format(
                path_gen_RT,num_basis,t,FALP_con_RT,FALP_slv_RT,FALP_opt_obj_val,train_LB_stat[0],test_LB_stat[0],test_LB_RT,'','','','',(time.time()-tot_time)/60),end='\n')  
            
            self.output_handler.append_to_outputs(algorithm_name            = 'FALP_non_adaptive',
                                                  basis_seed                = self.basis_func[0].basis_func_random_state,
                                                  num_basis_func            = num_basis,
                                                  num_constr                = self.num_VFA_sample_path,
                                                  FALP_obj                  = FALP_opt_obj_val,
                                                  ALP_con_runtime           = FALP_con_RT/60,
                                                  ALP_slv_runtime           = FALP_slv_RT/60,
                                                  train_LB_mean             = train_LB_stat[0],
                                                  train_LB_SE               = train_LB_stat[3],
                                                  test_LB_mean              = test_LB_stat[0],
                                                  test_LB_SE                = test_LB_stat[3], 
                                                  test_LB_runtime           = test_LB_RT/60,
                                                    total_runtime             = (time.time()-tot_time)/60
                                                    )
            #------------------------------------------------------------------ 
            # Re-initialize solver if FALP is solved iteratively, 
            # and not in one-shot                                         
            self.ALP_solver.re_initialize_solver()
            
            
    def SGFALP_fit_VFA(self):
        #----------------------------------------------------------------------
        # Self-guided FALP Implementation
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        # Show the algorithm's name to a user and initialize parameters
        self.print_algorithm_instance_info('Self-guided FALP', self.mdp.state_relevance_type)
        tot_time                                        = time.time()
        path_gen_RT                                     = time.time()
        self.generate_sample_paths()
        path_gen_RT                                     = time.time() - path_gen_RT
        basis_range                                     = np.arange(self.basis_func[0].batch_size, self.basis_func[0].max_basis_num+1,self.basis_func[0].batch_size)
        cur_feature_matrix                              = [None for _  in range(self.num_stages)] 
        discounted_expected_next_state_feature_matrix   = [None for _  in range(self.num_stages)] 
        continue_function                               = [None for _  in range(self.num_stages)] 

        #----------------------------------------------------------------------
        # [1] Load ALP components from file  
        num_times_basis_added               = -1
        
        for num_basis in basis_range:
            
            path            = 'Output/'+self.mdp.mdp_name+'/instance_'+self.mdp.instance_number+'/seed_'+str(int(self.VFA_random_seed/10))
            file_name       = path+ '/discounted_expected_VFA_train_batch_' +str(self.basis_func[0].preprocess_batch) 
            expct_VFA       = np.load(file_name+'.npz',allow_pickle=True)
            expct_VFA       = expct_VFA.f.arr_0
            expct_VFA       = expct_VFA.item()
            start                               = time.time()
            num_times_basis_added               += 1 
            print('| {:>9.2f} | {:>9d} |'.format(path_gen_RT,num_basis),end='\r')
 

            #------------------------------------------------------------------
            # [2] Construct ALP via a Gurobi model      
            #------------------------------------------------------------------
            for t in range(self.num_stages-1,-1,-1):
                print('| {:>9.2f} | {:>9d} | {:>9d} |'.format(path_gen_RT,num_basis,t),end='\r')
                state_list              = np.array(self.VFA_sample_paths[:,t,:])
                cur_reward              = self.VFA_paths_rewards[:,t] 
                
                # --> Load basis functions
                if  t <= self.num_stages-2:
                    if num_times_basis_added == 0:
                        self.basis_func[t].form_orthogonal_bases(state_list,path+'/basis_params/',stage=t,to_load=True)   
        
                    new_cols_in_cur_feature_matrix = self.basis_func[t].eval_basis(state_list,num_times_basis_added,False)
                    
                if t == self.num_stages-1:
                    self.ALP_solver.set_terminal_val(cur_reward*(self.discount))
                    print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9.2f} |'.format(
                            path_gen_RT,num_basis,t,time.time()-start,'','','','','','','','','',(time.time()-tot_time)/60),end='\r')   
                    
                elif t == self.num_stages-2:
                    if num_times_basis_added == 0:
                        self.ALP_solver.set_up_variables(self.basis_func[0].batch_size,t)
                    else:
                        self.ALP_solver.add_new_variables(self.basis_func[0].batch_size,t)
                    
                    cur_feature_matrix[t] = new_cols_in_cur_feature_matrix if cur_feature_matrix[t] is None \
                                                   else np.concatenate((cur_feature_matrix[t],new_cols_in_cur_feature_matrix),axis=1)
                     
                    self.ALP_solver.add_FALP_constr(new_cols_in_cur_feature_matrix,None,cur_reward,t,warm_start=None)            

                    if num_times_basis_added >= 1:
                        SG_RHS = cur_feature_matrix[t][:,0:num_times_basis_added*self.basis_func[0].batch_size]@self.basis_func_coef_matrix[:,t]
                        self.ALP_solver.incorporate_self_guiding_constraint(cur_feature_matrix[t],SG_RHS,t)
                        
                else:
                    if num_times_basis_added == 0:
                        self.ALP_solver.set_up_variables(self.basis_func[0].batch_size, t)
                    else:
                        self.ALP_solver.add_new_variables(self.basis_func[0].batch_size,t)
                
                    cur_feature_matrix[t]                                   = new_cols_in_cur_feature_matrix if cur_feature_matrix[t] is None \
                                                                                    else np.concatenate((cur_feature_matrix[t],new_cols_in_cur_feature_matrix),axis=1)

                    discounted_expected_next_state_feature_matrix_new_basis = self.basis_func[t+1].load_expected_basis(num_times_basis_added,expct_VFA, t+1)
                    
                    discounted_expected_next_state_feature_matrix[t]        = discounted_expected_next_state_feature_matrix_new_basis \
                                                                                    if discounted_expected_next_state_feature_matrix[t] is None\
                                                                                        else np.concatenate((discounted_expected_next_state_feature_matrix[t],discounted_expected_next_state_feature_matrix_new_basis),axis=1)

                    self.ALP_solver.add_FALP_constr(new_cols_in_cur_feature_matrix,discounted_expected_next_state_feature_matrix_new_basis,cur_reward,t,warm_start=None)
                    
                    if num_times_basis_added >= 1:
                        SG_RHS = cur_feature_matrix[t][:,0:num_times_basis_added*self.basis_func[0].batch_size]@self.basis_func_coef_matrix[:,t]
                        self.ALP_solver.incorporate_self_guiding_constraint(cur_feature_matrix[t],SG_RHS,t)#,discounted_expected_next_state_feature_matrix[t] , continue_function[:,t])
                           
                    
                    print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9.2f} |'.format(
                            path_gen_RT,num_basis,t,time.time()-start,'','','','','','','','','',(time.time()-tot_time)/60),end='\r')   
                  
                gc.collect()
                del state_list
                del cur_reward
                if  t <= self.num_stages-2:
                    del new_cols_in_cur_feature_matrix
                
            #------------------------------------------------------------------
            # Set objective function   
            FALP_obj = np.mean(self.basis_func[0].eval_basis(np.array(self.VFA_sample_paths[:,0,:]), num_times_basis_added,all_bases=True), axis=0)  
 
            if num_basis == self.basis_func[0].batch_size:
                self.ALP_solver.set_objective(FALP_obj,False,None)
            else:
                self.ALP_solver.set_objective(FALP_obj,False,self.basis_func_coef_matrix)
            
            #------------------------------------------------------------------
            # [3] Solve ALP 
            FALP_con_RT                         = time.time()-start
            start                               = time.time()
            self.ALP_solver.prepare()
            self.ALP_solver.optimize(num_times_basis_added)
            self.basis_func_coef_matrix         = self.ALP_solver.get_optimal_solution()
            FALP_opt_obj_val                    = self.ALP_solver.get_optimal_value()
            FALP_slv_RT                         = time.time()-start
            
            print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9.2f} |'.format(
                        path_gen_RT,num_basis,t,FALP_con_RT,FALP_slv_RT,FALP_opt_obj_val,'','','','','','','',(time.time()-tot_time)/60),end='\r')
               
            #------------------------------------------------------------------
            # [4] Policy Performance on Train Set
            #------------------------------------------------------------------
            continue_function                   = [None for _  in range(self.num_stages)] 
            for t in range(self.num_stages):  
                if t == self.num_stages-1:
                    continue_function[t]        = np.zeros_like(continue_function[0])
                elif  t == self.num_stages-2:
                    continue_function[t]        = self.VFA_paths_rewards[:,t+1]*self.discount
                else:
                    continue_function[t]        = discounted_expected_next_state_feature_matrix[t] @ self.basis_func_coef_matrix[:,t+1]
                    
            continue_function                   = np.array(continue_function).T
            train_LB_stat,pol_visited_state     = self.get_policy_from_continue_func(continue_function, self.VFA_sample_paths, self.VFA_paths_rewards)
            print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9s} | {:>9s} | {:>9s} | {:>9s}| {:>9s} | {:>9s} | {:>9.2f} |'.format(
                        path_gen_RT,num_basis,t,FALP_con_RT,FALP_slv_RT,FALP_opt_obj_val,train_LB_stat[0],'','','','','','',(time.time()-tot_time)/60),end='\r') 
            
            #------------------------------------------------------------------
            # [5]  Policy Performance on Test Set via CFA
            #------------------------------------------------------------------            
            start
            test_continue_function              = [None for _  in range(self.num_stages)] 
            file_name                           = path+ '/discounted_expected_VFA_test_batch_' +str(self.basis_func[0].preprocess_batch) 
            expct_VFA                           = np.load(file_name+'.npz',allow_pickle=True)
            expct_VFA                           = expct_VFA.f.arr_0
            expct_VFA                           = expct_VFA.item()
            
            for t in range(self.num_stages):
                if t == self.num_stages-1:
                    test_continue_function[t]   = np.zeros(self.num_pol_eval_sample_path)
                elif  t == self.num_stages-2:
                    test_continue_function[t]   = self.pol_sim_paths_rewards[:,t+1]*self.discount
                else:
                    test_discounted_expected_next_state_feature_matrix = self.basis_func[t+1].load_expected_basis(num_times_basis_added, expct_VFA,t+1,is_train=False)
                    test_continue_function[t]   = test_discounted_expected_next_state_feature_matrix @ self.basis_func_coef_matrix[:,t+1]

           
            test_continue_function                   = np.array(test_continue_function).T

            test_LB_stat,_      = self.get_policy_from_continue_func(test_continue_function, self.pol_sim_sample_paths, self.pol_sim_paths_rewards)
            test_LB_RT          = time.time()-start
            
            #------------------------------------------------------------------
            # [6] Show results and store them on disk
            #------------------------------------------------------------------ 
            print('| {:>9.2f} | {:>9d} | {:>9d} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9.2f} | {:>9s} | {:>9s} | {:>9s} | {:>9s} | {:>9.2f} |'.format(
                path_gen_RT,num_basis,t,FALP_con_RT,FALP_slv_RT,FALP_opt_obj_val,train_LB_stat[0],test_LB_stat[0],test_LB_RT,'','','','',(time.time()-tot_time)/60),end='\n')  
            
            self.output_handler.append_to_outputs(algorithm_name            = 'FGLP',
                                                  basis_seed                = self.basis_func[0].basis_func_random_state,
                                                  num_basis_func            = num_basis,
                                                  num_constr                = self.num_VFA_sample_path,
                                                  FALP_obj                  = FALP_opt_obj_val,
                                                  ALP_con_runtime           = FALP_con_RT/60,
                                                  ALP_slv_runtime           = FALP_slv_RT/60,
                                                  train_LB_mean             = train_LB_stat[0],
                                                  train_LB_SE               = train_LB_stat[3],
                                                  test_LB_mean              = test_LB_stat[0],
                                                  test_LB_SE                = test_LB_stat[3], 
                                                  test_LB_runtime           = test_LB_RT/60,
                                                  total_runtime             = (time.time()-tot_time)/60
                                                )
                                                        

            self.ALP_solver.re_initialize_solver()

        print('-'*self.print_len)
        



            