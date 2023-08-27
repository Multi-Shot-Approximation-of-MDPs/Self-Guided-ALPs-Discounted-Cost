# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""

from tqdm import tqdm
import numpy as np
from Bound.informationRelaxation import InformationRelaxation
from utils import output_handler_option_pricing
import time
from utils import mean_confidence_interval
from utils import make_text_bold
from Wrapper.gurobiWrapperBerOpt import gurobi_LP_wrapper
import gc



class SelfGuidedALPs:
    
    
    
    def __init__(self,instance_conf):
        
        self.mdp                                = instance_conf['mdp_conf']['mdp'](instance_conf)
        self.num_stages                         = self.mdp.num_stages
        self.basis_func                         = [instance_conf['basis_func_conf']['basis_func'](instance_conf) for _ in range(self.num_stages+1)]
        
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

        
        self.output_handler                     = output_handler_option_pricing(instance_conf)
        
        self.ALP_solver                         = gurobi_LP_wrapper(instance_conf['solver_conf'])
        
        
        self.print_len                          = 128
        

        self.IR                                 = InformationRelaxation(instance_conf,self.mdp)
        self.IR_inner_sample_seed               = instance_conf['IR_conf']['IR_inner_sample_seed']
        
    
    
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
        print('Basis function type:         \t'     + make_text_bold(self.basis_func[0].basis_func_type))
        print('State relevance:             \t'     + make_text_bold(self.mdp.state_relevance_type))
        print('Underlying random seed:      \t'     + make_text_bold(str(self.basis_func[0].basis_func_random_state)))
        print('='*self.print_len)
        print('| {:>8s} | {:>8s} | {:>20s} | {:>20s} | {:>20s} | {:>20s} | {:>10s} |'.format(
              '# Basis','Stage', 'Train LB', 'Test LB', 'UB (No Penalty)', 'UB (CFA Penalty)', 'Opt Gap') )
        
        print('-'*self.print_len)
            
    
    
    def generate_sample_paths(self):
        self.VFA_sample_paths           = self.mdp.get_sample_path(self.num_VFA_sample_path,self.VFA_random_seed,self.mdp.state_relevance_type)
        self.VFA_paths_rewards          = self.mdp.get_reward_of_path(self.VFA_sample_paths)
        self.pol_sim_sample_paths       = self.mdp.get_sample_path(self.num_pol_eval_sample_path,self.pol_random_seed)
        self.pol_sim_paths_rewards      = self.mdp.get_reward_of_path(self.pol_sim_sample_paths)
        
    
   
    def get_policy_from_continue_func(self,continue_func, paths_state, paths_rewards):
        reward              = []
        eliminated_paths    = []
        stopping_time       = np.zeros(len(paths_state))
        pol_visited_state   = [[] for _ in range(self.num_stages+1)]
            
        for t in range(self.num_stages+1):
            
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
        reward.extend([paths_rewards[_,T]*(self.discount**(T)) for _ in last_stage_stop])   
        
        return mean_confidence_interval(reward),pol_visited_state

        

    def FALP_fixed_basis(self):
        
        """
            Step 1: Generate sample paths and fix inner samples for training VFA
        """
        
        tot_runtime = time.time()
        path_gen_RT = time.time()
        self.print_algorithm_instance_info('FALP')
        self.generate_sample_paths()
        self.mdp.fix_inner_samples(self.mdp.inner_sample_size,self.num_VFA_sample_path,self.VFA_random_seed)
        path_gen_RT = time.time() - path_gen_RT
        
        
        """
            Step 2: Construct FALP and solve it to train VFA
        """
        
        CVFA_fitting_runtime                            = time.time()
        cur_feature_matrix                              = [None for _  in range(self.num_stages)] 
        discounted_expected_next_state_feature_matrix   = [None for _  in range(self.num_stages)] 
        num_basis                                       = self.basis_func[0].batch_size
        
        for t in range(self.num_stages,-1,-1):
            print('| {:>8d} | {:>8d} |'.format(num_basis, t),end='\r')

            state_list              = np.array(self.VFA_sample_paths[:,t,:])
            cur_reward              = self.VFA_paths_rewards[:,t]
            new_cols_in_cur_feature_matrix = self.basis_func[t].eval_basis(state_list)


            if t == self.num_stages:
                self.ALP_solver.set_terminal_val(cur_reward*self.discount)
            
            else:
                self.ALP_solver.set_up_variables(self.basis_func[0].batch_size,t)
    
                cur_feature_matrix[t]           = new_cols_in_cur_feature_matrix 
                
                if t == self.num_stages-1:
                    discounted_expected_next_state_feature_matrix_new_basis = None
            
                else:    
                    inner_next_state_samples                                = self.mdp.get_inner_samples(state_list) 
                    discounted_expected_next_state_feature_matrix_new_basis = self.basis_func[t+1].compute_expected_basis_func(
                                                                        inner_next_state_samples,
                                                                        self.num_VFA_sample_path,
                                                                        self.mdp.inner_sample_size)*self.discount

                    discounted_expected_next_state_feature_matrix[t]        = discounted_expected_next_state_feature_matrix_new_basis
                
                self.ALP_solver.add_FALP_constr(new_cols_in_cur_feature_matrix,
                                                discounted_expected_next_state_feature_matrix_new_basis,
                                                cur_reward,t,warm_start=None)
                    
            gc.collect()
            del state_list
            del cur_reward
            if  t <= self.num_stages-1:
                del new_cols_in_cur_feature_matrix                

        # Set objective function   
        FALP_obj = np.mean(self.basis_func[0].eval_basis(np.array(self.VFA_sample_paths[:,0,:])), axis=0)  
        self.ALP_solver.set_objective(FALP_obj,False,None)

        
        # Solve FALP
        self.ALP_solver.prepare()
        self.ALP_solver.optimize()
        self.basis_func_coef_matrix         = self.ALP_solver.get_optimal_solution()

        CVFA_fitting_runtime                = time.time() - CVFA_fitting_runtime
        
        """
            Step 3: Compute lower bound on optimal reward via policy simulation
        """
        
        # Train set
        lower_bound_runtime                 = time.time()
        continue_function                   = [None for _  in range(self.num_stages+1)] 
        for t in range(self.num_stages+1):  
            if t == self.num_stages:
                continue_function[t]        = np.zeros_like(continue_function[0])
            elif  t == self.num_stages-1:
                continue_function[t]        = self.VFA_paths_rewards[:,t+1]*self.discount
            else:
                continue_function[t]        = discounted_expected_next_state_feature_matrix[t] @ self.basis_func_coef_matrix[:,t+1]
                
        continue_function                   = np.array(continue_function).T
        train_LB_stat,pol_visited_state     = self.get_policy_from_continue_func(continue_function, self.VFA_sample_paths, self.VFA_paths_rewards)
            
        print('| {:>8d} | {:>8d} | {:>20.4f} |'.format(num_basis, t, train_LB_stat[0]),end='\r')
        
        
        # Switch to test set
        self.mdp.fix_inner_samples(self.mdp.inner_sample_size,self.num_pol_eval_sample_path,self.pol_random_seed)
        
        test_continue_function = [None for _  in range(self.num_stages+1)] 
        for t in range(self.num_stages+1):  
            if t == self.num_stages:
                test_continue_function[t]   = np.zeros(self.num_pol_eval_sample_path)
            elif  t == self.num_stages-1:
                test_continue_function[t]   = self.pol_sim_paths_rewards[:,t+1]*self.discount
            else:
                
                state_list                  = np.array(self.pol_sim_sample_paths[:,t,:])
                inner_next_state_samples    = self.mdp.get_inner_samples(state_list) 

                discounted_expected_next_state_feature_matrix_new_basis = self.basis_func[t+1].compute_expected_basis_func(
                                                                                                inner_next_state_samples,
                                                                                                self.num_pol_eval_sample_path,
                                                                                                self.mdp.inner_sample_size)*self.discount

                test_continue_function[t]   = discounted_expected_next_state_feature_matrix_new_basis @ self.basis_func_coef_matrix[:,t+1]

        test_continue_function              = np.array(test_continue_function).T
        test_LB_stat,_                      = self.get_policy_from_continue_func(test_continue_function, self.pol_sim_sample_paths, self.pol_sim_paths_rewards)
        

        print('| {:>8d} | {:>8d} | {:>20.4f} | {:>20.4f} |'.format(num_basis, t, train_LB_stat[0], test_LB_stat[0]),end='\r')
        lower_bound_runtime                 = time.time() - lower_bound_runtime

        """
            Step 4: Compute upper bound on optimal reward via information relaxation & duality
        """
        
        # *** Note that to compute IR bound, we use the same outer samples as test set,
        # but we use different inner samples based on IR_inner_sample_seed
        
        upp_bound_runtime  = time.time()     
        
        self.IR.set_sample_path(  num_sample_path               = self.num_pol_eval_sample_path,
                                  sample_path                   = self.pol_sim_sample_paths,
                                  sample_path_reward            = self.pol_sim_paths_rewards)
        
        self.IR.set_basis_func_coef(num_basis_func              = num_basis,
                                    basis_func_coef_matrix      = self.basis_func_coef_matrix,
                                    basis_func                  = self.basis_func,
                                    num_times_basis_added       = 0)
        
        dual_bound_no_penalty_stat = self.IR.get_dual_bound_no_penalty()

        print('| {:>8d} | {:>8d} | {:>20.4f} | {:>20.4f} | {:>20.4f} |'.format(num_basis, t, train_LB_stat[0],
                                                    test_LB_stat[0],dual_bound_no_penalty_stat[0]),end='\n') 
        
        
        
        # Switch the inner samples based on IR_inner_sample_seed
        self.mdp.fix_inner_samples(self.mdp.inner_sample_size,
                                   self.num_pol_eval_sample_path,
                                   self.IR_inner_sample_seed)
        
        
        # Compute continuation function based on FALP VFA and inner samples with seed IR_inner_sample_seed
        IR_continue_function                = [None for _  in range(self.num_stages+1)]         
        for t in range(self.num_stages+1):  
            if t == self.num_stages:
                IR_continue_function[t]     = np.zeros(self.num_pol_eval_sample_path)
            elif  t == self.num_stages-1:
                IR_continue_function[t]     = self.pol_sim_paths_rewards[:,t+1]*self.discount
            else:
                
                state_list                  = np.array(self.pol_sim_sample_paths[:,t,:])
                inner_next_state_samples    = self.mdp.get_inner_samples(state_list) 

                discounted_expected_next_state_feature_matrix_new_basis = self.basis_func[t+1].compute_expected_basis_func(
                                                                                                inner_next_state_samples,
                                                                                                self.num_pol_eval_sample_path,
                                                                                                self.mdp.inner_sample_size)*self.discount

                IR_continue_function[t]     = discounted_expected_next_state_feature_matrix_new_basis @ self.basis_func_coef_matrix[:,t+1]
        
        IR_continue_function          = np.array(IR_continue_function).T
        dual_bound_with_penalty_stat  = self.IR.get_dual_bound_from_VFA(IR_continue_function)
        opt_gap                       = 100*(dual_bound_with_penalty_stat[0] - test_LB_stat[0])/dual_bound_with_penalty_stat[0]
        upp_bound_runtime             = time.time() - upp_bound_runtime
        
        print('| {:>8d} | {:>8d} | {:>20.4f} | {:>20.4f} | {:>20.4f} | {:>20.4f} | {:>10.2f} |'.format(num_basis, t, train_LB_stat[0],
                                                                test_LB_stat[0],dual_bound_no_penalty_stat[0],
                                                                dual_bound_with_penalty_stat[0],opt_gap),end='\n')
        
        self.output_handler.append_to_outputs(
                          algorithm_name                        = 'FALP',
                          state_relevance_type                  = self.mdp.state_relevance_type,
                          basis_func_type                       = self.basis_func[0].basis_func_type,
                          basis_seed                            = self.basis_func[0].basis_func_random_state,
                          basis_bandwidth_str                   = ''.join(str(x) for x in self.basis_func[0].bandwidth),
                          abs_val_upp_bound                     = str(float('inf')),
                          max_basis_num                         = self.basis_func[0].max_basis_num,
                          num_basis_func                        = num_basis,
                          num_train_samples                     = self.num_VFA_sample_path,
                          num_test_samples                      = self.num_pol_eval_sample_path,
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
        
        

    def compute_expected_basis_func(self,seed_):
        """
        This function computes expected value of basis functions on training,
        testing, and IR sample paths. It also computes a preconditioner for
        basis functions based on PCA.
        """
        
        path                        = 'Output/'+self.mdp.mdp_name+'/instance_'+self.mdp.instance_number+'/seed_'+str(seed_)+'/'
        train_expected_basis_func   = {}
        test_expected_basis_func    = {}
        IR_expected_basis_func      = {}
        
        # Step 1. Form preconditioner
        self.print_algorithm_instance_info('Computing Expected VFAs')
        self.generate_sample_paths()
        
        
        if self.basis_func[0].basis_func_type == 'relu':
            
            for t in range(self.num_stages+1):
                normalizing_const = max(np.max(self.VFA_sample_paths[:,t,:]),np.max(self.VFA_sample_paths[:,t,:]))
                
                self.basis_func[t].set_normalizing_const(normalizing_const)
                
            
        self.mdp.fix_inner_samples(self.mdp.inner_sample_size,self.num_VFA_sample_path,self.VFA_random_seed)

        for t in tqdm(range(self.num_stages+1),ncols=self.print_len,leave=True,desc='Preconditioner'):
            state_list = np.array(self.VFA_sample_paths[:,t,:])       
            self.basis_func[t].form_orthogonal_bases(state_list, path+'/basis_params/',t,False)  
        
        
        # Step 2. Expected value of basis functions on train set
        for t in tqdm(range(self.num_stages),ncols=self.print_len,leave=True,desc='Train'):
            state_list                      = np.array(self.VFA_sample_paths[:,t,:])    
            inner_next_state_samples        = self.mdp.get_inner_samples(state_list) 
            evals                           = self.basis_func[t+1].compute_expected_basis_func(
                                                                inner_next_state_samples,
                                                                self.num_VFA_sample_path,
                                                                self.mdp.inner_sample_size,
                                                                self.discount,
                                                                t+1,
                                                                path +'/train/',
                                                                True)
            """
            *** evals is discounted
            """
            
            train_expected_basis_func.update({'expected_basis_func_'+str(t+1) : evals})
        
        np.savez_compressed(path+'discounted_expected_VFA_train_batch_'+str(self.basis_func[0].preprocess_batch) ,train_expected_basis_func)
        del train_expected_basis_func
    
        
        # Step 3. Expected value of basis functions on test set        
        self.mdp.fix_inner_samples(self.mdp.inner_sample_size,self.num_pol_eval_sample_path,self.pol_random_seed)
        for t in tqdm(range(self.num_stages),ncols=self.print_len,leave=True,desc='Test '):
            state_list                      = np.array(self.pol_sim_sample_paths[:,t,:])
            inner_next_state_samples        = self.mdp.get_inner_samples(state_list) 
            evals                           = self.basis_func[t+1].compute_expected_basis_func(
                                                              inner_next_state_samples,
                                                              self.num_pol_eval_sample_path,
                                                              self.mdp.inner_sample_size,
                                                              self.discount,
                                                              t+1,
                                                              path+'/test/',
                                                              False)
            
            test_expected_basis_func.update({'expected_basis_func_'+str(t+1) : evals})

        np.savez_compressed(path+'discounted_expected_VFA_test_batch_'+str(self.basis_func[0].preprocess_batch) ,test_expected_basis_func)
        del test_expected_basis_func

     

        # Step 4. Expected value of basis functions on test set with inner samples based on IR seed        
        self.mdp.fix_inner_samples(self.mdp.inner_sample_size,self.num_pol_eval_sample_path,self.IR_inner_sample_seed)
        for t in tqdm(range(self.num_stages),ncols=self.print_len,leave=True,desc='IR   '):
            state_list                      = np.array(self.pol_sim_sample_paths[:,t,:])
            inner_next_state_samples        = self.mdp.get_inner_samples(state_list) 
            evals                           = self.basis_func[t+1].compute_expected_basis_func(
                                                              inner_next_state_samples,
                                                              self.num_pol_eval_sample_path,
                                                              self.mdp.inner_sample_size,
                                                              self.discount,
                                                              t+1,
                                                              path+'/IR/',
                                                              False)
            
            IR_expected_basis_func.update({'expected_basis_func_'+str(t+1) : evals})                
                

        np.savez_compressed(path+'discounted_expected_VFA_IR_batch_'+str(self.basis_func[0].preprocess_batch) ,IR_expected_basis_func)
        del IR_expected_basis_func

        print('-'*self.print_len)
        
        
        

    def FALP_random_bases(self): 

        assert self.basis_func[0].batch_size == self.basis_func[0].max_basis_num, '\nThe FALP code is not tested when ALP is solved iteratively. Ensure that batch_size = max_basis_num.\n'
        
        """
            Step 1: Generate sample paths and fix inner samples for training VFA
        """
        tot_runtime                                     = time.time()
        path_gen_RT                                     = time.time()
        self.print_algorithm_instance_info('FALP')
        self.generate_sample_paths()
        path_gen_RT                                     = time.time() - path_gen_RT
        

        """
            Step 2: Construct FALP and solve it to train VFA
        """      
        CVFA_fitting_runtime                            = time.time()
        basis_range                                     = np.arange(self.basis_func[0].batch_size, self.basis_func[0].max_basis_num+1,self.basis_func[0].batch_size)
        cur_feature_matrix                              = [None for _  in range(self.num_stages)] 
        discounted_expected_next_state_feature_matrix   = [None for _  in range(self.num_stages)] 


        num_times_basis_added               = -1
        for num_basis in basis_range:
            path            = 'Output/'+self.mdp.mdp_name+'/instance_'+self.mdp.instance_number+'/seed_'+str(int(self.basis_func[0].basis_func_random_state))
            file_name       = path+ '/discounted_expected_VFA_train_batch_' +str(self.basis_func[0].preprocess_batch) 
            expct_VFA       = np.load(file_name+'.npz',allow_pickle=True)
            expct_VFA       = expct_VFA.f.arr_0
            expct_VFA       = expct_VFA.item()
        
            for num_basis in basis_range:
                num_times_basis_added               += 1 

                for t in range(self.num_stages,-1,-1):
                    print('| {:>8d} | {:>8d} |'.format(num_basis, t),end='\r')
                    
                    state_list              = np.array(self.VFA_sample_paths[:,t,:])
                    cur_reward              = self.VFA_paths_rewards[:,t]
                    path                    = 'Output/'+self.mdp.mdp_name+'/instance_'+self.mdp.instance_number+'/seed_'+str(int(self.basis_func[0].basis_func_random_state))
                                               
                    # Load bases
                    if num_times_basis_added == 0:
                        self.basis_func[t].form_orthogonal_bases(state_list,path+'/basis_params/',stage=t,to_load=True)   
                
                    new_cols_in_cur_feature_matrix = self.basis_func[t].eval_basis(state_list,num_times_basis_added,False)
    
                   
                    if t == self.num_stages:
                        self.ALP_solver.set_terminal_val(cur_reward*self.discount)
                        
                    else:
                        # Set up variables of ALP
                        if num_times_basis_added == 0:
                            self.ALP_solver.set_up_variables(self.basis_func[0].batch_size,t)
                        else:
                            self.ALP_solver.add_new_variables(self.basis_func[0].batch_size,t)  
                        
                        if t == self.num_stages-1:
    
                            cur_feature_matrix[t]                                   = new_cols_in_cur_feature_matrix if cur_feature_matrix[t] is None \
                                                                                            else np.concatenate((cur_feature_matrix[t],new_cols_in_cur_feature_matrix),axis=1)
                             
                            self.ALP_solver.add_FALP_constr(new_cols_in_cur_feature_matrix,None,cur_reward,t,warm_start=None)            
                        
                        else:
                            cur_feature_matrix[t]                                   = new_cols_in_cur_feature_matrix if cur_feature_matrix[t] is None \
                                                                                            else np.concatenate((cur_feature_matrix[t],new_cols_in_cur_feature_matrix),axis=1)
        
                            file_name                                               = path+ '/discounted_expected_VFA_train_batch_' +str(self.basis_func[t+1].preprocess_batch) 
                            
                            discounted_expected_next_state_feature_matrix_new_basis = self.basis_func[t+1].load_expected_basis(num_times_basis_added,expct_VFA, t+1)
                            
                            discounted_expected_next_state_feature_matrix[t]        = discounted_expected_next_state_feature_matrix_new_basis \
                                                                                            if discounted_expected_next_state_feature_matrix[t] is None\
                                                                                                else np.concatenate((discounted_expected_next_state_feature_matrix[t],discounted_expected_next_state_feature_matrix_new_basis),axis=1)
        
                            self.ALP_solver.add_FALP_constr(new_cols_in_cur_feature_matrix,discounted_expected_next_state_feature_matrix_new_basis,cur_reward,t,warm_start=None)
                                                  
                    gc.collect()
                    del state_list
                    del cur_reward
                    if  t <= self.num_stages-2:
                        del new_cols_in_cur_feature_matrix
             
            print('| {:>8d} | {:>8d} |'.format(num_basis, t),end='\r')
            
            # Set objective function 
            FALP_obj = np.mean(self.basis_func[0].eval_basis(np.array(self.VFA_sample_paths[:,0,:]), num_times_basis_added,all_bases=True), axis=0)  
 
            if num_basis == self.basis_func[0].batch_size:
                self.ALP_solver.set_objective(FALP_obj,False,None)
            else:
                self.ALP_solver.set_objective(FALP_obj,False,self.basis_func_coef_matrix)
            
            
            # Solve FALP
            self.ALP_solver.prepare()
            self.ALP_solver.optimize(num_times_basis_added)
            self.basis_func_coef_matrix         = self.ALP_solver.get_optimal_solution()            
            CVFA_fitting_runtime                = time.time() - CVFA_fitting_runtime
            
            
            """
                Step 3: Compute lower bound on optimal reward via policy simulation
            """
            # Train set
            lower_bound_runtime                 = time.time()
            continue_function                   = [None for _  in range(self.num_stages+1)] 
            for t in range(self.num_stages+1):  
                if t == self.num_stages:
                    continue_function[t]        = np.zeros_like(continue_function[0])
                elif  t == self.num_stages-1:
                    continue_function[t]        = self.VFA_paths_rewards[:,t+1]*self.discount
                else:
                    continue_function[t]        = discounted_expected_next_state_feature_matrix[t] @ self.basis_func_coef_matrix[:,t+1]
                    
            continue_function                   = np.array(continue_function).T
            train_LB_stat,pol_visited_state     = self.get_policy_from_continue_func(continue_function, self.VFA_sample_paths, self.VFA_paths_rewards)
            
            # Switch to test set
            test_continue_function              = [None for _  in range(self.num_stages+1)] 
            file_name                           = path+ '/discounted_expected_VFA_test_batch_' +str(self.basis_func[0].preprocess_batch) 
            expct_VFA                           = np.load(file_name+'.npz',allow_pickle=True)
            expct_VFA                           = expct_VFA.f.arr_0
            expct_VFA                           = expct_VFA.item()
            
            for t in range(self.num_stages+1):
                if t == self.num_stages:
                    test_continue_function[t]   = np.zeros(self.num_pol_eval_sample_path)
                elif  t == self.num_stages-1:
                    test_continue_function[t]   = self.pol_sim_paths_rewards[:,t+1]*self.discount
                else:
                    test_discounted_expected_next_state_feature_matrix = self.basis_func[t+1].load_expected_basis(num_times_basis_added, expct_VFA,t+1,is_train=False)
                    
                    test_continue_function[t]   = test_discounted_expected_next_state_feature_matrix @ self.basis_func_coef_matrix[:,t+1]

            test_continue_function              = np.array(test_continue_function).T
            test_LB_stat,_                      = self.get_policy_from_continue_func(test_continue_function, self.pol_sim_sample_paths, self.pol_sim_paths_rewards)
            lower_bound_runtime                 = time.time() - lower_bound_runtime

            print('| {:>8d} | {:>8d} | {:>20.4f} | {:>20.4f} |'.format(num_basis, t, train_LB_stat[0], test_LB_stat[0]),end='\r')
            

            """
                Step 4: Compute upper bound on optimal reward via information relaxation & duality
            """
            
            # *** Note that to compute IR bound, we use the same outer samples as test set,
            # but we use different inner samples based on IR_inner_sample_seed
        
            upp_bound_runtime  = time.time()   
            
            self.IR.set_sample_path(  num_sample_path               = self.num_pol_eval_sample_path,
                                      sample_path                   = self.pol_sim_sample_paths,
                                      sample_path_reward            = self.pol_sim_paths_rewards)
            
            self.IR.set_basis_func_coef(num_basis_func              = num_basis,
                                        basis_func_coef_matrix      = self.basis_func_coef_matrix,
                                        basis_func                  = self.basis_func,
                                        num_times_basis_added       = num_times_basis_added)
            
            dual_bound_no_penalty_stat    = self.IR.get_dual_bound_no_penalty()

            print('| {:>8d} | {:>8d} | {:>20.4f} | {:>20.4f} | {:>20.4f} |'.format(num_basis, t, train_LB_stat[0],
                                                        test_LB_stat[0], dual_bound_no_penalty_stat[0]),end='\n') 
            
            

            # Switch the inner samples based on IR_inner_sample_seed
            self.mdp.fix_inner_samples(self.mdp.inner_sample_size,
                                       self.num_pol_eval_sample_path,
                                       self.IR_inner_sample_seed)
            
            
            # Compute continuation function based on FALP VFA and inner samples with seed IR_inner_sample_seed
            IR_continue_function                = [None for _  in range(self.num_stages+1)] 
            file_name                           = path+ '/discounted_expected_VFA_IR_batch_' +str(self.basis_func[0].preprocess_batch) 
            expct_VFA                           = np.load(file_name+'.npz',allow_pickle=True)
            expct_VFA                           = expct_VFA.f.arr_0
            expct_VFA                           = expct_VFA.item()
            
            for t in range(self.num_stages+1):
                if t == self.num_stages:
                    IR_continue_function[t]   = np.zeros(self.num_pol_eval_sample_path)
                elif  t == self.num_stages-1:
                    IR_continue_function[t]   = self.pol_sim_paths_rewards[:,t+1]*self.discount
                else:
                    discounted_expected_next_state_feature_matrix = self.basis_func[t+1].load_expected_basis(num_times_basis_added, expct_VFA,t+1,is_train=False)
                    
                    IR_continue_function[t]   = discounted_expected_next_state_feature_matrix @ self.basis_func_coef_matrix[:,t+1]


            IR_continue_function          = np.array(IR_continue_function).T
            dual_bound_with_penalty_stat  = self.IR.get_dual_bound_from_VFA(IR_continue_function)
            opt_gap                       = 100*(dual_bound_with_penalty_stat[0] - test_LB_stat[0])/dual_bound_with_penalty_stat[0]
            upp_bound_runtime             = time.time() - upp_bound_runtime
            
            print('| {:>8d} | {:>8d} | {:>20.4f} | {:>20.4f} | {:>20.4f} | {:>20.4f} | {:>10.2f} |'.format(num_basis, t, train_LB_stat[0],
                                                                    test_LB_stat[0], dual_bound_no_penalty_stat[0],
                                                                    dual_bound_with_penalty_stat[0], opt_gap),end='\n')
            

        self.output_handler.append_to_outputs(
                          algorithm_name                        = 'FALP',
                          state_relevance_type                  = self.mdp.state_relevance_type,
                          basis_func_type                       = self.basis_func[0].basis_func_type,
                          basis_seed                            = self.basis_func[0].basis_func_random_state,
                          basis_bandwidth_str                   = ''.join(str(x) for x in self.basis_func[0].bandwidth),
                          abs_val_upp_bound                     = str(float('inf')),
                          max_basis_num                         = self.basis_func[0].max_basis_num,
                          num_basis_func                        = num_basis,
                          num_train_samples                     = self.num_VFA_sample_path,
                          num_test_samples                      = self.num_pol_eval_sample_path,
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
        
        
        
     
    def SG_FALP(self):
        
        """
            Step 1: Generate sample paths and fix inner samples for training VFA
        """
        tot_runtime                                     = time.time()
        start                                           = time.time()
        self.print_algorithm_instance_info('Self-guided FALP')
        self.generate_sample_paths()
        path_gen_RT                                     = (time.time() - start)
        
        """
            Step 2: Construct FALP and solve it to train VFA
        """      
        CVFA_fitting_runtime                            = 0.0
        lower_bound_runtime                             = 0.0
        upp_bound_runtime                               = 0.0
        best_upper_bound                                = float('inf')
        basis_range                                     = np.arange(self.basis_func[0].batch_size, self.basis_func[0].max_basis_num+1,self.basis_func[0].batch_size)
        cur_feature_matrix                              = [None for _  in range(self.num_stages)] 
        discounted_expected_next_state_feature_matrix   = [None for _  in range(self.num_stages)] 

        num_times_basis_added   = -1
        best_upp_bound          = float('inf')                  
        best_ALP_obj            = float('inf') 
        path                    = 'Output/'+self.mdp.mdp_name+'/instance_'+self.mdp.instance_number+'/seed_'+str(int(self.basis_func[0].basis_func_random_state))
        file_name               = path + '/discounted_expected_VFA_train_batch_' +str(self.basis_func[0].preprocess_batch) 
        
        
        for num_basis in basis_range:
            
            start           = time.time()
            
            # Ensure loading the VFA evals on train set
            path            = 'Output/'+self.mdp.mdp_name+'/instance_'+self.mdp.instance_number+'/seed_'+str(int(self.basis_func[0].basis_func_random_state))
            file_name       = path+ '/discounted_expected_VFA_train_batch_' +str(self.basis_func[0].preprocess_batch) 
            expct_VFA       = np.load(file_name+'.npz',allow_pickle=True)
            expct_VFA       = expct_VFA.f.arr_0
            expct_VFA       = expct_VFA.item()
                
            num_times_basis_added      += 1 
            self.mdp.fix_inner_samples(self.mdp.inner_sample_size,self.num_VFA_sample_path,self.VFA_random_seed)
        
        
        
            for t in range(self.num_stages,-1,-1):
                print('| {:>8d} | {:>8d} |'.format(num_basis, t),end='\r')
                
                state_list              = np.array(self.VFA_sample_paths[:,t,:])
                cur_reward              = self.VFA_paths_rewards[:,t]
                
                # Load bases
                if num_times_basis_added == 0:
                    self.basis_func[t].form_orthogonal_bases(state_list,path+'/basis_params/',stage=t,to_load=True)   
            
                new_cols_in_cur_feature_matrix = self.basis_func[t].eval_basis(state_list,num_times_basis_added,False)
                   
                
                    
                if t == self.num_stages:
                    self.ALP_solver.set_terminal_val(cur_reward*self.discount)
                
                
                else:
                    # Set up variables of ALP
                    if num_times_basis_added == 0:
                        self.ALP_solver.set_up_variables(self.basis_func[0].batch_size,t)
                    else:
                        self.ALP_solver.add_new_variables(self.basis_func[0].batch_size,t)  
                    
                    # Add constraints
                    
                   
                    
                    if t == self.num_stages-1:
                        cur_feature_matrix[t] = new_cols_in_cur_feature_matrix if cur_feature_matrix[t] is None \
                                                    else np.concatenate((cur_feature_matrix[t],new_cols_in_cur_feature_matrix),axis=1)
                         
                        self.ALP_solver.add_FALP_constr(new_cols_in_cur_feature_matrix,None,cur_reward,t,warm_start=None)            
                        
                        if num_times_basis_added >= 1:
                            SG_RHS = cur_feature_matrix[t][:,0:num_times_basis_added*self.basis_func[0].batch_size]@self.basis_func_coef_matrix[:,t]
                            self.ALP_solver.incorporate_self_guiding_constraint(cur_feature_matrix[t],SG_RHS,t)                
                
                
                
                    else:
                        
                        
                        cur_feature_matrix[t]                                   = new_cols_in_cur_feature_matrix if cur_feature_matrix[t] is None \
                                                                                        else np.concatenate((cur_feature_matrix[t],new_cols_in_cur_feature_matrix),axis=1)
    
                        file_name                                               = path+ '/discounted_expected_VFA_train_batch_' +str(self.basis_func[t+1].preprocess_batch) 
                        
                        discounted_expected_next_state_feature_matrix_new_basis = self.basis_func[t+1].load_expected_basis(num_times_basis_added,expct_VFA, t+1)
                        
                        discounted_expected_next_state_feature_matrix[t]        = discounted_expected_next_state_feature_matrix_new_basis \
                                                                                        if discounted_expected_next_state_feature_matrix[t] is None\
                                                                                            else np.concatenate((discounted_expected_next_state_feature_matrix[t],discounted_expected_next_state_feature_matrix_new_basis),axis=1)
    
                        self.ALP_solver.add_FALP_constr(new_cols_in_cur_feature_matrix,discounted_expected_next_state_feature_matrix_new_basis,cur_reward,t,warm_start=None)
                                                  
        
                        if num_times_basis_added >= 1:
                            SG_RHS = cur_feature_matrix[t][:,0:num_times_basis_added*self.basis_func[0].batch_size]@self.basis_func_coef_matrix[:,t]
                            self.ALP_solver.incorporate_self_guiding_constraint(cur_feature_matrix[t],SG_RHS,t)#,discounted_expected_next_state_feature_matrix[t] , continue_function[:,t])
                    
                            

                    
                    gc.collect()
                    del state_list
                    del cur_reward
                    if  t <= self.num_stages-1:
                        del new_cols_in_cur_feature_matrix
                    
                
                
            # Set FGLP objective function   
            FALP_obj = np.mean(self.basis_func[0].eval_basis(np.array(self.VFA_sample_paths[:,0,:]), num_times_basis_added,all_bases=True), axis=0)  

            if num_basis == self.basis_func[0].batch_size:
                self.ALP_solver.set_objective(FALP_obj,False,None)
            else:
                self.ALP_solver.set_objective(FALP_obj,False,self.basis_func_coef_matrix)
        

            # self.ALP_solver.add_box_constr()
            # self.ALP_solver.prepare()
            self.ALP_solver.optimize(num_times_basis_added,)
            self.basis_func_coef_matrix         = self.ALP_solver.get_optimal_solution()

            CVFA_fitting_runtime               += (time.time() - start)
            
            
            """
                Step 3: Compute lower bound on optimal reward via policy simulation
            """
            # Train set
            start                               = time.time()
            continue_function                   = [None for _  in range(self.num_stages+1)] 
            for t in range(self.num_stages+1):  
                if t == self.num_stages:
                    continue_function[t]        = np.zeros_like(continue_function[0])
                elif  t == self.num_stages-1:
                    continue_function[t]        = self.VFA_paths_rewards[:,t+1]*self.discount
                else:
                    continue_function[t]        = discounted_expected_next_state_feature_matrix[t] @ self.basis_func_coef_matrix[:,t+1]
                    
            continue_function                   = np.array(continue_function).T
            train_LB_stat,pol_visited_state     = self.get_policy_from_continue_func(continue_function, self.VFA_sample_paths, self.VFA_paths_rewards)
            
            
            print('| {:>8d} | {:>8d} | {:>20.4f} |'.format(num_basis, t, train_LB_stat[0]),end='\r')
            
            # Switch to test set
            self.mdp.fix_inner_samples(self.mdp.inner_sample_size,self.num_pol_eval_sample_path,self.pol_random_seed)
            test_continue_function              = [None for _  in range(self.num_stages+1)] 
            file_name                           = path+ '/discounted_expected_VFA_test_batch_' +str(self.basis_func[0].preprocess_batch) 
            expct_VFA                           = np.load(file_name+'.npz',allow_pickle=True)
            expct_VFA                           = expct_VFA.f.arr_0
            expct_VFA                           = expct_VFA.item()
            
            for t in range(self.num_stages+1):
                if t == self.num_stages:
                    test_continue_function[t]   = np.zeros(self.num_pol_eval_sample_path)
                elif  t == self.num_stages-1:
                    test_continue_function[t]   = self.pol_sim_paths_rewards[:,t+1]*self.discount
                else:
                    test_discounted_expected_next_state_feature_matrix = self.basis_func[t+1].load_expected_basis(num_times_basis_added, expct_VFA,t+1,is_train=False)
                    
                    test_continue_function[t]   = test_discounted_expected_next_state_feature_matrix @ self.basis_func_coef_matrix[:,t+1]

            test_continue_function              = np.array(test_continue_function).T
            test_LB_stat,_                      = self.get_policy_from_continue_func(test_continue_function, self.pol_sim_sample_paths, self.pol_sim_paths_rewards)
            lower_bound_runtime                += (time.time() - start)

            print('| {:>8d} | {:>8d} | {:>20.4f} | {:>20.4f} |'.format(num_basis, t, train_LB_stat[0], test_LB_stat[0]),end='\r')
            
            
            
            
            """
                Step 4: Compute upper bound on optimal reward via information relaxation & duality
            """
            
            # *** Note that to compute IR bound, we use the same outer samples as test set,
            # but we use different inner samples based on IR_inner_sample_seed
        
            start  = time.time()   
    
            self.IR.set_sample_path(  num_sample_path               = self.num_pol_eval_sample_path,
                                      sample_path                   = self.pol_sim_sample_paths,
                                      sample_path_reward            = self.pol_sim_paths_rewards)
            
            self.IR.set_basis_func_coef(num_basis_func              = num_basis,
                                        basis_func_coef_matrix      = self.basis_func_coef_matrix,
                                        basis_func                  = self.basis_func,
                                        num_times_basis_added       = num_times_basis_added)
            
            dual_bound_no_penalty_stat    = self.IR.get_dual_bound_no_penalty()

            print('| {:>8d} | {:>8d} | {:>20.4f} | {:>20.4f} | {:>20.4f} |'.format(num_basis, t, train_LB_stat[0],
                                                        test_LB_stat[0], dual_bound_no_penalty_stat[0]),end='\r') 
            
            

            # Switch the inner samples based on IR_inner_sample_seed
            self.mdp.fix_inner_samples(self.mdp.inner_sample_size,
                                        self.num_pol_eval_sample_path,
                                        self.IR_inner_sample_seed)
            
            
            # Compute continuation function based on FALP VFA and inner samples with seed IR_inner_sample_seed
            IR_continue_function                = [None for _  in range(self.num_stages+1)] 
            file_name                           = path+ '/discounted_expected_VFA_IR_batch_' +str(self.basis_func[0].preprocess_batch) 
            expct_VFA                           = np.load(file_name+'.npz',allow_pickle=True)
            expct_VFA                           = expct_VFA.f.arr_0
            expct_VFA                           = expct_VFA.item()
            
            for t in range(self.num_stages+1):
                if t == self.num_stages:
                    IR_continue_function[t]   = np.zeros(self.num_pol_eval_sample_path)
                elif  t == self.num_stages-1:
                    IR_continue_function[t]   = self.pol_sim_paths_rewards[:,t+1]*self.discount
                else:
                    IR_discounted_expected_next_state_feature_matrix = self.basis_func[t+1].load_expected_basis(num_times_basis_added, expct_VFA,t+1,is_train=False) 
                    
                    IR_continue_function[t]   = IR_discounted_expected_next_state_feature_matrix @ self.basis_func_coef_matrix[:,t+1]


            IR_continue_function          = np.array(IR_continue_function).T
            dual_bound_with_penalty_stat  = self.IR.get_dual_bound_from_VFA(IR_continue_function)
            
            best_upper_bound              = min(best_upper_bound,dual_bound_with_penalty_stat[0])  
            
            opt_gap                       = 100*(best_upper_bound - test_LB_stat[0])/best_upper_bound
            upp_bound_runtime             = (time.time() - start)
            
            print('| {:>8d} | {:>8d} | {:>20.4f} | {:>20.4f} | {:>20.4f} | {:>20.4f} | {:>10.2f} |'.format(num_basis, t, train_LB_stat[0],
                                                                    test_LB_stat[0], dual_bound_no_penalty_stat[0],
                                                                    dual_bound_with_penalty_stat[0], opt_gap),end='\n')


            self.output_handler.append_to_outputs(
                              algorithm_name                        = 'SG_FALP',
                              state_relevance_type                  = self.mdp.state_relevance_type,
                              basis_func_type                       = self.basis_func[0].basis_func_type,
                              basis_seed                            = self.basis_func[0].basis_func_random_state,
                              basis_bandwidth_str                   = ''.join(str(x) for x in self.basis_func[0].bandwidth),
                              abs_val_upp_bound                     = str(self.ALP_solver.abs_val_upp_bound),
                              max_basis_num                         = self.basis_func[0].max_basis_num,
                              num_basis_func                        = num_basis,
                              num_train_samples                     = self.num_VFA_sample_path,
                              num_test_samples                      = self.num_pol_eval_sample_path,
                              num_inner_samples                     = self.mdp.inner_sample_size,
                              train_LB_mean                         = train_LB_stat[0],
                              train_LB_SE                           = train_LB_stat[3],
                              test_LB_mean                          = test_LB_stat[0],
                              test_LB_SE                            = test_LB_stat[3], 
                              dual_bound_no_penalty_mean            = dual_bound_no_penalty_stat[0],
                              dual_bound_no_penalty_se              = dual_bound_no_penalty_stat[3],
                              dual_bound_with_penalty_mean          = dual_bound_with_penalty_stat[0],
                              dual_bound_with_penalty_se            = dual_bound_with_penalty_stat[3],
                              best_upper_bound                      = best_upper_bound,
                              opt_gap                               = opt_gap,
                              path_gen_runtime                      = path_gen_RT,
                              upp_bound_runtime                     = upp_bound_runtime,
                              lower_bound_runtime                   = lower_bound_runtime,
                              CVFA_fitting_runtime                  = CVFA_fitting_runtime,
                              total_runtime                         = (time.time()-tot_runtime)
                              )   
        
            """
            Step 5:     Reset ALP model for the next iteration
            """
            # self.ALP_solver.remove_box_constr()
            self.ALP_solver.re_initialize_solver()
            
        print('-'*self.print_len)

