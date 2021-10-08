# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from multiprocessing import Pool
from numpy import asarray,concatenate
from Bound.greedyPolicyUpperBound import GreedyPolicy
import time
from Bound.lowerBoundEstimator import LowerBound
from utils import output_handler,make_text_bold
from functools import partial
from math import floor
import numpy as np


def make_single_ALP_constriant(eval_basis,discount,expected_basis,get_batch_next_state,get_expected_cost,new_basis_param,state,action): 
    return eval_basis(state,new_basis_param), \
               discount*expected_basis(get_batch_next_state(state,action),new_basis_param),\
                   get_expected_cost(state,action)  
                              
                              
class Self_Guided_ALP:
    
    def __init__(self,instance_conf):
        #----------------------------------------------------------------------
        # Initialization
        #----------------------------------------------------------------------
        self.mdp                                = instance_conf['mdp_conf']['mdp'](instance_conf)
        self.basis_func                         = instance_conf['basis_func_conf']['basis_func'](instance_conf)
        self.constr_conf                        = instance_conf['constr_conf']
        self.mdp_sampler_conf                   = instance_conf['mdp_sampler_conf']
        self.max_basis_num                      = instance_conf['basis_func_conf']['max_basis_num']
        self.opt_gap_threshold                  = instance_conf['misc_conf']['opt_gap_threshold']
        self.num_cpu_core                       = instance_conf['misc_conf']['num_cpu_core']
        
        self.mdp.sample_fix_batch_mdp_noise()
              
        if hasattr(self.mdp, 'get_batch_samples_state_relevance'):   
             self.ALP_state_rel_sample          = self.mdp.get_batch_samples_state_relevance()  
        else:
            self.ALP_state_rel_sample           = None
            
        self.opt_gap_threshold                  = instance_conf['misc_conf']['opt_gap_threshold']
        self.ALP_solver                         = instance_conf['solver']['solver_name'](instance_conf['solver'])
        self.basis_param                        = None
        self.ALP_constr_matrix                  = None
        self.ALP_RHS                            = None 
        self.RHS_self_guide_constr              = None
        self.ALP_self_guide_constr_matrix       = None 
        self.basis_func_evals                   = None    
        self.greedy_policy                      = GreedyPolicy(self.mdp, instance_conf['greedy_pol_conf'])
        self.lower_bound_estimator              = LowerBound(self.mdp, instance_conf['lower_bound_conf'])
        self.output_handler                     = output_handler(instance_conf)
        
        
        if 'update_state_rel_via_greedy_pol' in instance_conf['greedy_pol_conf']:
            self.update_state_rel_via_greedy_pol    =  instance_conf['greedy_pol_conf']['update_state_rel_via_greedy_pol' ]
            self.state_relevance_inner_itr          =  instance_conf['greedy_pol_conf']['state_relevance_inner_itr' ]

        self.num_basis_to_update_pol_cost           = instance_conf['greedy_pol_conf']['num_basis_to_update_pol_cost' ]
        

    def compute_FALP_components(self,new_basis_param, state_list, act_list):
        #----------------------------------------------------------------------
        # This function computes the left-hand-side and right-hand-side of
        # FALP constraints for a given set of states and actions, and a batch
        # of basis functions.
        #----------------------------------------------------------------------
        discount                = self.mdp.discount
        eval_basis              = self.basis_func.eval_basis
        expected_basis          = self.basis_func.expected_basis
        get_batch_next_state    = self.mdp.get_batch_next_state
        get_expected_cost       = self.mdp.get_expected_cost        
        mapping                 = partial(make_single_ALP_constriant,
                                          eval_basis,discount,
                                          expected_basis,
                                          get_batch_next_state,
                                          get_expected_cost,
                                          new_basis_param)
        pool                    = Pool(self.num_cpu_core)
        X                       = pool.starmap(mapping,zip(state_list,act_list))
        pool.close()
        pool.join()
        num_ALP_constr          = len(act_list)
        constraint_Matrix       = asarray([X[i][0]-X[i][1]  for i in range(num_ALP_constr)])
        RHS_vector              = asarray([X[i][2]  for i in range(num_ALP_constr)])
        discounted_expct_VFA    = [X[i][1]  for i in range(num_ALP_constr)]   
        basis_func_evals        = [X[i][0]  for i in range(num_ALP_constr)]   

        return constraint_Matrix, RHS_vector, discounted_expct_VFA,basis_func_evals
        
    
    def add_FALP_constr(self, state_list, act_list,is_SGFALP=False): 
        #----------------------------------------------------------------------
        # Given the current set of bases, append constraints to FALP. The number
        # of variables is fixed, constraints are added
        #----------------------------------------------------------------------
        new_constraint_matrix, new_RHS_vector, new_expct_VFA, basis_func_evals\
                = self.compute_FALP_components(self.basis_param , state_list, act_list)

        #----------------------------------------------------------------------
        # Either construct ALP components from scratch if it is the first
        # iteration or update based on the existing components of ALP
        if self.ALP_constr_matrix is None:
            self.ALP_constr_matrix                  = new_constraint_matrix
            self.ALP_RHS                            = new_RHS_vector
            self.basis_func_evals                   = basis_func_evals
        else:
            self.ALP_constr_matrix                  = concatenate((self.ALP_constr_matrix,new_constraint_matrix),axis=0)
            self.ALP_RHS                            = concatenate((self.ALP_RHS,new_RHS_vector))
            self.basis_func_evals                   = concatenate((self.basis_func_evals,basis_func_evals)) 

        #----------------------------------------------------------------------
        # Add FALP constraints to linear programming solver (i.e. Gurobi)
        self.ALP_solver.add_FALP_constraint(new_constraint_matrix,new_RHS_vector)
        self.ALP_solver.prepare()


    def add_columns_to_FALP(self, num_basis,new_basis_param,state_list_all,act_list_all,greedy_pol_visited_state):
        #----------------------------------------------------------------------
        # Given the left-hand-side and right-hand-side of FALP_N with N bases, 
        # this function extends them to compute the  the left-hand-side and 
        # right-hand-side of FALP_{N+B} with N+B bases. The number constraints
        # is fixed, variables are added
        #----------------------------------------------------------------------
        self.ALP_solver.re_initialize_solver()
        self.ALP_solver.add_new_variables(self.basis_func.batch_size)
        
        if self.basis_param is None:
            self.basis_param = new_basis_param
        else:
            self.basis_param = self.basis_func.concat(self.basis_param,new_basis_param)
            old_state_new_basis_constraint_matrix, old_state_new_basis_RHS_vector,\
                old_state_new_basis_expct_VFA, old_state_new_basis_func_evals  = self.compute_FALP_components(new_basis_param , state_list_all, act_list_all)
            
            self.ALP_constr_matrix  = concatenate((self.ALP_constr_matrix,old_state_new_basis_constraint_matrix),axis=1)
            self.ALP_solver.add_FALP_constraint(old_state_new_basis_constraint_matrix, self.ALP_RHS)
            self.basis_func_evals =  concatenate((self.basis_func_evals,old_state_new_basis_func_evals),axis=1) 
            
        #----------------------------------------------------------------------
        # If the model is policy-guided FALP, then updated the ALP objective 
        # using states visited by the greedy policy; otherwise, use the 
        # provided state relevance distribution
        if (greedy_pol_visited_state is None) or (self.update_state_rel_via_greedy_pol==False):
            obj_coef = sum([self.basis_func.eval_basis(state,self.basis_param) for state in self.ALP_state_rel_sample])/len(self.ALP_state_rel_sample)
        else:
            obj_coef = sum([self.basis_func.eval_basis(state,self.basis_param) for state in greedy_pol_visited_state])/len(greedy_pol_visited_state)
      
        self.ALP_solver.set_objective(obj_coef)
        self.ALP_solver.prepare()


    def incorporate_self_guiding_constraint(self,VFA_prev_eval,state_list_all):
        #----------------------------------------------------------------------
        # This function uses the current FALP solution and adds guiding 
        # constraints to it to solve self-guided FALP.
        #----------------------------------------------------------------------
        self_guiding_constr_matrix                  = self.basis_func_evals      
        RHS_self_guiding_constr                     = VFA_prev_eval              
        basis_ceof,obj_ALP,num_self_guiding_constr  = None,0,len(RHS_self_guiding_constr) 
        
        if  num_self_guiding_constr >0:
            basis_ceof,obj_ALP = self.ALP_solver.incorporate_self_guiding_constraint(self_guiding_constr_matrix,RHS_self_guiding_constr)
            
        return basis_ceof,obj_ALP,num_self_guiding_constr
    
    
    def stop_constraint_adding(self,per_basis_func_num_constr):
        #----------------------------------------------------------------------
        # When to stop adding constraints. More involved stopping conditions 
        # can be used here.
        #----------------------------------------------------------------------
        return True

    
    def compute_lower_bound(self):
        #----------------------------------------------------------------------
        # This function is used to only compute lower bound. Starting from an 
        # FALP model, it generates a lower bound on the optimal cost
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        # Initialization
        greedy_pol_visited_state                = None
        num_sampled_basis                       = 0
        total_num_constr                        = 0
        state_list_all                          = []
        act_list_all                            = []
        best_lower_bound                        = -float('inf')
        lower_bound_list                        = []
        tot_num_times_constraints_added         = 1
        obj_ALP                                 = float('inf')    

        #----------------------------------------------------------------------
        # Show to the user what instance and model is getting solved (UI)
        print('\n')
        print('='*127)
        print('Instance number: \t'     + make_text_bold(self.mdp.instance_number))
        print('Algorithm name: \t'      + make_text_bold('Lower Bound Estimator'))
        print('State relevance: \t'     + make_text_bold(self.mdp_sampler_conf['state_relevance_name']))
        print('Random basis seed: \t'   + make_text_bold(str(self.basis_func.basis_func_random_state)))
        print('='*127)
        
        print('| {:>9s} | {:>12s} | {:>15s} | {:>8s} | {:>8s} | {:>15s} | {:>8s} | {:>16s} | {:>8s} |'.format(
              '# Basis','# Constrs','FALP Obj','ALP ConT', 'ALP SlvT','Lower Bound', 'LB RT','Best Lower Bound','TOT RT') )
        print('-'*127)

        #----------------------------------------------------------------------
        # Until finding a bounded ALP, we let ALP objective to be +inf
        tot_runtime = time.time()
        while num_sampled_basis < self.max_basis_num:
            
            #------------------------------------------------------------------
            # ADD a batch of basis functions function
            if num_sampled_basis == 0:
                new_basis_param     = self.basis_func.sample(add_constant_basis = True)   
            else:
                new_basis_param     = self.basis_func.sample(add_constant_basis = False)  
                    
            num_sampled_basis       += self.basis_func.batch_size
            self.basis_func.num_basis_func = num_sampled_basis
            basis,constrs            = self.ALP_solver.get_num_var_constr()
            print('| {:>9d} | {:>12d} |'.format(basis,constrs),end="\r")  
            
            ALP_con_runtime         =  time.time()
            
            #------------------------------------------------------------------
            # Given sampled bases, add columns to ALP
            self.add_columns_to_FALP(num_sampled_basis,new_basis_param,state_list_all,act_list_all,greedy_pol_visited_state)
            basis,constrs           = self.ALP_solver.get_num_var_constr()
            
            ALP_con_runtime         = time.time() - ALP_con_runtime
            
            if constrs >0:
                print('| {:>9d} | {:>12d} | {:>15s} | {:>8.3} |'.format(basis,constrs,'',ALP_con_runtime/60),end="\r")  
            
            ALP_slv_runtime         = time.time()
            self.ALP_solver.optimize()
            ALP_slv_runtime         = time.time() - ALP_slv_runtime
            STATUS                  = self.ALP_solver.get_status()
                
            #------------------------------------------------------------------
            # Add constraints until finding a bounded program
            if STATUS in ['INF_OR_UNBD','UNBOUNDED']:       
                additional_ALP_con_runtime              = time.time()
                new_state, new_action                   = self.constr_handler(tot_num_times_constraints_added)
                tot_num_times_constraints_added         += 1
                state_list_all.extend(new_state)
                act_list_all.extend(new_action)
                self.add_FALP_constr(new_state,new_action,False)
                total_num_constr+=1
                
                additional_ALP_con_runtime              = time.time() - additional_ALP_con_runtime
                ALP_con_runtime +=additional_ALP_con_runtime
                
                
                additional_ALP_slv_runtime              = time.time()
                self.ALP_solver.optimize()
                obj_ALP                                 = self.ALP_solver.get_optimal_value()
                additional_ALP_slv_runtime              =  time.time() - additional_ALP_slv_runtime
                ALP_slv_runtime                         += additional_ALP_slv_runtime
                
                print('| {:>9d} | {:>12d} | {:>15.1f} | {:>8.3} | {:>8.3} |'.format(basis,constrs,obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60),end="\r") 
            
            else:
                basis,constrs       = self.ALP_solver.get_num_var_constr()
                print('| {:>9d} | {:>12d} | {:>15.1f} | {:>8.3} | {:>8.3} |'.format(basis,constrs,obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60),end="\r") 
                obj_ALP             = self.ALP_solver.get_optimal_value()
        
    
            #------------------------------------------------------------------
            # VFA is now computed. Ready to find a lower bound.
            self.basis_func.set_param(self.basis_param)
            self.basis_func.set_optimal_coef(asarray(self.ALP_solver.get_optimal_solution()))
            self.lower_bound_estimator.set_basis_func(self.basis_func)
            
            lb_runtime                      = time.time()
            lb_mean, lb_lb,lb_ub, lb_se     = self.lower_bound_estimator.get_lower_bound()
            lb_runtime                      = time.time() - lb_runtime
            best_lower_bound                = max(best_lower_bound,lb_mean)
        
                
            #------------------------------------------------------------------
            # Show computed lower bound to user and store it on disk.
            print('| {:>9d} | {:>12d} | {:>15.1f} | {:>8.3} | {:>8.3} | {:>15.1f} | {:>8.3f} | {:>16.1f} | {:>8.3} |'.format(basis,constrs,obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60,
                                                                                                                  lb_mean,lb_runtime/60,best_lower_bound,(time.time()-tot_runtime)/60),end="\n")  
      
            lower_bound_list.append([basis,constrs,obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60,lb_mean, lb_lb,lb_ub, lb_se,lb_runtime/60,best_lower_bound,(time.time()-tot_runtime)/60])
            self.output_handler.save_lower_bound(lower_bound_list)

        print('='*127)

    
    
    def FALP(self):
        #----------------------------------------------------------------------
        # This function implements FALP model as well as policy-guided FALP
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        # Initialization
        greedy_pol_visited_state                = None
        num_sampled_basis                       = 0
        total_num_constr                        = 0
        state_list_all                          = []
        act_list_all                            = []
        best_lower_bound                        = -float('inf')
        best_upper_bound                        = float('inf')        
        tot_num_times_constraints_added         = 1
        tot_runtime                             = time.time()
        FALP_iterations                         = 0
        
        #----------------------------------------------------------------------
        # Show to the user what instance and model is getting solved (UI)        
        print('\n')
        print('='*155)
        print('Instance number: \t'     + make_text_bold(self.mdp.instance_number))
        if not self.update_state_rel_via_greedy_pol:
            print('Algorithm name: \t'      + make_text_bold('FALP'))
        else:
            print('Algorithm name: \t'      + make_text_bold('Policy-guided FALP'))
        print('State relevance: \t'     + make_text_bold(self.mdp_sampler_conf['state_relevance_name']))
        print('# inner updates: \t'     + make_text_bold(self.state_relevance_inner_itr))
        print('Random basis seed: \t'   + make_text_bold(str(self.basis_func.basis_func_random_state)))
        print('Optimality threshold: \t'      + make_text_bold(str(self.opt_gap_threshold*100)+'%'))
        print('='*155)
        print('| {:>9s} | {:>12s} | {:>15s} | {:>8s} | {:>8s} | {:>15s} | {:>8s} | {:>15s} | {:>8s} | {:>15s} | {:>8s} |'.format(
              '# Basis','# Constrs','FALP Obj','ALP ConT','ALP SlvT','Lower Bound', 'LB RT','Policy Cost', 'UB RT', 'Opt Gap (%)','TOT RT') )
        print('-'*155)
        
        
        #----------------------------------------------------------------------
        # Generate random basis functions until termination
        while True:
            
            ALP_con_runtime =  time.time()
            ALP_slv_runtime = time.time()
            
            #------------------------------------------------------------------
            # ADD a batch of basis functions function 
            if num_sampled_basis == 0:
                new_basis_param     = self.basis_func.sample(add_constant_basis = True)   
            else:
                new_basis_param     = self.basis_func.sample(add_constant_basis = False)  
                    
            num_sampled_basis  += self.basis_func.batch_size
            self.basis_func.num_basis_func = num_sampled_basis
            
            #------------------------------------------------------------------
            # Until finding a bounded ALP, we let ALP objective to be +inf
            obj_ALP             = float('inf')      
               
            #------------------------------------------------------------------
            # Given sampled bases, add columns to ALP
            self.add_columns_to_FALP(num_sampled_basis,new_basis_param,state_list_all,act_list_all,greedy_pol_visited_state)
            ALP_con_runtime =  time.time() - ALP_con_runtime
    
            #------------------------------------------------------------------
            # Generate or sample constraints until a stopping criteria is met
            per_basis_func_num_constr      = 0
            while True:
                basis,constrs = self.ALP_solver.get_num_var_constr()
                print('| {:>9d} | {:>12d} | {:>15.1f} | {:>8.3f} |'.format(basis,constrs,obj_ALP,ALP_con_runtime/60),end="\r")  
                self.ALP_solver.optimize()
                STATUS = self.ALP_solver.get_status()
            
                #--------------------------------------------------------------
                # Based on the FALP model solved by a solver (i.e., Gurobi),
                # we proceed
                if STATUS in ['INF_OR_UNBD','UNBOUNDED']:
                    if basis == self.basis_func.batch_size:
                        additional_ALP_slv_runtime      = time.time()
                        new_state, new_action           = self.constr_handler(tot_num_times_constraints_added)
                        tot_num_times_constraints_added += 1
                        state_list_all.extend(new_state)
                        act_list_all.extend(new_action)
                        self.add_FALP_constr(new_state,new_action,False)
                        total_num_constr                +=1
                        per_basis_func_num_constr       +=1
                        additional_ALP_slv_runtime      = time.time() - additional_ALP_slv_runtime
                        ALP_con_runtime                 +=  additional_ALP_slv_runtime
                        print('| {:>9d} | {:>12d} | {:>15.1f} | {:>8.3f} |'.format(basis,constrs,obj_ALP,ALP_con_runtime/60),end="\r")  

                    else:
                        #------------------------------------------------------
                        # Store results and exit if model is unbounded
                        print('Unbounded model ....')
                        self.output_handler.append_to_outputs(  
                                   algorithm_name              ='FALP',
                                   state_relevance_name        = self.mdp_sampler_conf['state_relevance_name'],
                                   basis_seed                  = self.basis_func.basis_func_random_state,
                                   num_basis_func              = basis,
                                   num_constr                  = constrs,
                                   FALP_obj                    = np.inf,     
                                   ALP_con_runtime             = np.inf,
                                   ALP_slv_runtime             = np.inf,
                                   best_lower_bound            = np.inf,
                                   lower_bound_lb              = np.inf,
                                   lower_bound_mean            = np.inf,
                                   lower_bound_se              = np.inf,
                                   lower_bound_ub              = np.inf,
                                   lower_bound_runtime         = np.inf,
                                   best_policy_cost            = np.inf,
                                   policy_cost_lb              = np.inf,
                                   policy_cost_mean            = np.inf,
                                   policy_cost_se              = np.inf,
                                   policy_cost_ub              = np.inf,
                                   policy_cost_runtime         = np.inf,
                                   total_runtime               = np.inf) 
                        return  True

                                
                elif STATUS in ['OPTIMAL', 'UNKNOWN'] :   
                    if not self.stop_constraint_adding(per_basis_func_num_constr):    
                        obj_ALP  = self.ALP_solver.get_optimal_value()
                        new_state, new_action = self.constr_handler(tot_num_times_constraints_added)
                        tot_num_times_constraints_added += 1
                        state_list_all.extend(new_state)
                        act_list_all.extend(new_action)
                        self.add_FALP_constr(new_state,new_action,False)
                        total_num_constr+=1
                        per_basis_func_num_constr +=1
                    
                    else:
                        obj_ALP  = self.ALP_solver.get_optimal_value()
                        break
                
                elif STATUS == 'INFEASIBLE':
                    raise Exception('ALP is infeasible ...') 
                    

                else:
                    raise Exception('Unknown status in linear programming solver') 
                
                
            ALP_slv_runtime = time.time() - ALP_slv_runtime - ALP_con_runtime
            print('| {:>9d} | {:>12d} | {:>15.1f} | {:>8.3f} | {:>8.3f} |'.format(basis,constrs,obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60),end="\r")  
            

            #------------------------------------------------------------------
            # Computing Optimality Gap 
            self.basis_func.set_param(self.basis_param)
            self.basis_func.set_optimal_coef(asarray(self.ALP_solver.get_optimal_solution()))
            self.greedy_policy.set_basis_func(self.basis_func)
            self.lower_bound_estimator.set_basis_func(self.basis_func)

            #------------------------------------------------------------------
            # Load lower bound from file
            lb_mean, lb_lb,lb_ub, lb_se,best_lb_mean    = self.output_handler.load_lower_bound() 
            lb_runtime                                  = 0
            lb_mean                                     = best_lb_mean
            best_lower_bound                            = max(best_lower_bound,lb_mean)
            
            print('| {:>9d} | {:>12d} | {:>15.1f} | {:>8.3f} | {:>8.3f} | {:>15.1f} | {:>8.3f} |'.format(basis,constrs,obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60,lb_mean,lb_runtime/60),end="\r") 
            
            #------------------------------------------------------------------
            # If user asked to update the upper bound, then do so
            start_time = time.time()
            if num_sampled_basis in  self.num_basis_to_update_pol_cost:
                cost_mean, cost_lb,cost_ub,cost_se, greedy_pol_visited_state  = self.greedy_policy.expected_cost()
                best_upper_bound = min(cost_mean,best_upper_bound)
                ub_runtime = time.time() - start_time
                opt_gap = floor((cost_mean-best_lower_bound)*100/cost_mean)
                print('| {:>9d} | {:>12d} | {:>15.1f} | {:>8.3f} | {:>8.3f} | {:>15.1f} | {:>8.3f} | {:>15.1f} | {:>8.3f} | {:>15.1f} | {:>8.3f} |'.format(
                    basis,constrs,obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60,lb_mean,lb_runtime/60,cost_mean,ub_runtime/60,opt_gap,(time.time()-tot_runtime)/60), end="\n")
        
            else:
                best_upper_bound, cost_mean, cost_lb,cost_ub,cost_se = float('inf'),float('inf'),float('inf'),float('inf'),float('inf')
                ub_runtime = 0.0
                opt_gap =  float('inf')
                print('| {:>9d} | {:>12d} | {:>15.1f} | {:>8.3f} | {:>8.3f} | {:>15.1f} | {:>8.3f} | {:>15.1f} | {:>8.3f} | {:>15.1f} | {:>8.3f} |'.format(
                    basis,constrs,obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60,lb_mean,lb_runtime/60,cost_mean,ub_runtime/60,opt_gap,(time.time()-tot_runtime)/60), end="\n")


            #------------------------------------------------------------------
            # Store results on disk and break if sampled enough bases
            if not self.update_state_rel_via_greedy_pol or  self.state_relevance_inner_itr ==0:    
                self.output_handler.append_to_outputs(  algorithm_name              ='FALP',
                                                        state_relevance_name        = self.mdp_sampler_conf['state_relevance_name'],
                                                        basis_seed                  = self.basis_func.basis_func_random_state,
                                                        num_basis_func              = basis,
                                                        num_constr                  = constrs,
                                                        FALP_obj                    = obj_ALP,     
                                                        ALP_con_runtime             = ALP_con_runtime/60,
                                                        ALP_slv_runtime             = ALP_slv_runtime/60,
                                                        best_lower_bound            = best_lower_bound,
                                                        lower_bound_lb              = lb_lb,
                                                        lower_bound_mean            = lb_mean,
                                                        lower_bound_se              = lb_se,
                                                        lower_bound_ub              = lb_ub,
                                                        lower_bound_runtime         = lb_runtime/60,
                                                        best_policy_cost            = best_upper_bound,
                                                        policy_cost_lb              = cost_lb,
                                                        policy_cost_mean            = cost_mean,
                                                        policy_cost_se              = cost_se,
                                                        policy_cost_ub              = cost_ub,
                                                        policy_cost_runtime         = ub_runtime/60,
                                                        total_runtime               = (time.time()-tot_runtime)/60)

                if num_sampled_basis >= self.max_basis_num:
                    break
            

            else:
                #--------------------------------------------------------------
                # This is for policy-guided FALP
                #--------------------------------------------------------------
                if self.state_relevance_inner_itr>0:
                    
                    #----------------------------------------------------------
                    # Iterate to update state relevance distribution
                    for inner_updates in range(self.state_relevance_inner_itr):
                        
                        print('| {:>9d} | {:>12d} |'.format(basis,constrs), end="\r")
                
                        additional_ALP_slv_runtime      = time.time()
                        obj_coef                        = sum([self.basis_func.eval_basis(state,self.basis_param) 
                                                               for state in greedy_pol_visited_state])/len(greedy_pol_visited_state)
                        self.ALP_solver.set_objective(obj_coef)
                        self.ALP_solver.prepare()
                        self.ALP_solver.optimize()
                        obj_ALP                         = self.ALP_solver.get_optimal_value()
                        self.basis_func.set_param(self.basis_param)
                        self.basis_func.set_optimal_coef(asarray(self.ALP_solver.get_optimal_solution()))
                        self.greedy_policy.set_basis_func(self.basis_func)
                        additional_ALP_slv_runtime      = time.time() - additional_ALP_slv_runtime
                        ALP_slv_runtime                 +=additional_ALP_slv_runtime 
                        
                        print('| {:>9d} | {:>12d} | {:>15.1f} | {:>8.3f} | {:>8.3f} | {:>15.1f} | {:>8.3f} | {:>15.1f} |'.format(
                                basis,constrs,obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60,lb_mean,lb_runtime/60,cost_mean), end="\r")
                        
                        additional_ub_runtime = time.time()
                        cost_mean, cost_lb,cost_ub,cost_se, greedy_pol_visited_state  = self.greedy_policy.expected_cost()
                        opt_gap = floor((cost_mean-best_lower_bound)*100/cost_mean)
                        additional_ub_runtime = time.time() - additional_ub_runtime
                        ub_runtime+=additional_ub_runtime 

                        print('| {:>9d} | {:>12d} | {:>15.1f} | {:>8.3f} | {:>8.3f} | {:>15.1f} | {:>8.3f} | {:>15.1f} | {:>8.3f} | {:>15.1f} | {:>8.3f} |'.format(
                                basis,constrs,obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60,lb_mean,lb_runtime/60,cost_mean,ub_runtime/60,opt_gap,(time.time()-tot_runtime)/60), end="\n")
                    
                        #--------------------------------------------------------------   
                        # Store results   on disk                                    
                        self.output_handler.append_to_outputs(  
                                           algorithm_name              ='PG_FALP',
                                           state_relevance_name        = self.mdp_sampler_conf['state_relevance_name'],
                                           basis_seed                  = self.basis_func.basis_func_random_state,
                                           num_basis_func              = basis,
                                           num_constr                  = constrs,
                                           FALP_obj                    = obj_ALP,     
                                           ALP_con_runtime             = ALP_con_runtime/60,
                                           ALP_slv_runtime             = ALP_slv_runtime/60,
                                           best_lower_bound            = best_lower_bound,
                                           lower_bound_lb              = lb_lb,
                                           lower_bound_mean            = lb_mean,
                                           lower_bound_se              = lb_se,
                                           lower_bound_ub              = lb_ub,
                                           lower_bound_runtime         = lb_runtime/60,
                                           best_policy_cost            = best_upper_bound,
                                           policy_cost_lb              = cost_lb,
                                           policy_cost_mean            = cost_mean,
                                           policy_cost_se              = cost_se,
                                           policy_cost_ub              = cost_ub,
                                           policy_cost_runtime         = ub_runtime/60,
                                           total_runtime               = (time.time()-tot_runtime)/60)            
            
            if num_sampled_basis >= self.max_basis_num:
                break
            
            FALP_iterations+=1 


    
    def SGFALP(self):
        #----------------------------------------------------------------------
        # This function implements self-guided FALP
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        # Initialization
        greedy_pol_visited_state                = None
        num_sampled_basis                       = 0
        total_num_constr                        = 0
        state_list_all                          = []
        act_list_all                            = []
        tot_num_times_constraints_added         = 1
        is_SGFALP                                 = True
        best_lower_bound                        = -float('inf')
        best_upper_bound                        = float('inf')        
        tot_runtime                             = time.time()
        #----------------------------------------------------------------------
        # Show to the user what instance and model is getting solved (UI)          
        print('\n')
        print('='*184)
        print('Instance number: \t'     + make_text_bold(self.mdp.instance_number))
        print('Algorithm name: \t'      + make_text_bold('SGFALP'))
        print('State relevance: \t'     + make_text_bold(self.mdp_sampler_conf['state_relevance_name']))
        print('Random basis seed: \t'   + make_text_bold(str(self.basis_func.basis_func_random_state)))
        print('Optimality threshold: \t'      + make_text_bold(str(100*self.opt_gap_threshold)+'%'))
        print('='*184)
        print('| {:>9s} | {:>12s} | {:>15s} | {:>15s} | {:>8s} | {:>8s} | {:>8s} | {:>15s} | {:>8s} | {:>15s} | {:>8s} | {:>15s} | {:>8s} |'.format(
              '# Basis','# Constrs' ,'FALP Obj','SGFALP Obj', 'ALP ConT','ALP SlvT','SG T','Lower Bound', 'LB RT','Policy Cost', 'UB RT', 'Opt Gap (%)','TOT RT') )

        print('-'*184)
        
        #----------------------------------------------------------------------
        # Generate random basis functions until termination
        while True:
            
            ALP_con_runtime =  time.time()
            ALP_slv_runtime = time.time()
            
            #------------------------------------------------------------------
            # Add a batch of basis functions function 
            if num_sampled_basis == 0:
                new_basis_param     = self.basis_func.sample(add_constant_basis = True)   
            else:
                new_basis_param     = self.basis_func.sample(add_constant_basis = False)  
                    
            num_sampled_basis       += self.basis_func.batch_size
            self.basis_func.num_basis_func = num_sampled_basis
        
            #------------------------------------------------------------------
            # Until finding a bounded ALP, we let ALP objective to be +inf
            FALP_obj_ALP             = float('inf')      
            
            #------------------------------------------------------------------
            # Given sampled bases, add columns to ALP
            self.add_columns_to_FALP(num_sampled_basis,new_basis_param,state_list_all,act_list_all,greedy_pol_visited_state)
            ALP_con_runtime          = time.time() - ALP_con_runtime
            per_basis_func_num_constr= 0
            
            #------------------------------------------------------------------
            # Generate or sample constraints until a stopping criteria is met
            while True:
                basis,constrs = self.ALP_solver.get_num_var_constr()
                
                print('| {:>9d} | {:>12d} | {:>15.1f} |'.format(basis,constrs,FALP_obj_ALP),end="\r")  
                self.ALP_solver.optimize()
            
                STATUS = self.ALP_solver.get_status()
                
                #--------------------------------------------------------------
                # Add constraints until finding a bounded program
                if STATUS in ['INF_OR_UNBD','UNBOUNDED']:
                    
                    additional_ALP_slv_runtime = time.time()
                    #----------------------------------------------------------
                    # generate a batch of new constraints
                    new_state, new_action = self.constr_handler(tot_num_times_constraints_added)
                    tot_num_times_constraints_added += 1
                    state_list_all.extend(new_state)
                    act_list_all.extend(new_action)
                    self.add_FALP_constr(new_state,new_action,is_SGFALP)
                    total_num_constr+=1
                    per_basis_func_num_constr +=1
                    
                    additional_ALP_slv_runtime = time.time() - additional_ALP_slv_runtime
                    ALP_con_runtime +=additional_ALP_slv_runtime
                    
                elif STATUS in ['OPTIMAL', 'UNKNOWN']:   
                    if not self.stop_constraint_adding(per_basis_func_num_constr):    
                        FALP_obj_ALP  = self.ALP_solver.get_optimal_value()
                        new_state, new_action = self.constr_handler(tot_num_times_constraints_added)
                        tot_num_times_constraints_added += 1
                        state_list_all.extend(new_state)
                        act_list_all.extend(new_action)
                        self.add_FALP_constr(new_state,new_action,is_SGFALP)
                        total_num_constr+=1
                        per_basis_func_num_constr +=1
                        # print('\n OPT')
                    else:
                        FALP_obj_ALP  = self.ALP_solver.get_optimal_value()
                        # print('\n OPT - Then Break')
                        break
                
                elif STATUS == 'INFEASIBLE':
                    self.ALP_solver.infeasbile_report()
                    raise Exception('ALP is infeasible ...') 

                else:
                    raise Exception('Unknown status in linear programming solver') 
                    
                    
            ALP_slv_runtime = time.time() - ALP_slv_runtime - ALP_con_runtime
            print('| {:>9d} | {:>12d} | {:>15.1f} | {:>15s} | {:>8.3f} | {:>8.3f} |'.format(basis,constrs,FALP_obj_ALP,'',ALP_con_runtime/60,ALP_slv_runtime/60),end="\r")
            
            
            #------------------------------------------------------------------
            # Add Self-guiding Constraints
            #------------------------------------------------------------------
            self_guide_time =  time.time()
            is_self_guiding_constraints_needed = False 
            VFA_prev_eval = [-float('inf')]*len(state_list_all)  
            if self.basis_func.opt_coef is not None:
                VFA_prev_eval = [self.basis_func.get_VFA(state) for state in state_list_all]
                is_self_guiding_constraints_needed = True
            
            self.basis_func.set_param(self.basis_param)
            self.basis_func.set_optimal_coef(asarray(self.ALP_solver.get_optimal_solution()))
            
            if is_self_guiding_constraints_needed:
                basis_ceof,SGFALP_obj_ALP,num_self_guiding_constr = self.incorporate_self_guiding_constraint(VFA_prev_eval,state_list_all)   
                if basis_ceof is not None:
                    self.basis_func.set_optimal_coef(asarray(basis_ceof))
                else:
                    SGFALP_obj_ALP = FALP_obj_ALP
                    
            else:
                SGFALP_obj_ALP = FALP_obj_ALP

                            
            self_guide_time = time.time()  - self_guide_time 
              
            print('| {:>9d} | {:>12d} | {:>15.1f} | {:>15.1f} | {:>8.3f} | {:>8.3f} | {:>8.3f} |'.format(basis,constrs,FALP_obj_ALP,
                                                                                            SGFALP_obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60,self_guide_time/60),end="\r")
            
            
            #------------------------------------------------------------------
            # Computing Optimality Gap 
            #------------------------------------------------------------------
            lb_mean, lb_lb,lb_ub, lb_se,best_lower_bound    = self.output_handler.load_lower_bound()
            lb_runtime                                      = 0
            lb_mean                                         = best_lower_bound

            self.greedy_policy.set_basis_func(self.basis_func)
            if num_sampled_basis in  self.num_basis_to_update_pol_cost:
                start_time = time.time()
                cost_mean, cost_lb,cost_ub,cost_se,greedy_pol_visited_state  = self.greedy_policy.expected_cost()
                ub_runtime = time.time() - start_time
                
                best_upper_bound = min(cost_mean,best_upper_bound)
                opt_gap = floor((cost_mean-lb_mean)*100/cost_mean)

                print('| {:>9d} | {:>12d} | {:>15.1f} | {:>15.1f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>15.1f} | {:>8.3f} | {:>15.1f} | {:>8.3f} | {:>15.1f} | {:>8.3f} |'.format(basis,constrs,FALP_obj_ALP,
                                                                                            SGFALP_obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60,self_guide_time/60,
                                                                                            lb_mean,lb_runtime/60,cost_mean,ub_runtime/60,opt_gap, (time.time()-tot_runtime)/60),end="\n")

            else:
                best_upper_bound, cost_mean, cost_lb,cost_ub,cost_se = float('inf'),float('inf'),float('inf'),float('inf'),float('inf')
                ub_runtime = 0.0
                opt_gap =  float('inf')
                print('| {:>9d} | {:>12d} | {:>15.1f} | {:>15.1f} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>15.1f} | {:>8.3f} | {:>15.1f} | {:>8.3f} | {:>15.1f} | {:>8.3f} |'.format(basis,constrs,FALP_obj_ALP,
                                                                                            SGFALP_obj_ALP,ALP_con_runtime/60,ALP_slv_runtime/60,self_guide_time/60,
                                                                                            lb_mean,lb_runtime/60,float('inf'),0.0, float('inf'), (time.time()-tot_runtime)/60),end="\n")           

            
            #------------------------------------------------------------------
            # Store the results on disk
            self.output_handler.append_to_outputs(  algorithm_name              ='SGFALP',
                                                    state_relevance_name        = self.mdp_sampler_conf['state_relevance_name'],
                                                    basis_seed                  = self.basis_func.basis_func_random_state,
                                                    num_basis_func              = basis,
                                                    num_constr                  = constrs,
                                                    FALP_obj                    = FALP_obj_ALP,     
                                                    ALP_con_runtime             = ALP_con_runtime/60,
                                                    ALP_slv_runtime             = ALP_slv_runtime/60,
                                                    best_lower_bound            = best_lower_bound,
                                                    lower_bound_lb              = lb_lb,
                                                    lower_bound_mean            = lb_mean,
                                                    lower_bound_se              = lb_se,
                                                    lower_bound_ub              = lb_ub,
                                                    lower_bound_runtime         = lb_runtime/60,
                                                    best_policy_cost            = best_upper_bound,
                                                    policy_cost_lb              = cost_lb,
                                                    policy_cost_mean            = cost_mean,
                                                    policy_cost_se              = cost_se,
                                                    policy_cost_ub              = cost_ub,
                                                    policy_cost_runtime         = ub_runtime/60,
                                                    total_runtime               = (time.time()-tot_runtime)/60,
                                                    SGFALP_obj                  = SGFALP_obj_ALP,
                                                    SG_runtime                  = self_guide_time/60,
                                                    )
                          
            if num_sampled_basis >= self.max_basis_num: #or opt_gap <= 100*self.opt_gap_threshold:
                break
  

    def constr_handler(self,total_num_constrains):
        #----------------------------------------------------------------------
        # Handles what constrains of semi-infinite programs
        #----------------------------------------------------------------------
        constr_gen_type = self.constr_conf['constr_gen_type']
        if constr_gen_type == 'constr_sampling':
            return self.mdp.get_state_act_for_ALP_constr(random_seed = total_num_constrains)
        else:
            raise ('Constraint handler of type ('+constr_gen_type+') is not implemented.' )
         


    
        
        
        
        
        

                