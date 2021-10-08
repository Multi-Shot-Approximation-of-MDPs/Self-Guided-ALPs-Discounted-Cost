# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from scipy.stats import sem,t
import numpy as np
import pandas as pd
import os
from datetime import datetime
from shutil import copyfile
from itertools import chain, combinations 


def index_unique_sub_list(input_list):
    #--------------------------------------------------------------------------
    # Returns the location of locations in a list with unique values
    #--------------------------------------------------------------------------
    _, indices = np.unique(np.asarray(input_list), return_index=True,axis=0)
    return indices

def mean_confidence_interval(data, confidence=0.95):
    #--------------------------------------------------------------------------
    # Computes confidence interval around mean
    #--------------------------------------------------------------------------
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h,se

def make_text_bold(string):
    #--------------------------------------------------------------------------
    # Makes a text bold in terminal
    #--------------------------------------------------------------------------
    return '{}{}{}'.format('\033[1m', string, '\033[0m')


class output_handler:
    #--------------------------------------------------------------------------
    # Collects and stores outputs of an algorithm.
    #--------------------------------------------------------------------------

    def __init__(self,instance_conf):
        #----------------------------------------------------------------------
        # Inititalization
        #----------------------------------------------------------------------
        self.mdp_name                       = instance_conf['mdp_conf']['mdp_name']             
        self.basis_func_type                = instance_conf['basis_func_conf']['basis_func_type']   
        self.batch_size                     = instance_conf['basis_func_conf']['batch_size']   
        self.instance_number                = instance_conf['mdp_conf']['instance_number'] 
        self.state_relevance_inner_itr      = instance_conf['greedy_pol_conf']['state_relevance_inner_itr']
        self.output_table                   = pd.DataFrame()
        self.path                           = None
        self.filename                       = None
        self.lb_filename                    = '/LowerBound_' + self.mdp_name +'.csv'
        self.setup_output_path()
        
    
    def setup_output_path(self):
        #----------------------------------------------------------------------
        # Set the path to store outputs
        #----------------------------------------------------------------------
        self.path       = 'Output/' + self.mdp_name
        assert os.path.isdir(self.path)
        if not os.path.isdir(self.path + '/instance_'+self.instance_number):
            os.mkdir(self.path + '/instance_'+self.instance_number)
            
        self.path = self.path + '/instance_'+self.instance_number
        copyfile('MDP/'+ self.mdp_name+ '/Instances/instance_'+self.instance_number+'.py', self.path + '/instance_'+self.instance_number+'.py')
        
    def save_lower_bound(self,lower_bound_list):
        #----------------------------------------------------------------------
        # Save lower bound into a file
        #----------------------------------------------------------------------
        pd.DataFrame(lower_bound_list,columns=['# bases','# constrs','FALP Obj','ALP ConT', 'ALP SlvT','lb_mean', 'lb_lb','lb_ub', 'lb_se','LB RT','best_lower_bound','TOT RT']).to_csv(self.path + self.lb_filename)
        
    def load_lower_bound(self):
        #----------------------------------------------------------------------
        # Load lower bound from a file
        #----------------------------------------------------------------------
        df = pd.read_csv(self.path + self.lb_filename)
        df = df[['lb_mean', 'lb_lb','lb_ub', 'lb_se','best_lower_bound']]
        return np.asarray(df.iloc[[-1]]).flatten()
        
    
    def append_to_outputs( self,      
                algorithm_name:         str,    # FALP, FGLP
                state_relevance_name:   str,    # uniform, (5,5,5), greedy_policy
                basis_seed:             int,    # seed number for basis function
                num_basis_func:         int,    # 10, 20, ...
                num_constr:             int,    # num of constraints in ALP 
                FALP_obj:               float,  # value of ALP objective
                ALP_con_runtime:        float,  # time to construct ALP to get VFA
                ALP_slv_runtime:        float,  # time tosolve ALP to get VFA
                best_lower_bound:       float,  # best lower bound on the optimal cost until the current iteration
                lower_bound_lb:         float,  # 95% lower bound on the optimal cost lower bound
                lower_bound_mean:       float,  # mean lower bound on the optimal cost
                lower_bound_se:         float,  # standard error of the lower bound on the optimal cost
                lower_bound_ub:         float,  # 95% upper bound on the optimal cost lower bound
                lower_bound_runtime:    float,  # runtime of computing lower bound on  the optimla cost
                best_policy_cost:       float,  # best upper bound (policy cost) on the optimal cost until the current iteration
                policy_cost_lb:         float,  # 95% lower bound on the greedy policy cost
                policy_cost_mean:       float,  # mean of the greedy policy cost
                policy_cost_se:         float,  # standard error of greedy policy cost
                policy_cost_ub:         float,  # 95% upper bound on the greedy policy cost
                policy_cost_runtime:    float,  # runtime of computing greedy policy cost  
                total_runtime:          float,  # total runtime
                SGFALP_obj:             float = None,  
                SG_runtime:             float = None,  
                ):
        
        #----------------------------------------------------------------------
        # Having algorithm's results up to the current iteration, append
        # new results to it.
        #----------------------------------------------------------------------
        self.filename   = '/' + self.mdp_name + '_' + self.basis_func_type + '_' + algorithm_name  + '_' +\
                             state_relevance_name+'_inner_update_'+str(self.state_relevance_inner_itr)+\
                                '_Batch_'+str(self.batch_size)+ '_seed_' +  str(basis_seed)  +'.csv'

        SGFALP_         = None if SGFALP_obj is None else[round(SGFALP_obj,1)]
        SG_runtime_     = None if SG_runtime is None else[round(SG_runtime,4)]     
        if not policy_cost_mean in [0.0,float('inf')]:
            opt_gap_    = 100*(policy_cost_mean - lower_bound_mean)/policy_cost_mean
        else:
            opt_gap_    = float('inf')

        info            =\
              { 'update time'               : datetime.now().strftime("%d-%m-%Y - %H : %M"),        
                'mdp'                       : [self.mdp_name],
                'algorithm'                 : [algorithm_name],
                'basis_func_seed'           : [basis_seed], 
                'state relevance'           : [state_relevance_name],
                '# bases'                   : [num_basis_func],
                '# constrs'                 : [num_constr],
                'FALP obj'                  : [round(FALP_obj,1)],
                'SGFALP'                    : SGFALP_,
                'ALP Constr time'           : [round(ALP_con_runtime,4)],
                'ALP Solve time'            : [round(ALP_slv_runtime,4)],
                'SG time'                   : SG_runtime_,
                'best_lower_bound'          : [round(best_lower_bound,1)],
                'lower bound lb'            : [round(lower_bound_lb,1)],
                'lower bound mean'          : [round(lower_bound_mean,1)],
                'lower bound se'            : [round(lower_bound_se,2)],
                'lower bound ub'            : [round(lower_bound_ub,1)],
                'lower bound runtime'       : [round(lower_bound_runtime,4)],
                'best_policy_cost'          : [round(best_policy_cost,1)],
                'policy cost lb'            : [round(policy_cost_lb,1)],
                'policy cost mean'          : [round(policy_cost_mean,1)],
                'policy cost se'            : [round(policy_cost_se,2)],
                'policy cost ub'            : [round(policy_cost_ub,1)],
                'policy cost runtime'       : [round(policy_cost_runtime,4)],
                'tot runtime'               : [round(total_runtime,4)],
                'opt gap'                   : [round(opt_gap_,1)],
                'lower bound fluctuation'   : [round(100*(lower_bound_mean - best_lower_bound)/best_lower_bound,1)],
                'policy cost fluctuation'   : [round(100*(best_policy_cost - policy_cost_mean)/best_policy_cost,1)],
                }

        self.output_table = self.output_table.append(pd.DataFrame(info),ignore_index = True)
        self.output_table.to_csv(self.path + self.filename)
        
        

def is_PIC_config_valid(config):
    #--------------------------------------------------------------------------
    # Add assertion if you need to check an instance of the PIC application
    # is "valid". This function is called inside each instance.
    #--------------------------------------------------------------------------
    pass

def prune_similar_columns(matrix,threshold):
    #--------------------------------------------------------------------------
    # Prune similar columns of a matrix; not used in the current code.
    #--------------------------------------------------------------------------
    already_considered      = []
    similar_columns         = []
    for i in range(len(matrix.T)):
        column              = matrix.T[i]
        if not i in already_considered:
            column          = np.asarray([column]).T
            diff            = column - matrix 
            norm            = np.max(np.abs(diff),axis=0)
            index           = [_ for _ in range(len(norm)) if norm[_] < threshold]
            already_considered += index        
            similar_columns.append((i,index))

    keep                    = [similar_columns[_][0] for _ in range(len(similar_columns))]  
    remove                  = [_ for _ in range(len(similar_columns)) if not _ in keep]            

    return remove



class output_handler_option_pricing:
    #--------------------------------------------------------------------------
    # Collects and stores outputs of an algorithm.
    #--------------------------------------------------------------------------
    def __init__(self,instance_conf):
        #----------------------------------------------------------------------
        # Inititalization
        #----------------------------------------------------------------------
        self.mdp_name                       = instance_conf['mdp_conf']['mdp_name']             
        self.state_relevance_type           = instance_conf['mdp_conf']['state_relevance_type']  
        self.basis_func_type                = instance_conf['basis_func_conf']['basis_func_type']   
        self.batch_size                     = instance_conf['basis_func_conf']['batch_size']   
        self.instance_number                = instance_conf['mdp_conf']['instance_number'] 
        self.output_table                   = pd.DataFrame()
        self.path                           = None
        self.filename                       = None
        self.setup_output_path()
        
    
    def setup_output_path(self):
        #----------------------------------------------------------------------
        # Set the path to store outputs
        #----------------------------------------------------------------------
        self.path       = 'Output/' + self.mdp_name
        assert os.path.isdir(self.path)
        if not os.path.isdir(self.path + '/instance_'+self.instance_number):
            os.mkdir(self.path + '/instance_'+self.instance_number)

        self.path = self.path + '/instance_'+self.instance_number
        copyfile('MDP/'+ self.mdp_name+ '/Instances/instance_'+self.instance_number+'.py', self.path + '/instance_'+self.instance_number+'.py')
        
    
    def append_to_outputs( self,      
                algorithm_name:         str,    # FALP, FGLP
                basis_seed:             int,    # seed number for basis function
                num_basis_func:         int,    # 10, 20, ...
                num_constr:             int,    # num of constraints in ALP 
                FALP_obj:               float,  # value of ALP objective
                ALP_con_runtime:        float,  # time to construct ALP to get VFA
                ALP_slv_runtime:        float,  # time tosolve ALP to get VFA
                train_LB_mean:          float,  # lower bound on the training sample paths
                train_LB_SE:            float,  # lower bound on the training sample paths
                test_LB_mean:           float,  # lower bound on the training sample paths
                test_LB_SE:             float,  # lower bound on the training sample paths
                test_LB_runtime:        float,  #    untime of computing greedy policy cost 
                upp_bound               = 0.0,
                upp_bound_sd            = 0.0,
                best_upp_bound          = 0.0,
                upp_bound_runtime       = 0.0,
                train_opt_gap           = 0.0,
                test_opt_gap            = 0.0,
                total_runtime           = 0.0,  # total runtime
                ):
        
        #----------------------------------------------------------------------
        # Having algorithm's results up to the current iteration, append
        # new results to it.
        #----------------------------------------------------------------------
        self.filename = '/' + self.mdp_name + '_' + self.basis_func_type + '_' + algorithm_name + '_'+ self.state_relevance_type + '_batch_'+str(self.batch_size)+ '_seed_' +  str(basis_seed)  +'.csv'
          
        info ={ 'update time'                 : datetime.now().strftime("%d-%m-%Y - %H : %M"),        
                'mdp'                         : [self.mdp_name],
                'algorithm'                   : [algorithm_name],
                'basis_func_seed'             : [basis_seed], 
                '# bases'                     : [num_basis_func],
                '# constrs'                   : [num_constr],
                'FALP obj'                    : [round(FALP_obj,1)],
                'ALP Constr time'             : [round(ALP_con_runtime,4)],
                'ALP Solve time'              : [round(ALP_slv_runtime,4)],
                'Train pol cost mean'         : [round(train_LB_mean,4)],  
                'Train pol cost SE'           : [round(train_LB_SE,4)],  
                'Test pol cost mean'          : [round(test_LB_mean,4)],  
                'Test pol cost SE'            : [round(test_LB_SE,4)],  
                'Test pol runbtime'           : [round(test_LB_runtime,1)],  
                'Upper Bound'                 : [round(upp_bound,4)],
                'Upper Bound SD'              : [round(upp_bound_sd,4)],
                'Best Upper Bound'            : [round(best_upp_bound,4)],
                'Upper Bound runbtime'        : [round(upp_bound_runtime,1)],
                'Train Opt Gap'               : [round(train_opt_gap,4)],
                'Test Opt Gap'                : [round(test_opt_gap,4)],
                'tot runtime'                 : [round(total_runtime,4)],
                }

        self.output_table = self.output_table.append(pd.DataFrame(info),ignore_index = True)
        self.output_table.to_csv(self.path + self.filename)


