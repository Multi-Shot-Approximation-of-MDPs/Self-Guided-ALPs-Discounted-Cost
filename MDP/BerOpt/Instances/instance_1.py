# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from MDP.BerOpt.bermudanOptionPricing import BermudanOption
import numpy as np


VOLATILITY          = 0.20
INTEREST_RATE       = 0.05
INIT_PRICE          = 90
NUM_EXERCISE        = 54

def get_experiment_setup():
    #--------------------------------------------------------------------------
    #   This function specifies the parameters of a perishable inventory 
    #   control instance, including the MDP cost function parameters, dimension
    #   of the state space, demand distribution, optimality gap threshold, 
    #   number of CPU cores, etc.
    #--------------------------------------------------------------------------
    mdp_conf                                    =   {}
    upp_bound_conf                              =   {}
    basis_func_conf                             =   {}
    solver                                      =   {}                
    misc_conf                                   =   {}

    #--------------------------------------------------------------------------
    # Miscellaneous configuration
    #--------------------------------------------------------------------------
    misc_conf.update({ 
                'num_cpu_core'                  :   12,
                })

    #--------------------------------------------------------------------------
    # MDP configuration
    #--------------------------------------------------------------------------
    mdp_conf.update({
                'mdp'                           :   BermudanOption,
                'mdp_name'                      :   'BerOpt',
                'instance_number'               :   '1',
                'time_horizon'                  :   3,
                'num_asset'                     :   4,
                'dim_act'                       :   1,
                'knock_out_price'               :   170.0,
                'strike_price'                  :   100.0,
                'CFA_random_seed'               :   None, 
                'pol_random_seed'               :   None, 
                'num_CFA_sample_path'           :   None,
                'num_pol_eval_sample_path'      :   None,
                'inner_sample_size'             :   None,
                'inner_sample_seed'             :   None,
                'num_cpu_core'                  :   misc_conf['num_cpu_core'],
                'state_relevance_type'          :   None, 
                })
    
    mdp_conf.update({
                'dim_state'                     :   mdp_conf['num_asset'] + 1,
                'volatility'                    :   np.array([VOLATILITY for _ in range(mdp_conf['num_asset'])]) ,
                'init_price'                    :   np.array([INIT_PRICE for _ in range(mdp_conf['num_asset'])]),
                'interest_rate'                 :   np.array([INTEREST_RATE for _ in range(mdp_conf['num_asset'])]) ,
                'cor_matrix'                    :   np.identity(mdp_conf['num_asset']),
                'num_stages'                    :   NUM_EXERCISE,
                'discount'                      :   np.exp(-INTEREST_RATE*(mdp_conf['time_horizon']/NUM_EXERCISE)),
    })
    
    #--------------------------------------------------------------------------
    # Basis functions setting
    #--------------------------------------------------------------------------
    basis_func_conf.update({ 
                'basis_func_type'               :   None, 
                'basis_func'                    :   None, 
                'dim_state'                     :   mdp_conf['dim_state'],
                'max_basis_num'                 :   None,
                'batch_size'                    :   None,
                'bandwidth'                     :   None, 
                'basis_func_random_state'       :   None,  
                'num_cpu_core'                  :   misc_conf['num_cpu_core']
        })
    
    #--------------------------------------------------------------------------
    # Upper bound setting
    #--------------------------------------------------------------------------
    upp_bound_conf.update({     
                'upp_bound'                     : None,
                'upp_bound_algo_name'           : 'LNS',
                'num_MC_init_states'            : None,
                'MC_sample_path_len'            : None,
                'MC_burn_in'                    : None,     
                'num_cpu_core'                  : None
                
        })
    
    #--------------------------------------------------------------------------
    # ALP solver setting
    #--------------------------------------------------------------------------
    solver.update({
                'num_cpu_core'                  : misc_conf['num_cpu_core'],
                'num_stages'                    : mdp_conf['num_stages'],
                'basis_func_batch_size'         : basis_func_conf['batch_size'],
                'abs_val_upp_bound'             : None,
        })
    
    config                                      =   {}
    config['mdp_conf']                          =   mdp_conf
    config['basis_func_conf']                   =   basis_func_conf
    config['misc_conf']                         =   misc_conf
    config['upp_bound_conf']                    =   upp_bound_conf
    config['solver_conf']                       =   solver
    return config
