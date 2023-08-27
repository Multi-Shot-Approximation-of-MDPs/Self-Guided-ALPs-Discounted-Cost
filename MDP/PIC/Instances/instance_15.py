"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from scipy.stats import uniform
from math import log,gamma,pi
from numpy.linalg import norm
from MDP.PIC.perishableInventory import PerishableInventoryPartialBacklogLeadTime
from BasisFunction.fourierBasisFunctions import FourierBasis
from Wrapper.gurobiWrapper import gurobi_LP_wrapper 
from utils import is_PIC_config_valid

def get_experiment_setup():
    
    mdp_conf                                    =   {}
    mdp_noise_conf                              =   {}
    mdp_sampler_conf                            =   {}
    greedy_pol_conf                             =   {}
    lower_bound_conf                            =   {}
    constr_conf                                 =   {}
    basis_func_conf                             =   {}
    solver                                      =   {}                
    misc_conf                                   =   {}    
 
    #--------------------------------------------------------------------------
    # Miscellaneous configuration
    misc_conf.update({
                'opt_gap_threshold'             :     0.05,
                'num_cpu_core'                  :     16            })
   
    #--------------------------------------------------------------------------
    # MDP configuration
    mdp_conf.update({
        
                'mdp'                           :   PerishableInventoryPartialBacklogLeadTime,
                'mdp_name'                      :   'PIC',
                'instance_number'               :   '15',
                'lead_time'                     :   4,
                'life_time'                     :   2,
                'dim_act'                       :   1,
                'discount'                      :   .95,
                'random_seed'                   :   12345 
                
        })
    

    mdp_conf.update({
                'dim_state'                     :   mdp_conf['lead_time'] + mdp_conf['life_time'] -1,
                'purchase_cost'                 :   10*pow(mdp_conf.get('discount'),2),
                
                'holding_cost'                  :   1,
                'disposal_cost'                 :   2,
                'backlogg_cost'                 :   8,
                'lostsale_cost'                 :   1000,
                
                'max_order'                     :   10,     
                'max_backlog'                   :  -10,  
                
                
        })
    
    
    mdp_noise_conf.update({
        
                'dist_mean'                     :   5,
                'dist_std'                      :   5,
                'dist_min'                      :   0,
                'dist_max'                      :   10,  
                'num_sample_noise'              :   2000,
        
        })
    
    
    
    #--------------------------------------------------------------------------
    # Greedy policy configuration       
    greedy_pol_conf.update({
                'update_state_rel_via_greedy_pol'   :   None,
                'len_traj'                          :   1000,    
                'num_traj'                          :   200,   
                'init_state_sampler'                :   [uniform(loc=5.0,scale=0.0) for _ in range(mdp_conf['dim_state'])],
                'num_cpu_core'                      :   misc_conf['num_cpu_core']   ,
                'action_selector_name'              :   'discretization',
                'action_discrete_param'             :   1,
                'state_round_decimal'               :   0,    
                'num_basis_to_update_pol_cost'      :   [],
        })
    
    
    mdp_sampler_conf.update({
        
                'state_sampler'                     :   [ uniform(loc = mdp_conf.get('max_backlog'),scale = -mdp_conf.get('max_backlog')+mdp_conf.get('max_order'))]+\
                                                        [ uniform(loc = 0,scale = mdp_conf.get('max_order')) for _ in range(mdp_conf['dim_state']-1) ],
                            
                'act_sampler'                       :   uniform( loc   = 0, scale=mdp_conf.get('max_order')), 

                'state_relevance_name'              :   None,
                'state_relevance'                   :   None,

        })   
    
    

    
    
    #--------------------------------------------------------------------------
    # Constraints configuration       
    constr_conf.update({
                'constr_gen_type'               :   'constr_sampling',
                'max_num_constr'                :   None, 
                'constr_gen_batch_size'         :   None, 
                'num_random_sample'             :   None,
                'constr_gen_stop_param'         :   None,
                
        })
    
    
    #--------------------------------------------------------------------------
    # Basis functions setting
    basis_func_conf.update({ 
                'basis_func_type'               :   'fourier',
                'basis_func'                    :   FourierBasis,       
                'dim_state'                     :   mdp_conf['dim_state'],
                'max_basis_num'                 :   None,
                'batch_size'                    :   None,
                'bandwidth'                     :   None,
                'basis_func_random_state'       :   None,
        })
    
    
    
    #--------------------------------------------------------------------------
    # Lower bound configuration
    lower_bound_conf.update({
         
                'dim_state_act'                 :   mdp_conf['dim_state'] + mdp_conf['dim_act'],
                'radius_ball_in_state_action'   :   mdp_conf['max_order']/2,
                'volume_state_action'           :   (mdp_conf['max_order'] - mdp_conf['max_backlog'])*pow(mdp_conf['max_order'],3),
                'diameter_state_action'         :   norm([(mdp_conf['max_order'] - mdp_conf['max_backlog']),
                                                           mdp_conf['max_order'],
                                                           mdp_conf['max_order']]),
                'lower_bound_algo_name'         : 'LNS',
                'num_MC_init_states'            : 500,
                'MC_sample_path_len'            : 1000,
                'MC_burn_in'                    : 500,
                
                'num_cpu_core'                  : misc_conf['num_cpu_core']  
                
        })

    


    #--------------------------------------------------------------------------
    # LP solver configuration
    solver.update({
                'solver_name'                   :     gurobi_LP_wrapper,
                'num_cpu_core'                  :     misc_conf['num_cpu_core'],
                'dual_opt_gap'                  :     0.05,
                })


    
    #--------------------------------------------------------------------------
    # All configurations
    config                                      =   {}
    config['mdp_conf']                          =   mdp_conf
    config['mdp_noise_conf']                    =   mdp_noise_conf
    config['mdp_sampler_conf']                  =   mdp_sampler_conf
    config['greedy_pol_conf']                   =   greedy_pol_conf
    config['lower_bound_conf']                  =   lower_bound_conf
    config['constr_conf']                       =   constr_conf
    config['basis_func_conf']                   =   basis_func_conf
    config['solver']                            =   solver
    config['misc_conf']                         =   misc_conf
    
    
    is_PIC_config_valid(config)
    return config



