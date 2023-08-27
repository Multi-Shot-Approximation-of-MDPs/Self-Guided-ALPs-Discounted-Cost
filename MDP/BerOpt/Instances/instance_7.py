"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""


from MDP.BerOpt.bermudanOptionPricing import BermudanOption
from BasisFunction.berOptSpecificBasisFunction import BerOptBasisFunction
import numpy as np
from scipy.stats import uniform
from BasisFunction.fourierBasisFunctionsForBerOpt import FourierBasisForBerOpt


volatility_         = 0.20
interest_rate_      = 0.05
init_price_         = 90.0    
num_exercise_       = 54


def get_experiment_setup():
    
    mdp_conf                                    =   {}
    upp_bound_conf                              =   {}
    basis_func_conf                             =   {}
    solver                                      =   {}                
    misc_conf                                   =   {}


    misc_conf.update({ 
                'num_cpu_core'                  :   12,
                })

    
    mdp_conf.update({
                'mdp'                           :   BermudanOption,
                'mdp_name'                      :   'BerOpt',
                'instance_number'               :   '7',
                'time_horizon'                  :   3,
                'num_asset'                     :   16,
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
                'volatility'                    :   np.array([volatility_ for _ in range(mdp_conf['num_asset'])]) ,
                'init_price'                    :   np.array([init_price_ for _ in range(mdp_conf['num_asset'])]),
                'interest_rate'                 :   np.array([interest_rate_ for _ in range(mdp_conf['num_asset'])]) ,
                'cor_matrix'                    :   np.identity(mdp_conf['num_asset']),
                'num_stages'                    :   num_exercise_,
                'discount'                      :   np.exp(-interest_rate_*(mdp_conf['time_horizon']/num_exercise_)),
                # 'state_relevance'               :   [uniform(loc=init_price_,scale=0.0) for _ in range(mdp_conf['num_asset'])] + [uniform(loc=0.0,scale=0.0)] ,
                # 'state_relevance'               :   [uniform(loc=init_price_,scale=(mdp_conf['knock_out_price']-init_price_)*.2) for _ in range(mdp_conf['num_asset'])] + [uniform(loc=0.0,scale=0.0)] ,
        
    })
    


    #--------------------------------------------------------------------------
    # Basis functions setting
    basis_func_conf.update({ 
                'basis_func_type'               :   None, #'fourier',
                'basis_func'                    :   None, #FourierBasisForBerOpt,       
                'dim_state'                     :   mdp_conf['dim_state'],
                'max_basis_num'                 :   None,
                'batch_size'                    :   None,
                'bandwidth'                     :   None, #[.001],
                'basis_func_random_state'       :   None, #111,
                'num_cpu_core'                  :   misc_conf['num_cpu_core']
        })
    
    

    
    
    # #--------------------------------------------------------------------------
    # # Basis functions setting
    # basis_func_conf.update({ 
    #             'basis_func_type'               :   'BerOpt',
    #             'basis_func'                    :   BerOptBasisFunction,       
    #             'dim_state'                     :   mdp_conf['dim_state'],
    #             'max_basis_num'                 :   None,
    #             'batch_size'                    :   None,
    #             'bandwidth'                     :   None,
    #             'basis_func_random_state'       :   None,
    #     })
    
    
    # misc_conf.update({ 
    #             'num_cpu_core'                  :   12,
    #             })

    upp_bound_conf.update({
         
                # 'dim_state_act'                 :   mdp_conf['dim_state'] + mdp_conf['dim_act'],
                # 'radius_ball_in_state_action'   :   mdp_conf['max_order']/2,
                # 'volume_state_action'           :   (mdp_conf['max_order'] - mdp_conf['max_backlog'])*pow(mdp_conf['max_order'],3),
                # 'diameter_state_action'         :   norm([(mdp_conf['max_order'] - mdp_conf['max_backlog']),
                                                           # mdp_conf['max_order'],
                                                           # mdp_conf['max_order']]),
                                                           

                'upp_bound_algo_name'           : 'LNS',
                'num_MC_init_states'            : 200,
                'MC_sample_path_len'            : 100,
                'MC_burn_in'                    : 0,
                
                'num_cpu_core'                  : misc_conf['num_cpu_core']  
                
        })
    
    
    solver.update({
                'num_cpu_core'                  : misc_conf['num_cpu_core'],
                'num_stages'                    : mdp_conf['num_stages'],
                'basis_func_batch_size'         : basis_func_conf['batch_size'],
                'abs_val_upp_bound'             : None,
        })
    
    
    #
    #--------------------------------------------------------------------------
    # All configurations
    config                                      =   {}
    config['mdp_conf']                          =   mdp_conf
    config['basis_func_conf']                   =   basis_func_conf
    config['misc_conf']                         =   misc_conf
    config['upp_bound_conf']                    =   upp_bound_conf
    config['solver_conf']                       =   solver
    
    
    
    return config