# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from MDP.instanceHandler import make_instance
from LSM import LeastSquaresMonteCarlo
from finiteTimeSelfGuidedALPs import SelfGuidedALPs
from BasisFunction.berOptSpecificBasisFunction import BerOptBasisFunction
from BasisFunction.fourierBasisFunctionsForBerOpt import FourierBasisForBerOpt


def BerOpt_run_instance():
    mdp_name                = 'BerOpt'
    instance_number_list    = ['1'] 
    seed_pack               = [111]#,222]

    for instance_number in instance_number_list:
        #----------------------------------------------------------------------
        # Set params of isntances
        #----------------------------------------------------------------------    
        BANDWIDTH                   = [1e-5] 
        MAX_BASIS_NUM               = 100
        BATCH_SIZE                  = 25
        NUM_CFA_SAMPLE_PATH         = 2000
        NUM_POL_EVAL_SAMPLE_PATH    = 10000
        INNER_SAMPLE_SIZE           = 500
        NUM_CPU_CORE                = 18
        PREPROCESS_BATCH            = 1000
        STATE_RELEVANCE_TYPE        = 'lognormal'
        ABS_VAL_UPP_BOUND           = 300
        
        for random_seed in seed_pack:
            
            seed_1,seed_2,seed_3,seed_4 = random_seed*10,random_seed*100,random_seed*1000,random_seed*10000
   
            """ -------------------------------------------------------------------
            LSML with initial dist
            -----------------------------------------------------------------------"""
            instance_conf                                                   = make_instance(mdp_name,instance_number,trial=random_seed)
            instance_conf['basis_func_conf']['basis_func_type']             = 'DFM_2012'
            instance_conf['basis_func_conf']['basis_func']                  = BerOptBasisFunction
            instance_conf['mdp_conf']['num_CFA_sample_path']                = 100000
            instance_conf['mdp_conf']['num_pol_eval_sample_path']           = 200000
            instance_conf['mdp_conf']['state_relevance_type']               = 'init'
            instance_conf['mdp_conf']['num_cpu_core']                       = NUM_CPU_CORE
            instance_conf['basis_func_conf']['basis_func_random_state']     = seed_1
            instance_conf['mdp_conf']['CFA_random_seed']                    = seed_2
            instance_conf['mdp_conf']['pol_random_seed']                    = seed_3 

            LSMN            = LeastSquaresMonteCarlo(instance_conf)
            LSMN.LSMN_fit_CFA()
            
            """ -------------------------------------------------------------------
            FALP with DFM basis
            -----------------------------------------------------------------------"""
            instance_conf                                                   = make_instance(mdp_name,instance_number,trial=random_seed)
            instance_conf['basis_func_conf']['basis_func_type']             = 'DFM_2012'
            instance_conf['basis_func_conf']['basis_func']                  = BerOptBasisFunction
            instance_conf['mdp_conf']['state_relevance_type']               = 'init'
            instance_conf['basis_func_conf']['batch_size']                  = instance_conf['mdp_conf']['num_asset']+2
            instance_conf['basis_func_conf']['max_basis_num']               = instance_conf['mdp_conf']['num_asset']+2
            instance_conf['mdp_conf']['num_CFA_sample_path']                = NUM_CFA_SAMPLE_PATH
            instance_conf['mdp_conf']['num_pol_eval_sample_path']           = NUM_POL_EVAL_SAMPLE_PATH
            instance_conf['mdp_conf']['num_cpu_core']                       = NUM_CPU_CORE
            instance_conf['mdp_conf']['inner_sample_size']                  = INNER_SAMPLE_SIZE
            instance_conf['solver_conf']['batch_size']                      = BATCH_SIZE 
            instance_conf['solver_conf']['abs_val_upp_bound']               = ABS_VAL_UPP_BOUND 
            instance_conf['basis_func_conf']['basis_func_random_state']     = seed_1
            instance_conf['mdp_conf']['CFA_random_seed']                    = seed_2
            instance_conf['mdp_conf']['pol_random_seed']                    = seed_3 
            instance_conf['mdp_conf']['inner_sample_seed']                  = seed_4
              
            Model           = SelfGuidedALPs(instance_conf)
            Model.FALP_non_adaptive_fit_fix_basis()                              
            
            """ -------------------------------------------------------------------
            Computing expected VFAs 
            -----------------------------------------------------------------------"""
            instance_conf                                                   = make_instance(mdp_name,instance_number,trial=random_seed)
            instance_conf['basis_func_conf']['basis_func_type']             = 'fourier'
            instance_conf['basis_func_conf']['basis_func']                  = FourierBasisForBerOpt
            instance_conf['basis_func_conf']['bandwidth']                   = BANDWIDTH
            instance_conf['basis_func_conf']['batch_size']                  = BATCH_SIZE
            instance_conf['basis_func_conf']['max_basis_num']               = MAX_BASIS_NUM
            instance_conf['basis_func_conf']['preprocess_batch']            = PREPROCESS_BATCH
            instance_conf['mdp_conf']['state_relevance_type']               = STATE_RELEVANCE_TYPE
            instance_conf['mdp_conf']['num_CFA_sample_path']                = NUM_CFA_SAMPLE_PATH
            instance_conf['mdp_conf']['num_pol_eval_sample_path']           = NUM_POL_EVAL_SAMPLE_PATH
            instance_conf['mdp_conf']['num_cpu_core']                       = NUM_CPU_CORE
            instance_conf['mdp_conf']['inner_sample_size']                  = INNER_SAMPLE_SIZE
            instance_conf['solver_conf']['batch_size']                      = BATCH_SIZE 
            instance_conf['solver_conf']['abs_val_upp_bound']               = ABS_VAL_UPP_BOUND 
            instance_conf['basis_func_conf']['basis_func_random_state']     = seed_1
            instance_conf['mdp_conf']['CFA_random_seed']                    = seed_2
            instance_conf['mdp_conf']['pol_random_seed']                    = seed_3 
            instance_conf['mdp_conf']['inner_sample_seed']                  = seed_4
            
            Model = SelfGuidedALPs(instance_conf)
            Model.compute_expected_basis_func(seed_1)
            
            """ -------------------------------------------------------------------
            FALP Non-adaptive; One-shot
            -----------------------------------------------------------------------"""
            instance_conf                                                   = make_instance(mdp_name,instance_number,trial=random_seed)
            instance_conf['basis_func_conf']['basis_func_type']             = 'fourier'
            instance_conf['basis_func_conf']['basis_func']                  = FourierBasisForBerOpt
            instance_conf['basis_func_conf']['bandwidth']                   = BANDWIDTH
            instance_conf['basis_func_conf']['batch_size']                  = MAX_BASIS_NUM
            instance_conf['basis_func_conf']['max_basis_num']               = MAX_BASIS_NUM
            instance_conf['basis_func_conf']['preprocess_batch']            = PREPROCESS_BATCH
            instance_conf['mdp_conf']['state_relevance_type']               = STATE_RELEVANCE_TYPE
            instance_conf['mdp_conf']['num_CFA_sample_path']                = NUM_CFA_SAMPLE_PATH
            instance_conf['mdp_conf']['num_pol_eval_sample_path']           = NUM_POL_EVAL_SAMPLE_PATH
            instance_conf['mdp_conf']['num_cpu_core']                       = NUM_CPU_CORE
            instance_conf['mdp_conf']['inner_sample_size']                  = INNER_SAMPLE_SIZE
            instance_conf['solver_conf']['batch_size']                      = MAX_BASIS_NUM 
            instance_conf['solver_conf']['abs_val_upp_bound']               = ABS_VAL_UPP_BOUND 
            instance_conf['basis_func_conf']['basis_func_random_state']     = seed_1
            instance_conf['mdp_conf']['CFA_random_seed']                    = seed_2
            instance_conf['mdp_conf']['pol_random_seed']                    = seed_3 
            instance_conf['mdp_conf']['inner_sample_seed']                  = seed_4
               
            Model           = SelfGuidedALPs(instance_conf)
            Model.FALP_non_adaptive_fit_VFA()
                    
            
            """ -------------------------------------------------------------------
            FGLP 
            -----------------------------------------------------------------------"""  
            instance_conf                                                   = make_instance(mdp_name,instance_number,trial=random_seed)
            instance_conf['basis_func_conf']['basis_func_type']             = 'fourier'
            instance_conf['basis_func_conf']['basis_func']                  = FourierBasisForBerOpt
            instance_conf['basis_func_conf']['bandwidth']                   = BANDWIDTH
            instance_conf['basis_func_conf']['batch_size']                  = BATCH_SIZE
            instance_conf['basis_func_conf']['max_basis_num']               = MAX_BASIS_NUM
            instance_conf['basis_func_conf']['preprocess_batch']            = PREPROCESS_BATCH
            instance_conf['mdp_conf']['state_relevance_type']               = STATE_RELEVANCE_TYPE
            instance_conf['mdp_conf']['num_CFA_sample_path']                = NUM_CFA_SAMPLE_PATH
            instance_conf['mdp_conf']['num_pol_eval_sample_path']           = NUM_POL_EVAL_SAMPLE_PATH
            instance_conf['mdp_conf']['num_cpu_core']                       = NUM_CPU_CORE
            instance_conf['mdp_conf']['inner_sample_size']                  = INNER_SAMPLE_SIZE
            instance_conf['solver_conf']['batch_size']                      = BATCH_SIZE 
            instance_conf['solver_conf']['abs_val_upp_bound']               = ABS_VAL_UPP_BOUND 
            instance_conf['basis_func_conf']['basis_func_random_state']     = seed_1
            instance_conf['mdp_conf']['CFA_random_seed']                    = seed_2
            instance_conf['mdp_conf']['pol_random_seed']                    = seed_3 
            instance_conf['mdp_conf']['inner_sample_seed']                  = seed_4
            
            Model           = SelfGuidedALPs(instance_conf)
            Model.SGFALP_fit_VFA()


if __name__== "__main__":
    BerOpt_run_instance()
    
