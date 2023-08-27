# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from MDP.instanceHandler import make_instance
from Algorithm.LSM import LeastSquaresMonteCarlo
from BasisFunction.berOptSpecificBasisFunction import BerOptBasisFunction
import numpy as np
import sys

if __name__== "__main__":
    
    # Load parameters
    algo_name                       = 'LSMN'
    basis_func_type                 = 'DFM'
    instance_number                 = int(sys.argv[1])
    CFA_num_train_sample_path       = int(sys.argv[2])
    CFA_num_test_sample_path        = int(sys.argv[3])
    random_seed                     = int(sys.argv[4])
    num_cpu_core                    = int(sys.argv[5])
    IR_inner_sample_size            = int(sys.argv[6])

    
    # Create the  bermudan options pricing instance
    instance_conf = make_instance('BerOpt',str(instance_number), random_seed)
    instance_conf['LSM_conf']   = {}
    bandwidth                   = [0.]
    
    
    # Set up random seeds using for simulations
    np.random.seed(random_seed)
    seeds = list(np.random.randint(1e1,1e3,4))
    seed_1, seed_2, seed_3, seed_4,seed_5 = random_seed,seeds[0],seeds[1],seeds[2],seeds[3]
    
    instance_conf['basis_func_conf']['basis_func_random_state']     = seed_1     
    instance_conf['LSM_conf']['train_random_seed']                  = None    # LSMN does not need inner samples
    instance_conf['LSM_conf']['test_random_seed']                   = None    # LSMN does not need inner samples
    instance_conf['mdp_conf']['inner_sample_seed']                  = None    # LSMN does not need inner samples
    instance_conf['mdp_conf']['inner_sample_size']                  = IR_inner_sample_size # It will be used in information relaxtion
    instance_conf['mdp_conf']['num_CFA_sample_path']                = CFA_num_train_sample_path
    instance_conf['mdp_conf']['num_pol_eval_sample_path']           = CFA_num_test_sample_path
    
    instance_conf['IR_conf'] = {}
    instance_conf['IR_conf']['inner_sample_seed']                   = seed_5

    # Update instance_conf
    instance_conf['LSM_conf']['CFA_num_train_sample_path']          = CFA_num_train_sample_path
    instance_conf['LSM_conf']['CFA_num_test_sample_path']           = CFA_num_test_sample_path  

           
    
    instance_conf['basis_func_conf']['basis_func_type']             = 'DFM'
    instance_conf['basis_func_conf']['basis_func']                  = BerOptBasisFunction  
    instance_conf['basis_func_conf']['bandwidth']                   = bandwidth
    instance_conf['basis_func_conf']['batch_size']                  = instance_conf['mdp_conf']['dim_state']+1
    instance_conf['basis_func_conf']['max_basis_num']               = instance_conf['mdp_conf']['dim_state']+1    
    instance_conf['mdp_conf']['state_relevance_type']               = 'init'
 
    LSM = LeastSquaresMonteCarlo(instance_conf)
    LSM.LSMN()


    
    