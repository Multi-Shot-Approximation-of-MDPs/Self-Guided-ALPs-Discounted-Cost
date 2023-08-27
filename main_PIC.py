# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
import numpy as np
import sys
from Algorithm.selfGuidedALPs import Self_Guided_ALP
from MDP.instanceHandler import make_instance
from scipy.stats import truncnorm
from utils import output_handler, is_PIC_config_valid
from BasisFunction.fourierBasisFunctions import FourierBasis
from BasisFunction.reluBasisFunctions import ReLUBasis
from BasisFunction.stumpBasisFunctions import StumpBasis
from BasisFunction.lnsBasis import LNS_BasisFunction


if __name__== "__main__":
    
    mdp_name                        = str(sys.argv[1]) 
    algo_name                       = str(sys.argv[2]) 
    basis_func_type                 = str(sys.argv[3])
    instance_number                 = str(sys.argv[4])
    max_num_constr                  = int(sys.argv[5])
    max_basis_num                   = int(sys.argv[6])
    batch_size                      = int(sys.argv[7])
    num_cpu_core                    = int(sys.argv[8])
    basis_func_random_state         = int(sys.argv[9])
    state_relevance_inner_itr       = int(sys.argv[10])
    update_constr_via_greedy_pol    = False #Adjusting this parameter is possible if the sampled constraint needs updating based on the states visited under a greedy policy.

    
    instance_conf                   = make_instance(mdp_name,instance_number)
    
    if basis_func_type == 'fourier':
        bandwidth                   = [1e-3, 1e-4]
        basis_func                  = FourierBasis
        
    elif basis_func_type == 'relu': 
        basis_func                  = ReLUBasis
        bandwidth                   = instance_conf['mdp_conf']['max_order']

    elif basis_func_type == 'stump': 
        basis_func                  = StumpBasis
        bandwidth                   = [instance_conf['mdp_conf']['max_order'] for _ in range(instance_conf['mdp_conf']['dim_state'] )]

    elif basis_func_type == 'lns':
        basis_func                  = LNS_BasisFunction
        
        # For LNS, we pass statistic of demand distribution as the 
        # "bandwidth" parameter of the basis functions.
        demand_dist                 = truncnorm(a       = (instance_conf['mdp_noise_conf']['dist_min'] - instance_conf['mdp_noise_conf']['dist_mean'])/instance_conf['mdp_noise_conf']['dist_std'],
                                                b       = (instance_conf['mdp_noise_conf']['dist_max'] - instance_conf['mdp_noise_conf']['dist_mean'])/instance_conf['mdp_noise_conf']['dist_std'],
                                                loc     = instance_conf['mdp_noise_conf']['dist_mean'],
                                                scale   = instance_conf['mdp_noise_conf']['dist_std'])
        
        # print(demand_dist.ppf(.25), , demand_dist.ppf(.5))
        
        bandwidth                   = [demand_dist.ppf(.25), demand_dist.mean(), demand_dist.ppf(.75)]
        max_basis_num               = (2*len(bandwidth) + 1) * instance_conf['mdp_conf']['dim_state'] + 1 - len(bandwidth)
        batch_size                  = (2*len(bandwidth) + 1) * instance_conf['mdp_conf']['dim_state'] + 1 - len(bandwidth)
        
    else:
        raise Exception('Basis function type is off.')
    
    """ 
        ----------------------------------------------------------
        Computing lower bound via FALP with uniform state relvance
        ----------------------------------------------------------
    """
    instance_conf['misc_conf']['num_cpu_core']          = num_cpu_core
    instance_conf['greedy_pol_conf']['num_cpu_core']    = num_cpu_core
    instance_conf['lower_bound_conf']['num_cpu_core']   = num_cpu_core
    instance_conf['solver']['num_cpu_core']             = num_cpu_core

    """ 
        --------------
        FALP Algorithm
        --------------
    """
    if algo_name=='FALP':
        num_basis_to_update_pol_cost                                            = [max_basis_num]  
        instance_conf['mdp_sampler_conf']['state_relevance']                    = instance_conf['mdp_sampler_conf']['state_sampler']
        instance_conf['mdp_sampler_conf']['state_relevance_name']               = 'uniform_non_adaptive'
        instance_conf['greedy_pol_conf']['update_state_rel_via_greedy_pol']     = False
        instance_conf['greedy_pol_conf']['update_constr_via_greedy_pol']        = update_constr_via_greedy_pol
        instance_conf['greedy_pol_conf']['state_relevance_inner_itr']           = 0
        
        instance_conf['basis_func_conf']['basis_func_random_state']             = basis_func_random_state
        instance_conf['basis_func_conf']['basis_func_type']                     = basis_func_type
        instance_conf['basis_func_conf']['basis_func']                          = basis_func
        instance_conf['basis_func_conf']['max_basis_num']                       = max_basis_num
        instance_conf['basis_func_conf']['batch_size']                          = max_basis_num
        instance_conf['basis_func_conf']['bandwidth']                           = bandwidth
        
        
        instance_conf['constr_conf']['max_num_constr']                          = max_num_constr		
        instance_conf['constr_conf']['constr_gen_batch_size']                   = max_num_constr
        instance_conf['greedy_pol_conf']['num_basis_to_update_pol_cost']        = num_basis_to_update_pol_cost
        is_PIC_config_valid(instance_conf)
        output_handler(instance_conf)               
        
        ADP = Self_Guided_ALP(instance_conf)
        ADP.FALP()
        del ADP       


    """ 
    -----------------
    Policy-guided  FALP 
    ----------------- 
    """  
    if algo_name=='PG-FALP':
        
        num_basis_to_update_pol_cost    = np.arange(batch_size,max_basis_num+batch_size,batch_size)
        
        instance_conf['mdp_sampler_conf']['state_relevance']                    = instance_conf['mdp_sampler_conf']['state_sampler']
        instance_conf['mdp_sampler_conf']['state_relevance_name']               = 'uniform_adaptive'
        instance_conf['greedy_pol_conf']['update_state_rel_via_greedy_pol']     = True
        instance_conf['greedy_pol_conf']['update_constr_via_greedy_pol']        = update_constr_via_greedy_pol
        instance_conf['greedy_pol_conf']['state_relevance_inner_itr']           = state_relevance_inner_itr
        instance_conf['basis_func_conf']['basis_func_random_state']             = basis_func_random_state
        instance_conf['basis_func_conf']['basis_func_type']                     = basis_func_type
        instance_conf['basis_func_conf']['basis_func']                          = basis_func
        instance_conf['basis_func_conf']['max_basis_num']                       = max_basis_num
        instance_conf['basis_func_conf']['batch_size']                          = max_basis_num
        instance_conf['basis_func_conf']['bandwidth']                           = bandwidth
        
        
        instance_conf['constr_conf']['max_num_constr']                          = max_num_constr		
        instance_conf['constr_conf']['constr_gen_batch_size']                   = max_num_constr
        instance_conf['greedy_pol_conf']['num_basis_to_update_pol_cost']        = num_basis_to_update_pol_cost
        
        is_PIC_config_valid(instance_conf)
        output_handler(instance_conf)               
        
        ADP = Self_Guided_ALP(instance_conf)
        ADP.FALP()
        del ADP   


    """ 
    -----------------
    Self-guided FALP
    ----------------- 
    """  
    if algo_name=='SG-FALP':
    
        num_basis_to_update_pol_cost    = np.arange(batch_size,max_basis_num+batch_size,batch_size)
        instance_conf                                                           = make_instance(mdp_name,instance_number)
        instance_conf['basis_func_conf']['basis_func_random_state']             = basis_func_random_state
        instance_conf['mdp_sampler_conf']['state_relevance']                    = instance_conf['mdp_sampler_conf']['state_sampler']
        instance_conf['mdp_sampler_conf']['state_relevance_name']               = 'uniform_non_adaptive'
        instance_conf['greedy_pol_conf']['update_state_rel_via_greedy_pol']     = False
        instance_conf['greedy_pol_conf']['update_constr_via_greedy_pol']        = update_constr_via_greedy_pol
        instance_conf['greedy_pol_conf']['state_relevance_inner_itr']           = 0
        
        instance_conf['basis_func_conf']['basis_func_random_state']             = basis_func_random_state
        instance_conf['basis_func_conf']['basis_func_type']                     = basis_func_type
        instance_conf['basis_func_conf']['basis_func']                          = basis_func
        instance_conf['basis_func_conf']['max_basis_num']                       = max_basis_num
        instance_conf['basis_func_conf']['batch_size']                          = batch_size
        instance_conf['basis_func_conf']['bandwidth']                           = bandwidth
        
        instance_conf['constr_conf']['max_num_constr']                          = max_num_constr		
        instance_conf['constr_conf']['constr_gen_batch_size']                   = max_num_constr
        instance_conf['greedy_pol_conf']['num_basis_to_update_pol_cost']        = num_basis_to_update_pol_cost
        is_PIC_config_valid(instance_conf)
        output_handler(instance_conf)               
        
        ADP = Self_Guided_ALP(instance_conf)
        ADP.SGFALP()
        del ADP

