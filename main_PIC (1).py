# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from selfGuidedALPs import Self_Guided_ALP
from MDP.instanceHandler import make_instance
import numpy as np
from utils import output_handler,is_PIC_config_valid


def PIC_run_instance():
    mdp_name                        = 'PIC'
    instance_number_list            = ['1']
    basis_func_random_state         = [111,222]
    bandwidth                       = [0.001]
    max_num_constr                  = 200000
    max_basis_num                   = 50
    num_cpu_core                    = 16

    for  instance_number in instance_number_list:
        
        batch_size                      = 100
        num_basis_to_update_pol_cost    = [max_basis_num]        

        """ 
            -----------
            LOWER BOUND
                - state relevance = uniform_non_adaptive
            -----------
        """

        # instance_conf                                                           = make_instance(mdp_name,instance_number)
        # instance_conf['basis_func_conf']['basis_func_random_state']             = basis_func_random_state[0] 
        # instance_conf['mdp_sampler_conf']['state_relevance']                    = instance_conf['greedy_pol_conf']['init_state_sampler']
        # instance_conf['mdp_sampler_conf']['state_relevance_name']               = 'init_dist_non_adaptive' 
        # instance_conf['greedy_pol_conf']['update_state_rel_via_greedy_pol']     = False
        # instance_conf['misc_conf']['num_cpu_core']                              = num_cpu_core
        # instance_conf['greedy_pol_conf']['state_relevance_inner_itr']           = 0
        # instance_conf['basis_func_conf']['max_basis_num']                       = max_basis_num
        # instance_conf['basis_func_conf']['batch_size']                          = batch_size
        # instance_conf['basis_func_conf']['bandwidth']                           = bandwidth
        # instance_conf ['constr_conf']['max_num_constr']                         = max_num_constr		
        # instance_conf ['constr_conf']['constr_gen_batch_size']                  = max_num_constr
        # instance_conf['greedy_pol_conf']['num_basis_to_update_pol_cost']        = num_basis_to_update_pol_cost
        # is_PIC_config_valid(instance_conf)
        # output_handler(instance_conf)
        
        # ADP  = Self_Guided_ALP(instance_conf)
        # ADP.compute_lower_bound()
        # del ADP

        for random_seed in basis_func_random_state:
            
            batch_size                      = max_basis_num
            num_basis_to_update_pol_cost    = [max_basis_num]

            """ -----------------------------
            FALP 
            --------------------------------- """  
            # instance_conf                                                           = make_instance(mdp_name,instance_number)
            # instance_conf['basis_func_conf']['basis_func_random_state']             = random_seed
            # instance_conf['mdp_sampler_conf']['state_relevance']                    = instance_conf['mdp_sampler_conf']['state_sampler']
            # instance_conf['mdp_sampler_conf']['state_relevance_name']               = 'uniform_non_adaptive'
            # instance_conf['greedy_pol_conf']['update_state_rel_via_greedy_pol']     = False
            # instance_conf['misc_conf']['num_cpu_core']                              = num_cpu_core
            # instance_conf['greedy_pol_conf']['state_relevance_inner_itr']           = 0
            # instance_conf['basis_func_conf']['max_basis_num']                       = max_basis_num
            # instance_conf['basis_func_conf']['batch_size']                          = batch_size
            # instance_conf['basis_func_conf']['bandwidth']                           = bandwidth
            # instance_conf['constr_conf']['max_num_constr']                          = max_num_constr		
            # instance_conf['constr_conf']['constr_gen_batch_size']                   = max_num_constr
            # instance_conf['greedy_pol_conf']['num_basis_to_update_pol_cost']        = num_basis_to_update_pol_cost
            # is_PIC_config_valid(instance_conf)
            # output_handler(instance_conf)               
            
            # ADP = Self_Guided_ALP(instance_conf)
            # ADP.FALP()
            # del ADP       

            """ -----------------------------
            Policy-guided  FALP 
            --------------------------------- """  
            # num_basis_to_update_pol_cost    = np.arange(batch_size,max_basis_num+batch_size,batch_size)
            # for state_relevance_inner_itr in [3]:
            
            #     instance_conf                                                           = make_instance(mdp_name,instance_number)
            #     instance_conf['basis_func_conf']['basis_func_random_state']             = random_seed
            #     instance_conf['mdp_sampler_conf']['state_relevance']                    = instance_conf['mdp_sampler_conf']['state_sampler']
            #     instance_conf['mdp_sampler_conf']['state_relevance_name']               = 'uniform_adaptive'
            #     instance_conf['greedy_pol_conf']['update_state_rel_via_greedy_pol']     = True
            #     instance_conf['misc_conf']['num_cpu_core']                              = num_cpu_core
            #     instance_conf['greedy_pol_conf']['state_relevance_inner_itr']           = state_relevance_inner_itr
            #     instance_conf['basis_func_conf']['max_basis_num']                       = max_basis_num
            #     instance_conf['basis_func_conf']['batch_size']                          = batch_size
            #     instance_conf['basis_func_conf']['bandwidth']                           = bandwidth
            #     instance_conf['constr_conf']['max_num_constr']                          = max_num_constr		
            #     instance_conf['constr_conf']['constr_gen_batch_size']                   = max_num_constr
            #     instance_conf['greedy_pol_conf']['num_basis_to_update_pol_cost']        = num_basis_to_update_pol_cost
            #     is_PIC_config_valid(instance_conf)
            #     output_handler(instance_conf)               
                
            #     ADP = Self_Guided_ALP(instance_conf)
            #     ADP.FALP()
            #     del ADP  


            """ -----------------------------
            Self-guided FALP
            --------------------------------- """  
            batch_size                      = 25
            num_basis_to_update_pol_cost    = [max_basis_num]
            instance_conf                                                           = make_instance(mdp_name,instance_number)
            instance_conf['basis_func_conf']['basis_func_random_state']             = random_seed
            instance_conf['mdp_sampler_conf']['state_relevance']                    = instance_conf['mdp_sampler_conf']['state_sampler']
            instance_conf['mdp_sampler_conf']['state_relevance_name']               = 'uniform_non_adaptive'
            instance_conf['greedy_pol_conf']['update_state_rel_via_greedy_pol']     = False
            instance_conf['misc_conf']['num_cpu_core']                              = num_cpu_core
            instance_conf['greedy_pol_conf']['state_relevance_inner_itr']           = 0
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
                   
            
        
if __name__== "__main__":
    PIC_run_instance()




