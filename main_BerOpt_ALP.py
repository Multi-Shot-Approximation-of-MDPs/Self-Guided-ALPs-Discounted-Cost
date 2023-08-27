# -*- coding: utf-8 -*-clear


"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""

from MDP.instanceHandler import make_instance
from Algorithm.finiteTimeSelfGuidedALPs import SelfGuidedALPs
from BasisFunction.fourierBasisFunctionsForBerOpt import FourierBasisForBerOpt
from BasisFunction.berOptSpecificBasisFunction import BerOptBasisFunction
from BasisFunction.reluBasisFunctionsForBerOpt import ReLUBasis
from utils import output_handler_option_pricing,output_handler_option_pricing,aggregate_all_algorithm_output_BerOpt
import sys
import numpy as np

if __name__== "__main__":

    mdp_name                        = 'BerOpt'
    algo_name                       = str(sys.argv[1])
    instance_number                 = str(sys.argv[2])
    random_seed                     = int(sys.argv[3])
    VFA_num_train_sample_path       = int(sys.argv[4])
    VFA_num_test_sample_path        = int(sys.argv[5])
    inner_sample_size               = int(sys.argv[6])
    state_relevance_type            = str(sys.argv[7])
    abs_val_upp_bound               = float(sys.argv[8])   
    compute_IR_bound                = True if str(sys.argv[9]) == 'True' else False
    basis_func_type                 = str(sys.argv[10])  
    max_basis_num                   = int(sys.argv[11])  
    batch_size                      = int(sys.argv[12])
    num_cpu_core                    = int(sys.argv[13])  
    preprocess_batch                = int(sys.argv[14])  


    instance_conf                   = make_instance(mdp_name,instance_number,trial=random_seed)
    
 
    if basis_func_type == 'DFM':
        basis_func                  = BerOptBasisFunction
        batch_size                  = instance_conf['mdp_conf']['dim_state']+1
        max_basis_num               = instance_conf['mdp_conf']['dim_state']+1
        preproccess_batch           = 0
        bandwidth                   = [0]
        
    elif basis_func_type == 'fourier':
        basis_func                  = FourierBasisForBerOpt
        preprocess_batch            = preprocess_batch
        
        if instance_number in ['9', '8' , '7',]:
            bandwidth               = [1e-3,1e-4]
            abs_val_upp_bound       = .2
            
            
        if instance_number in ['1', '2' , '3',]:
            bandwidth               = [1e-3,1e-4]
            abs_val_upp_bound       = 5
            
    
    elif basis_func_type == 'relu':
        basis_func                  = ReLUBasis
        bandwidth                   = None              # It must be set after generating sample paths.
        preprocess_batch            = preprocess_batch
        
    
    
    
    
    """ Set up basis functions """
    
    instance_conf['basis_func_conf']['basis_func_type']             = basis_func_type
    instance_conf['basis_func_conf']['basis_func']                  = basis_func
    instance_conf['basis_func_conf']['bandwidth']                   = bandwidth
    instance_conf['basis_func_conf']['batch_size']                  = batch_size
    instance_conf['basis_func_conf']['max_basis_num']               = max_basis_num
    
    if basis_func_type in ['fourier','relu']:
        instance_conf['basis_func_conf']['preprocess_batch']        = preprocess_batch
            
    """ Misc.  """
    instance_conf['mdp_conf']['state_relevance_type']               = state_relevance_type
    instance_conf['mdp_conf']['num_CFA_sample_path']                = VFA_num_train_sample_path
    instance_conf['mdp_conf']['num_pol_eval_sample_path']           = VFA_num_test_sample_path
    instance_conf['mdp_conf']['num_cpu_core']                       = num_cpu_core
    instance_conf['mdp_conf']['inner_sample_size']                  = inner_sample_size
    instance_conf['solver_conf']['batch_size']                      = batch_size 
    instance_conf['solver_conf']['abs_val_upp_bound']               = abs_val_upp_bound 
        
    """ Set up random seeds """
    np.random.seed(random_seed)
    seeds = list(np.random.randint(1e1,1e3,4))
    seed_1, seed_2, seed_3, seed_4,seed_5 = random_seed,seeds[0],seeds[1],seeds[2],seeds[3]
    
    instance_conf['basis_func_conf']['basis_func_random_state']     = seed_1
    instance_conf['mdp_conf']['CFA_random_seed']                    = seed_2
    instance_conf['mdp_conf']['pol_random_seed']                    = seed_3 
    instance_conf['mdp_conf']['inner_sample_seed']                  = seed_4
    
    instance_conf['IR_conf'] = {}
    instance_conf['IR_conf']['IR_inner_sample_seed']                = seed_5
    
    
    """ -------------------------------------------------------------------
    Computing expected VFAs 
    -----------------------------------------------------------------------"""

    if algo_name == 'preprocess':
        assert basis_func_type in ['fourier', 'relu']
        Model = SelfGuidedALPs(instance_conf)
        Model.compute_expected_basis_func(random_seed)
        

    if algo_name == 'FALP':
        Model           = SelfGuidedALPs(instance_conf)
        
        if basis_func_type == 'DFM':
            Model.FALP_fixed_basis()     
        elif basis_func_type in ['fourier', 'relu']:
            Model.FALP_random_bases()
        else:
            raise Exception('Basis functions is not defined!')
    

    if algo_name == 'SG_FALP':
        assert basis_func_type in ['fourier', 'relu']
        
        Model = SelfGuidedALPs(instance_conf)
        Model.SG_FALP()
   
    
    
    
    
    
    
    
    
    
    
    


    
