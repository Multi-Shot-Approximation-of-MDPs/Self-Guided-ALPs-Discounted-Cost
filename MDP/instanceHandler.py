# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
import importlib
from os import path,mkdir

def make_instance(mdp_name,instance_number,trial=None):
    assert isinstance(mdp_name,str), 'Argument of wrong type!'
    assert isinstance(instance_number,str), 'Argument of wrong type!'
    
    instance_path  = 'MDP.'+mdp_name+'.Instances.instance_' + instance_number 
    instance_setup = None
    
    if not path.exists('MDP/'+mdp_name+'/Instances/instance_'+instance_number+'.py'):
        raise Exception('Instance ' + mdp_name +  ' does not exist.')
    
    if mdp_name in ['PIC','BerOpt']:
        instance_setup = importlib.import_module(instance_path).get_experiment_setup()
    elif mdp_name in ['GJR']:
        instance_setup = importlib.import_module(instance_path).get_experiment_setup(trial)    
    else:
        raise Exception('Class of type (' + mdp_name +') is not implemented!')
    
    if mdp_name == 'BerOpt':
        assert type(trial) == int
    
        if not path.exists('Output/'+mdp_name+'/instance_'+instance_number):
            mkdir('Output/'+mdp_name+'/instance_'+instance_number)
        
        path_list  = ['Output/'+mdp_name+'/instance_'+instance_number+'/seed_'+str(trial),
                      'Output/'+mdp_name+'/instance_'+instance_number+'/seed_'+str(trial)+'/basis_params'
                      ]
        
        for _ in  path_list:
            if not path.exists(_):
                mkdir(_)
    
    if instance_setup is None:
        raise Exception('Instance ' + mdp_name + 'is not loaded properly!')
    
    else:
        return instance_setup
    
    
    


    
    
    

