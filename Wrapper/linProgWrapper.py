# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""




class lin_prog_wrapper:
    
    def __init__(self,solver_conf):
        self.solver_conf = solver_conf
        
    def set_up_variables(self,num_basis):
        pass

    def add_FALP_constraint(self,ALP_constr_matrix,ALP_RHS):
        pass

    def set_objective(self,obj_coef,is_maximum = True):
        pass

    def prepare(self):
        pass

    def optimize(self):
        pass
    
    def get_num_var_constr(self):
        pass
    
    def get_optimal_value(self):
        pass
    
    def get_optimal_solution(self):
        pass
        
    def get_status(self):
        pass
    
    def incorporate_self_guiding_constraint(self):
        pass
    