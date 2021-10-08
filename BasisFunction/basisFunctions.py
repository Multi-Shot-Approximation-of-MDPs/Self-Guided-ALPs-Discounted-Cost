# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""

from numpy import ndarray


class BasisFunctions:

    def __init__(self, basis_setup):
        
        #--------------------------------------------------------------------------
        #    Initialize the dimension of state space, the number of bases, and 
        #    optimal bases coefficients.
        #--------------------------------------------------------------------------
        self.basis_func_type         = basis_setup['basis_func_conf']['basis_func_type'] 
        self.dim_state               = basis_setup['basis_func_conf']['dim_state']
        self.max_basis_num           = basis_setup['basis_func_conf']['max_basis_num']
        self.opt_coef                = None
        self.basis_func_random_state = basis_setup['basis_func_conf']['basis_func_random_state'] 
        self.num_basis_func          = 0
    

    def eval_basis(self,state):
        #--------------------------------------------------------------------------
        # Should return a vector with the evaluation of each basis on a given state.
        #--------------------------------------------------------------------------
        pass


    def set_optimal_coef(self, opt_coef):
        #--------------------------------------------------------------------------
        # Setter for optimal coefficients, e.g., set weights of basis functions to
        # ALP optimal solution.
        #--------------------------------------------------------------------------
        assert isinstance(opt_coef,ndarray)
        self.opt_coef = opt_coef


    def get_VFA(self,state):
        #--------------------------------------------------------------------------
        # Given optimal weights, computes VFA at a state.
        #--------------------------------------------------------------------------
        pass
    
    def get_expected_VFA(self,state_list):
        #--------------------------------------------------------------------------
        # This function computes average of VFAs at a list of states.
        #--------------------------------------------------------------------------
        pass 
    """
        Concats the two sets of coeffcents 
    """
    def concat(self,basis_coef_1,basis_coef_2):
        #----------------------------------------------------------------------
        # Concat parameters of random bases with parameters 
        # basis_coef_1 nd basis_coef_2
        #----------------------------------------------------------------------
        pass
    
