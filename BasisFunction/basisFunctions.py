"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""

from numpy import ndarray

"""
    Class framework for general basis functions
"""
class BasisFunctions:
    
    """ 
        Initialize the dimension of state space, the number of bases, and optimal
        bases coefficients.
    """
    def __init__(self, basis_setup):
        self.basis_func_type         = basis_setup['basis_func_conf']['basis_func_type'] 
        self.dim_state               = basis_setup['basis_func_conf']['dim_state']
        self.max_basis_num           = basis_setup['basis_func_conf']['max_basis_num']
        self.opt_coef                = None
        self.basis_func_random_state = basis_setup['basis_func_conf']['basis_func_random_state'] 
        self.num_basis_func          = 0
    
    """
        Return a vector that has the evaluation of each basis on a given state.
    """
    def eval_basis(self,state):
        pass

    """
        Setter for optimal coefficients, e.g., set weights of basis functions to
        ALP optimal solution.
    """
    def set_optimal_coef(self, opt_coef):
        assert isinstance(opt_coef,ndarray)
        self.opt_coef = opt_coef
    
    """
        Given optimal weights, computes VFA at a state.
    """
    def get_VFA(self,state):
        pass
    
    def get_expected_VFA(self,state_list):
        pass 
    """
        Concats the two sets of coeffcents 
    """
    def concat(self,basis_coef_1,basis_coef_2):
        pass
    
