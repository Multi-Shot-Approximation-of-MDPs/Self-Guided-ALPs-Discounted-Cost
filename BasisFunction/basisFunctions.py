"""
###############################################################################
# Created: Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
#                           | http://business.uic.edu/faculty/parshan-pakiman
#                          
# Licensing Information: The MIT License
###############################################################################
"""

"""
    Class framework for general basis functions
"""
class BasisFunctions:
    
    #--------------------------------------------------------------------------
    # Initialize the dimension of state space, the number of bases, and optimal
    # bases coefficients.
    def __init__(self, BF_Setup):
        self.dimX = BF_Setup['dimX']
        self.BF_number = BF_Setup['BF_num']
        self.optCoef = None
    
    #--------------------------------------------------------------------------
    # Return a vector that has the evaluation of each basis on a given state.
    def evalBasisList(self,state):
        pass

    #--------------------------------------------------------------------------
    # Setter for optimal coefficients, e.g., set weights of basis functions to
    # ALP optimal solution.
    def setOptimalCoef(self, opt_coef):
        self.optCoef = opt_coef
    
    #--------------------------------------------------------------------------
    # Given optimal weights, computes VFA at a state.
    def getVFA(self,state):
        pass