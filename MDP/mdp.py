"""
###############################################################################
# Created: Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
#                           | http://business.uic.edu/faculty/parshan-pakiman
#                          
# Licensing Information: The MIT License
###############################################################################
"""

"""
    Class framework for general MDPs
"""
class MarkovDecisionProcess:
    
    """
        Constructing an MDP object
    """
    def __init__(self, mdpSetup):
    #--------------------------------------------------------------------------
      # MDP setting
      self.mdpName      = mdpSetup['mdp_name']  
      self.dimX         = mdpSetup['dimX']      # Dimension of state
      self.dimU         = mdpSetup['dimU']      # Dimension of action
      self.discount     = mdpSetup['discount']  # Discount  factor
      
      #------------------------------------------------------------------------
      # Constraint sampling information
      try:
        self.CS_numStates = mdpSetup['CS_numStates'] 
        self.CS_numAction = mdpSetup['CS_numAction'] 
      
      except: 
        self.CS_numStates =    None
        self.CS_numAction =    None
     
      #------------------------------------------------------------------------
      # Sample avergae approximation: number of samples in an expectations;
      # By what ration split the action set if discretization is needed.
      self.SA_numSamples    = mdpSetup['SA_numSamples'] 
      self.actionDiscFactor = mdpSetup['actionDiscFactor']
  
    """ 
        Generates a batch of samples from randomness in the MDP
    """
    def getBatchInitStateDistribution(self):
        pass

    """ 
        Given a state, generate all feasible actions can be taken.
    """
    def getFeasibleActions(self, curState):
        pass

    """ 
        Given state and action, generate a batch of sampled next state.
    """
    def getBatchOfNextStates(self, curState, curAction):
        pass
    
    """ 
        Given state and action, it generates the expected immediate cost.
    """
    def getExpectedCost(self, curState, curAction, listExoInfo=None):
        pass

    """ 
        Sample state-action pairs.
    """    
    def constSamplerForALP(self):
        pass
        
    """ 
        Sample & fix a set of realized sampled from the MDP exogenous random variable.
    """        
    def setBatchOfExoInfo(self):
        pass

    """ 
        From a file, load sampled exogenous samples, e.g., demand, price, ...
    """       
    def readBatchOfExoInfo(self,fileName):
        pass

    """ 
        Given an object from basis function class, compute the expected cost of
        greedy policy w.r.t. the VFA defined by the input basis function object.        
    """     
    def getExpectedCostByExecPolicyFromVFA(self,BF):
        pass    
    
    """
        Rounding function used in practice with 1-D input.
    """
    def myRound(self,x):
        return float(int(x/self.actionDiscFactor)*self.actionDiscFactor)

    """
        Rounding function that rounds all coordinates of a state.
    """    
    def roundState(self,state):
        for _ in range(self.dimX):
            state[_] = self.myRound(state[_])
        return state
    
    
    
    
    
    
    