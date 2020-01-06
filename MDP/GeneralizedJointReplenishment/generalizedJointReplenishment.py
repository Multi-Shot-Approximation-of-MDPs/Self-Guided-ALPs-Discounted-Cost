"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from multiprocessing import Pool
from MDP.mdp import MarkovDecisionProcess
from scipy.stats import truncnorm
import numpy as np
from itertools import product


"""
    MDP class modeling generalized joint replenishment (GJR).
"""
class GeneralizedJointReplenishment(MarkovDecisionProcess):
    """
      Given MDP configuration (mdpSetup), initialize parameters of MDP class.
    """      
    def __init__(self, mdpSetup):
        #--------------------------------------------------------------------------
        # Call supper class constructor
        super().__init__(mdpSetup)
        self.minorCost          = mdpSetup['minorFixCost']
        self.majorCost          = mdpSetup['majorFixCost']
        self.consumptionRate    = mdpSetup['consumptionRate']
        self.Threads            = mdpSetup['Threads']
        self.Adrr               = mdpSetup['Adrr']
        self.invUppBounds       = mdpSetup['invUppBounds']
        self.maxOrder           = mdpSetup['maxOrder']
        
    def transTime(self,curState, curAction):
        return np.min([(curState[_]+curAction[_])/self.consumptionRate[_] for _ in range(self.dimX)])
    
    def getExpectedCost(self,curState, curAction):    
        toBeReplenished = []
        for i in range(len(curAction)):
            if not curAction[i]==0:
                toBeReplenished.append(i)
            
        # At least an item should be replnishted
        if len(toBeReplenished)==0:
            return 0
            
        return self.majorCost + sum(self.minorCost[_] for _ in toBeReplenished)

   
    def getFixCost(self,itemSubset):    

        
        if len(itemSubset)==0:
            return 0
        return self.majorCost + sum(self.minorCost[_] for _ in itemSubset)


    
    
    def getNextState(self,curState, curAction):
        invUpdate = np.asarray([(curState[_]+curAction[_])/self.consumptionRate[_] for _ in range(self.dimX)])
        stockOut = invUpdate.min()
        
        nState =np.add(np.add(curState, curAction),-self.transTime(curState, curAction)*self.consumptionRate)
           
        for _ in range(len(nState)):
            if abs(invUpdate[_] - stockOut) < 1e-4:
                nState[_] = 0.0
               
        return np.round(nState,5)