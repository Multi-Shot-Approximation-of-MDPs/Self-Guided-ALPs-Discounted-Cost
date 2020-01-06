"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
import numpy as np
from BasisFunction.basisFunctions import BasisFunctions


"""
    The following class implements the hat basis functions introduced in [1]
"""
class HatBasis(BasisFunctions):
   
    """
       Given basis functions configuration (BF_Setup), initialize parameters of
       basis functions.
    """
    def __init__(self, BF_Setup):  
        #----------------------------------------------------------------------
        # Call supper class constructor
        super().__init__(BF_Setup)
        
        #----------------------------------------------------------------------
        # Configuring hat bases parameters
        self.breakPoints    = BF_Setup['breakPoints']
        self.ridgeVector    = BF_Setup['ridgeVector']
        self.numRidgeVec    = BF_Setup['numRidgeVec']
        self.numBreakPts    = BF_Setup['numBreakPts']
        self.optCoef        = [None for _ in range(self.numRidgeVec)]
        self.indexUnitVec   = BF_Setup['indexUnitVec']
        self.indexUnitVec   = BF_Setup['indexPairVec']

    """
       Eliminating parameters of the hat basis functions
    """            
    def cleanUp(self):
        self.breakPoints    = [None for _ in range(self.numRidgeVec)]
        self.ridgeVector    = [None for _ in range(self.numRidgeVec)]
        self.optCoef        = [None for _ in range(self.numRidgeVec)]
        self.numRidgeVec    = 0
        self.numBreaPts     = 0

    """
       Redefining the hat basis functions
    """          
    def reinit(self, BF_Setup):
        #----------------------------------------------------------------------
        # Call supper class constructor
        super().__init__(BF_Setup)
        
        #----------------------------------------------------------------------
        # Configuring hat bases parameters
        self.breakPoints    = [None for _ in range(self.numRidgeVec)]
        self.ridgeVector    = [None for _ in range(self.numRidgeVec)]
        self.optCoef        = [None for _ in range(self.numRidgeVec)]
        self.numRidgeVec    = 0
        self.numBreaPts     = 0

    """
       Redefining the hat basis functions
    """         
    def setSampledParms(self,breakPointsList,ridgeVectorList):
        #----------------------------------------------------------------------
        # Setting the ridge vectors and breakpoints
        for _ in range(self.BF_number):
            self.ridgeVector[_] = ridgeVectorList[_]
            self.breakPoints[_] = breakPointsList[_]

    """
       A hat basis functions defined on range [B_l,B_c] & [B_c,B_r]
    """   
    def hatFunction(self,B_l,B_c,B_r,x):
        #----------------------------------------------------------------------
        # Value of the hat basis outside [B_l,B_r] is zero
        value = 0.0
        
        #----------------------------------------------------------------------
        # If x in [B_l,B_c], then set the value as a linear function of x
        if   x >= B_l and x <= B_c:
            value = (x-B_l)/(B_c-B_l)
            
        #----------------------------------------------------------------------
        # If x in [B_c,B_r], then set the value as a linear function of x    
        elif x >= B_c and x <= B_r:
            value = (B_r - x)/(B_r-B_c)
        #----------------------------------------------------------------------
        # Return the value
        return value
    
    """
       Check if x in (-inf,B_l], in [B_l,B_c], in [B_c,B_r], or in [B_r,inf)
    """      
    def whereLocated(self,B_l,B_c,B_r,x):
        if x < B_l:
            return 0
        elif x>= B_l and x< B_c:
            return 1
        elif x>= B_c and x<= B_r:
            return 2
        else:
            return 3
        
    """
       This function computes the of j-th hat function on two points X and Y
    """    
    def deltaHat(self,j,i,X,Y):
        #----------------------------------------------------------------------
        # Retrieving B_l, B_c, and B_r
        B_l = self.breakPoints[j][i]
        B_c = self.breakPoints[j][i+1]
        B_r = self.breakPoints[j][i+2]
        
        #----------------------------------------------------------------------
        # Computing inner product of X and Y with the ridge vectors
        x = np.dot(self.ridgeVector[j],X)
        y = np.dot(self.ridgeVector[j],Y)
        
        #----------------------------------------------------------------------
        # Set the value of hat function on x and y
        valDeltaHatFunction     = 0.0
        
        #----------------------------------------------------------------------
        # Set up location of x and y in the intervals defining the hat function
        I =  self.whereLocated(B_l,B_c,B_r,x)
        J =  self.whereLocated(B_l,B_c,B_r,y)
        
        #----------------------------------------------------------------------
        # Based on the location of x and y, define valDeltaHatFunction
        if I == 0 and J == 0:
            valDeltaHatFunction = 0.0
        
        elif I == 0 and J == 1:
            valDeltaHatFunction = (B_l - y)/(B_c-B_l)
        
        elif I == 0 and J == 2:
            valDeltaHatFunction = (y - B_r)/(B_r-B_c)

        elif I == 0 and J == 3:
            valDeltaHatFunction = 0.0
            
        elif I == 1 and J == 0:
            valDeltaHatFunction = (x-B_l)/(B_c-B_l)
        
        elif I == 1 and J == 1:
            valDeltaHatFunction = (x-y)/(B_c-B_l) 
        
        elif I == 1 and J == 2:
            valDeltaHatFunction = (x-B_l)/(B_c-B_l) + (y - B_r)/(B_r-B_c)
        
        elif I == 1 and J == 3:
            valDeltaHatFunction = (x-B_l)/(B_c-B_l) 

        elif I == 2 and J == 0:
            valDeltaHatFunction = (B_r - x)/(B_r-B_c)
        
        elif I == 2 and J == 1:
            valDeltaHatFunction = (B_r - x)/(B_r-B_c) + (B_l - y)/(B_c-B_l)
        
        elif I == 2 and J == 2:
            valDeltaHatFunction = (y - x)/(B_r-B_c) 
        
        elif I == 2 and J == 3:
            valDeltaHatFunction = (B_r - x)/(B_r-B_c)
        
        elif I == 3 and J == 0:
            valDeltaHatFunction = 0.0
        
        elif I == 3 and J == 1:
            valDeltaHatFunction = (B_l-y)/(B_c-B_l)
        
        elif  I ==3 and J == 2:
            valDeltaHatFunction = (y - B_r)/(B_r-B_c)
        
        else:
            valDeltaHatFunction = 0.0
        
        #----------------------------------------------------------------------
        # Return the value of the difference of hat function on x and y
        return valDeltaHatFunction
    
    """
       Compute the value of a hat basis defined on (j, i) on state state
    """    
    def getHatVal(self,j,i,state):
        x = np.dot(self.ridgeVector[j],state)
        return self.hatFunction(self.breakPoints[j][i],self.breakPoints[j][i+1],self.breakPoints[j][i+2],x)        

    """
       Given optimal weights, computes VFA at a state.
    """      
    def getVFA(self, state):
        #----------------------------------------------------------------------
        # Set up the hat basis
        hat             = self.hatFunction
        breakPoints     = self.breakPoints
        dot             = np.dot
        val             = 0
        optCoef         = self.optCoef
        
        #----------------------------------------------------------------------
        # Compute the value of the linear combination of bases on state
        for j in range(self.BF_number):
            x = dot(state,self.ridgeVector[j])
            B = breakPoints[j]
            weight = optCoef[j]
            for i in range(len(B)):
                val+= weight[i]*hat(B[i],B[i+1],B[i+2],x)
                
        #----------------------------------------------------------------------  
        # Return the value
        return val





























        
        