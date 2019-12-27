"""
  #############################################################################
  # Created: Mar 27, 2019
  #          Parshan Pakiman | http://business.uic.edu/faculty/parshan-pakiman
  #
  # File:    fourierBasisFunctions.py
  # ------
  # Licensing Information:  
  #
  #
  # Attribution Information: 
  #
  #
  #############################################################################
"""


from BasisFunction.basisFunctions import BasisFunctions
from scipy.stats import randint
import numpy as np

class HatBasis(BasisFunctions):
    
    def __init__(self, BF_Setup):        
        super().__init__(BF_Setup)
        self.breakPoints    = BF_Setup['breakPoints']
        self.ridgeVector    = BF_Setup['ridgeVector']
        self.numRidgeVec    = BF_Setup['numRidgeVec']
        self.numBreakPts    = BF_Setup['numBreakPts']
        self.optCoef        = [None for _ in range(self.numRidgeVec)]
        self.indexUnitVec   = BF_Setup['indexUnitVec']
        self.indexUnitVec   = BF_Setup['indexPairVec']
            
    def cleanUp(self):
        self.breakPoints    = [None for _ in range(self.numRidgeVec)]
        self.ridgeVector    = [None for _ in range(self.numRidgeVec)]
        self.optCoef        = [None for _ in range(self.numRidgeVec)]
        self.numRidgeVec    = 0
        self.numBreaPts     = 0

        
    def reinit(self, BF_Setup):
        super().__init__(BF_Setup)
        self.breakPoints    = [None for _ in range(self.numRidgeVec)]
        self.ridgeVector    = [None for _ in range(self.numRidgeVec)]
        self.optCoef        = [None for _ in range(self.numRidgeVec)]
        self.numRidgeVec    = 0
        self.numBreaPts     = 0
        
    def setSampledParms(self,breakPointsList,ridgeVectorList):
        for _ in range(self.BF_number):
            self.ridgeVector[_] = ridgeVectorList[_]
            self.breakPoints[_] = breakPointsList[_]

    def hatFunction(self,B_l,B_c,B_r,x):
        z = 0.0
        if   x >= B_l and x <= B_c:
            z= (x-B_l)/(B_c-B_l)
            
        elif x >= B_c and x <= B_r:
            z = (B_r - x)/(B_r-B_c)
               
        return z
    
    
    def whereLocated(self,B_l,B_c,B_r,x):
        if x < B_l:
            return 0
        elif x>= B_l and x< B_c:
            return 1
        elif x>= B_c and x<= B_r:
            return 2
        else:
            return 3
        
    
    def deltaHat(self,j,i,X,Y):
        
        B_l = self.breakPoints[j][i]
        B_c = self.breakPoints[j][i+1]
        B_r = self.breakPoints[j][i+2]
        
        x = np.dot(self.ridgeVector[j],X)
        y = np.dot(self.ridgeVector[j],Y)
        
        
        z = 0.0
        
        I =  self.whereLocated(B_l,B_c,B_r,x)
        J =  self.whereLocated(B_l,B_c,B_r,y)
        
        if I == 0 and J == 0:
            z = 0.0
        
        elif I == 0 and J == 1:
            z =  (B_l - y)/(B_c-B_l)
        
        elif I == 0 and J == 2:
            z =  (y - B_r)/(B_r-B_c)

        elif I == 0 and J == 3:
            z =  0.0
            
        elif I == 1 and J == 0:
            z = (x-B_l)/(B_c-B_l)
        
        elif I == 1 and J == 1:
            z = (x-y)/(B_c-B_l) 
        
        elif I == 1 and J == 2:
            z = (x-B_l)/(B_c-B_l) + (y - B_r)/(B_r-B_c)
        
        elif I == 1 and J == 3:
            z = (x-B_l)/(B_c-B_l) 

        elif I == 2 and J == 0:
            z = (B_r - x)/(B_r-B_c)
        
        elif I == 2 and J == 1:
            z = (B_r - x)/(B_r-B_c) + (B_l - y)/(B_c-B_l)
        
        elif I == 2 and J == 2:
            z = (y - x)/(B_r-B_c) 
        
        elif I == 2 and J == 3:
            z = (B_r - x)/(B_r-B_c)
        
        elif I == 3 and J == 0:
            z = 0.0
        
        elif I == 3 and J == 1:
            z = (B_l-y)/(B_c-B_l)
        
        elif  I ==3 and J == 2:
            z =  (y - B_r)/(B_r-B_c)
        
        else:
            z = 0.0
        
        
#        if abs(z) < 1e-4:
#            z = 0.0
#            print('=================================================> ',z,I,J)
        return z
    

    def getHatVal(self,j,i,state):
        x = np.dot(self.ridgeVector[j],state)
        return self.hatFunction(self.breakPoints[j][i],self.breakPoints[j][i+1],self.breakPoints[j][i+2],x)        

    
    def getVFA(self, state):
        hat = self.hatFunction
        val = 0
        for j in range(self.BF_number):
            x = np.dot(state,self.ridgeVector[j])
            B = self.breakPoints[j]
            weight = self.optCoef[j]
            for i in range(len(B)):
                val+= weight[i]*hat(B[i],B[i+1],B[i+2],x)
                
        return val





























        
        