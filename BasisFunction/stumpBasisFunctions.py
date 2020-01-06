"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from BasisFunction.basisFunctions import BasisFunctions
from scipy.stats import randint
import numpy as np

"""
    Class of random stump bases
"""
class StumpBasis(BasisFunctions):
        
    """
       Given basis functions configuration (BF_Setup), initialize parameters of
       basis functions.
    """
    def __init__(self, BF_Setup):    
        #----------------------------------------------------------------------
        # Call supper class constructor
        super().__init__(BF_Setup)
        
        #----------------------------------------------------------------------
        # Bandwidth of sampling distribution
        self.isStationary   = BF_Setup['isStationary']
        self.bandwidth      = BF_Setup['bandwidth']
        self.bandWidth_LB   = BF_Setup['bandWidth_LB']
        self.bandWidth_UB   = BF_Setup['bandWidth_UB']
        self.margin         = BF_Setup['Margin']
        
        #----------------------------------------------------------------------
        # Consider phi(s;(threshold,index)) = sign (s_{index} -threshold). The 
        # following lists track the values of samples (index, threshold). For 
        # each index, we associate a random unit vector that has zero in all of
        # its coordinates instead of coordinate "index"
        self.threshold_list = None
        self.index_list     = None
        self.rndUnitVect    = None
        
    """
        Remove weights of basis functions.
    """
    def cleanUp(self):
        self.threshold_list = None
        self.index_list = None
        self.optCoef = None
        self.rndUnitVect    = None
        
    """
        Call constructor again!
    """    
    def reinit(self, BF_Setup):
        super().__init__(BF_Setup)
        self.isStationary   = BF_Setup['isStationary']
        self.bandwidth      = BF_Setup['BF_disStd']
        self.bandWidth_LB   = BF_Setup['bandWidth_LB']
        self.bandWidth_UB   = BF_Setup['bandWidth_UB']
        self.threshold_list = None
        self.index_list     = None
        self.rndUnitVect    = None
    
    """
        A piecewise linear approximation of sign function that is continuous 
        at zero.
    """          
    def softSign(self,q):        
        if   q>=self.margin:
            return 1.0
        elif q<= -self.margin:
            return -1.0
        else:
            return float(q/self.margin)
        
    """
        Getter for samples of random bases.
    """      
    def getSample(self):
       
        index     = randint(0,self.dimX).rvs(self.BF_number)
       
        if self.isStationary:
            threshold = np.asarray([np.random.uniform(low=-self.bandwidth,high=self.bandwidth)
                                        for _ in range(self.BF_number)])
   
            return  threshold, index
         
        else:
            k = int(np.ceil(self.BF_number/self.dimX))
            index       = np.zeros(self.BF_number,dtype=np.int)
            threshold   = np.zeros(self.BF_number)
            l = 0
            for i in range(0,k):    
                for j in range(self.dimX) :
                    if l < self.BF_number:
                        index[l]     = int(j)
                        threshold[l] = round(np.random.uniform(low=min(self.bandWidth_LB),high=max(self.bandWidth_UB)),5)
                        l+=1

                    else:
                        break
        return  threshold,  index    
             
           

    def setSampledParms(self,index,threshold):
       
        self.index_list  = index
        self.threshold_list   = threshold
       
        self.rndUnitVect = np.asarray([np.eye(1,self.dimX,self.index_list[i]).flatten()
                                              for i in range(self.BF_number)],dtype=float)
       

    def setRandBasisCoefList(self):
          self.threshold_list, self.index_list = self.getSample()
          self.rndUnitVect = np.asarray([np.eye(1,self.dimX,self.index_list[i]).flatten()
                                              for i in range(self.BF_number)],dtype=float)

       
    def evalBasisList(self, state):
        IDX=self.index_list
        THR=self.threshold_list
        M  =self.margin
        def localSign (idx):
            q = state[IDX[idx]]
            if   q - THR[idx] >=  M:
                return float(1.0)
            elif q - THR[idx] <= -M:
                return float(-1.0)
            else:
                return (q-THR[idx])/M
       
        return np.asarray([localSign(_) for _ in range(self.BF_number)]) 
