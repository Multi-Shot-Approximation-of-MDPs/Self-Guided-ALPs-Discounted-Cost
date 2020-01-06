"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from BasisFunction.basisFunctions import BasisFunctions
from scipy.stats import multivariate_normal
from math import pi
from numpy import mean,cos,zeros,multiply,eye,transpose,array,random,zeros_like,add

"""
    Class of random  Fourier bases
"""
class FourierBasis(BasisFunctions):
    
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
        
        #----------------------------------------------------------------------
        # Initialize weights of basis functions
        self.intercept_list = None
        self.theta_list     = None
        self.thetaT         = None
        
        #----------------------------------------------------------------------
        # Setup random bases sampling distribution
        if self.isStationary:
            self.rndVarTheta    = multivariate_normal( mean=zeros(shape=self.dimX),
                                                       cov = multiply(2*self.bandwidth, eye(self.dimX)))
    """
        Remove weights of basis functions.
    """
    def cleanUp(self):
        self.intercept_list = None
        self.theta_list = None
        self.optCoef = None
        
    """
        Call constructor again!
    """       
    def reinit(self, BF_Setup):
        super().__init__(BF_Setup)
        self.isStationary   = BF_Setup['isStationary']
        self.bandwidth      = BF_Setup['BF_disStd']
        self.bandWidth_LB   = BF_Setup['bandWidth_LB']
        self.bandWidth_UB   = BF_Setup['bandWidth_UB']
        self.intercept_list = None
        self.theta_list     = None
        if self.isStationary:
            self.rndVarTheta    = multivariate_normal( mean=zeros(shape=self.dimX),
                                                       cov = multiply(2*self.bandwidth, eye(self.dimX)))
    """
        Getter for samples of random bases.
    """     
    def getSample(self):
          #--------------------------------------------------------------------
          # If bandwidth is fixed:
          if self.isStationary:
              
              #----------------------------------------------------------------
              # phi(s;(intercept,theta)) = cos(s.theta + intercept) where:
              #     1) intercept ~ unif[-pi,pi]
              #     2) intercept ~ MVN(0,bandwidth*I)
              intercept = transpose(array([random.uniform(low=-pi,high=pi) for _ in range(self.BF_number)]))
              theta     = self.rndVarTheta.rvs(self.BF_number)
              
              #----------------------------------------------------------------
              # The first basis function is intercept
              intercept[0] = 0
              theta[0] = zeros(self.dimX)
              
              #----------------------------------------------------------------
              # Return sampled parameters 
              return  intercept,theta
          
          #--------------------------------------------------------------------
          # If bandwidth is randomized:
          else:
              if self.bandWidth_UB <= self.bandWidth_LB:
                  raise Exception('Non-stationary random basis sampling LB/UB.')
              else:
                  #-------------------------------------------------------------
                  # Sample intercept in 
                  # phi(s;(intercept,theta)) = cos(s.theta + intercept) where:
                  #     1) intercept ~ unif[-pi,pi]
                  #     2) intercept ~ MVN(0,bandwidth*I)
                  self.bandwidth = 0
                  intercept      = array([random.uniform(low=-pi,high=pi)
                                          for _ in range(self.BF_number)])
                  
                  #-------------------------------------------------------------
                  # Increment bandwidth gradually, and sample parameter theta.       
                  incr  = (self.bandWidth_UB - self.bandWidth_LB)/self.BF_number
                  theta = zeros((self.BF_number,self.dimX))
                  for i in range(self.BF_number):
                    self.bandwidth = self.bandWidth_LB + i*incr
                    RV = multivariate_normal( mean=zeros(shape=self.dimX),
                                                         cov  = multiply((2*self.bandwidth), eye(self.dimX)))
                    theta[i,:]=RV.rvs(1)
              
              #----------------------------------------------------------------
              # Permute random basis functions 
              pIntercept        = zeros_like(intercept)  
              pTheta            = zeros_like(theta)  
              perm              = random.permutation(self.BF_number)
              for i in range(self.BF_number):
                  pIntercept[i] = intercept[perm[i]]
                  pTheta[i]     = theta[perm[i]]
              
              #----------------------------------------------------------------
              # Return permuted bases parameters
              pIntercept[0]     = 0
              pTheta[0]         = zeros(self.dimX)
              return pIntercept,pTheta
    
    """
        Please see the supper class.
    """
    def evalBasisList(self,state):
        return cos(self.theta_list.dot(state)+self.intercept_list)
    
    """
       Compute the expected value of random bases on a batch of states.
       *** Remark: parameters of random bases are fixed.
    """
    def expectedBasisList(self,stateList):
        return mean(cos(add(stateList@self.thetaT,self.intercept_list)),0)
    
    """
       Compute the expected value of random bases on a batch of states.
       *** Remark: parameters of random bases are fixed.
    """
    def setRandBasisCoefList(self):
          [self.intercept_list,self.theta_list] = self.getSample()
          self.thetaT                           = self.theta_list.T
          
    """
        Setter for random parameters of basis functions
    """
    def setSampledParms(self,theta,intercept):
        self.theta_list=theta
        self.thetaT = self.theta_list.T
        self.intercept_list = intercept
    
    """
        Please see the supper class.
    """    
    def getVFA(self,state):
        return cos(self.theta_list.dot(state)+self.intercept_list).dot(self.optCoef)