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
    MDP class modeling perishable inventory control (PIC) with partial 
    backlogging and lead time.
"""
class PerishableInventoryPartialBacklogLeadTime(MarkovDecisionProcess):

  """
      Given MDP configuration (mdpSetup), initialize parameters of MDP class.
  """      
  def __init__(self, mdpSetup):
    #--------------------------------------------------------------------------
    # Call supper class constructor
    super().__init__(mdpSetup)
    self.c_p                    = mdpSetup['purchase_cost']         #Purchasing cost
    self.c_b                    = mdpSetup['backlogg_cost']         #Backlogging cost     
    self.c_d                    = mdpSetup['disposal_cost']         #Disposal cost
    self.c_l                    = mdpSetup['lostsale_cost']         #Lost sales cost
    self.c_h                    = mdpSetup['holding_cost']          #Holding cost
    self.maxOrder               = mdpSetup['maxOrder']              #Max order level
    self.leadTime               = mdpSetup['leadTime']              #Lead time
    self.lifeTime               = mdpSetup['lifeTime']              #Merchandise lifetime
    self.maxBacklog             = mdpSetup['maxBacklog']            #Max backlogging level
    self.distMean               = mdpSetup['distMean']              #Mean of sampling distribution
    self.distStd                = mdpSetup['distStd']               #STD of sampling distribution
    self.distMin                = mdpSetup['distMin']               #Min of sampling distribution
    self.distMax                = mdpSetup['distMax']               #Max of sampling distribution
    self.actionDiscFactor       = mdpSetup['actionDiscFactor']      #Action discretization factor
    self.trajLen                = mdpSetup['trajLen']               #Length of a trajectory
    self.trajNum                = mdpSetup['trajNum']               #Number of trajectories
    self.stateSamplingRvs       = mdpSetup['stateSamplingRvs']      #Random variable for sampling states
    self.actionSamplingRvs      = mdpSetup['actionSamplingRvs']     #Random variable for sampling actions
    self.initDistRV_GreedyPol   = mdpSetup['initDistRV_GreedyPol']  #Initial distribution used for policy simulation
    self.initDistRV_ALP         = mdpSetup['initDistRV_ALP']        #Initial distribution used in ALP objective
    self.CPU_CORE               = mdpSetup['CPU_CORE']              #Number of CPU cores
    self.fixedListDemand        = None                              #List of demans
    self.demandRV               = truncnorm(a  =(self.distMin - self.distMean)/self.distStd,
                                            b  =(self.distMax - self.distMean)/self.distStd,
                                            loc=self.distMean,scale=self.distStd)
                                                                    # Demand sampling distribution
    self.fullStates             = None                              # A lsit of all states (if needed)
    self.allActs                = None                              # A lsit of all actions (if needed)
    self.costDictionary         = None                              # A table tracking cost of state-actions pairs


    
  def reinit(self, mdpSetup):    
      super().__init__(mdpSetup)
    
  
  def getSamplesFromStateSpace(self,CS_num):
      return [np.asarray(self.stateSamplingRvs.rvs()) for _ in range(CS_num)]
         
  def getSamplesFromActionSpace(self,CS_num):
      return [np.asarray(self.actionSamplingRvs.rvs()) for _ in range(CS_num)]
              
  def getSamplesFromInitalDist(self,forALP,CS_num):
      if forALP:
        return [self.initDistRV_ALP.rvs() for _ in range(CS_num)]
      else:
        return [self.initDistRV_GreedyPol.rvs() for _ in range(CS_num)]
  def getBatchOfNextStates(self, cur_state, cur_action, demandList = None):
    if demandList is not None:
      self.fixedListDemand = demandList
    
    maxBacklog = self.maxBacklog          
    return list(map(lambda dmnd: [max( (cur_state[1] - max(0,dmnd - cur_state[0])) , maxBacklog), ##
                                cur_state[2],
                                cur_action], self.fixedListDemand))      
    
  def setBatchOfExoInfo(self):
      self.fixedListDemand = self.getBatchSampleFromExogenousInfo()
      return self.fixedListDemand    
          
  def getBatchSampleFromExogenousInfo(self,number=None):
      if number is None:
        return self.demandRV.rvs(self.SA_numSamples) 
      else:
        return self.demandRV.rvs(number)  
  
   
  def isItAStateFeasible(self,state):
        if state[0] >self.maxOrder or state[0]<self.maxBacklog:
            return False
        
        if state[1] >self.maxOrder or state[1]<0:
            return False
        
        if state[2] >self.maxOrder or state[2]<0:
            return False
        return True  
 
  def isItAActionFeasible(self,action):
      if action<0 or action>self.maxOrder:
            return False
      return True

    
  def getExpectedCostOnActionList(self, cur_state, actionList, list_exo_info=None):
    if list_exo_info is None:
      list_exo_info =self.fixedListDemand           
    
    c_p=self.c_p
    dictionary=self.costDictionary
    asarray=np.asarray
    return list(map(lambda act: c_p*act + dictionary[asarray(cur_state).tobytes()],actionList)) 
                     
    
  def getSingleNextStatesWithDemand(self, cur_state, cur_action,dmnd):
      maxBacklog = self.maxBacklog
      return    np.asarray([max( (cur_state[1] - max(0,dmnd - cur_state[0])) ,
                            (maxBacklog - 0)), # 0 =sum_inventories
                            cur_state[2],
                            cur_action]) 
  
  def getListNextStatesWithDemandWithoutAction(self, cur_state, dmnd):
      maxBacklog = self.maxBacklog
      return    [np.asarray([max( (cur_state[1] - max(0,dmnd - cur_state[0])) ,
                            (maxBacklog - 0)), 
                            cur_state[2],
                            cur_action])   for cur_action in self.allActs]
    
  def getCostWithDemand(self, cur_state, cur_action, demand):
      nState=np.asarray(cur_state).tobytes()
      
      if nState in self.costDictionary:
           return self.c_p*cur_action  + self.costDictionary[nState]
      else:
           return self.c_p*cur_action  + \
                        (self.c_h*(max(0,cur_state[1] - max(0,demand-cur_state[0]))) +  
                         self.c_b*(max(0,demand- (cur_state[1]+cur_state[0]))) +
                         self.c_d*(max(0,cur_state[0]-demand)) +
                         self.c_l*(max(0,(self.maxBacklog + demand - \
                                       (cur_state[1]+cur_state[0])))))     
    
  def getExpectedCost(self, cur_state, cur_action, list_exo_info=None):
        if list_exo_info is None:
          list_exo_info =self.fixedListDemand
           
        sum_inventories = cur_state[1]+cur_state[0]
        c_h = self.c_h
        c_b = self.c_b
        c_d = self.c_d
        c_l = self.c_l
        c_p = self.c_p
        maxBacklog = self.maxBacklog
        fixedListDemand = self.fixedListDemand
        return c_p*cur_action+sum(map(
                    lambda  demand:
                            c_h*(max(0,cur_state[1] - max(0,demand-cur_state[0]))) +\
                            c_b*(max(0,demand- sum_inventories)) +\
                            c_d*(max(0,cur_state[0]-demand)) +\
                            c_l*(max(0,(maxBacklog + demand -sum_inventories)))
                                , fixedListDemand))/len(fixedListDemand)





  def setDesceretizedStatesAndActions(self):    
    R = self.myRound
    I = np.arange(self.maxBacklog, self.maxOrder + self.actionDiscFactor, self.actionDiscFactor)
    J = np.arange(0, self.maxOrder + self.actionDiscFactor, self.actionDiscFactor)
    K = np.arange(0, self.maxOrder + self.actionDiscFactor, self.actionDiscFactor)
    
    global roundIt
    def roundIt(i,j,k):
        return [R(i),R(j),R(k)]
    
    pool = Pool(self.CPU_CORE)
    X= pool.starmap(roundIt,list(product(I,J,K)))
    pool.close()
    pool.join()    
    
    self.fullStates=np.asarray(X)  
    self.allActs = np.arange(start=0, stop =self.maxOrder+self.actionDiscFactor, step =self.actionDiscFactor)
    
    for _ in range(len(self.allActs)):
      self.allActs[_] = self.myRound(self.allActs[_])
      
    self.setParitalCosts()  

  def setParitalCosts(self):
    c_h = self.c_h
    c_b = self.c_b
    c_d = self.c_d
    c_l = self.c_l
    c_p = self.c_p
    maxBacklog = self.maxBacklog
    fixedListDemand = self.fixedListDemand  
    
    global paritalCost
    def paritalCost(cur_state):
        cur_state = np.asarray(cur_state)
        sum_inventories = cur_state[1]+cur_state[0]  
        return sum(map(lambda demand: c_h*(max(0,cur_state[1] - max(0,demand-cur_state[0]))) +\
                                      c_b*(max(0,demand- sum_inventories)) +\
                                      c_d*(max(0,cur_state[0]-demand)) +\
                                      c_l*(max(0,(maxBacklog + demand -sum_inventories))),
                              fixedListDemand)\
                   )/len(fixedListDemand)   
     
    pool = Pool(self.CPU_CORE)
    X= pool.map(paritalCost,self.fullStates)
    pool.close()
    pool.join()
    
    self.costDictionary = {self.fullStates[i].tobytes(): X[i] for i in range(len(X))}
    

  def getBestAction(self,curState,dicts):
    
    val             = self.getExpectedCostOnActionList(curState,self.allActs)
    R               = self.myRound
    discount        = self.discount
    SA_numSamples   = self.SA_numSamples     
    allActs         = self.allActs
    ratio           = discount/SA_numSamples
    
    for nextState in np.asarray(self.getBatchOfNextStates(curState,0)):
        nextState[0] = R(nextState[0])
        nextState[1] = R(nextState[1])
        for I in range(len(allActs)):
            nextState[2] = allActs[I]
            val[I] +=  ratio*dicts[nextState.tobytes()]  
    return np.argmin(val)*self.actionDiscFactor # R?
    

  def getSamplePath(self,pathLen):
      return list(self.getBatchSampleFromExogenousInfo(pathLen))


