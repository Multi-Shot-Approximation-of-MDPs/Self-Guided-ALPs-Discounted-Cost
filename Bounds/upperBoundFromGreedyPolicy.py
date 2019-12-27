"""
###############################################################################
# Created: Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
#                           | http://business.uic.edu/faculty/parshan-pakiman
#                          
# Licensing Information: The MIT License
###############################################################################
"""
from multiprocessing import Manager
from multiprocessing import Pool
from functools import partial
from numpy import asarray,std,sqrt,mean

#--------------------------------------------------------------------------
# A global object
mapping = Manager().dict()

"""
    This function receives initial state (initState), MDP (mdp), state-value 
    table (state_VFA_dict), and a fixed set of demand realization (trjDemandList),
    and computes the cost of greedy policy on this trajectory of demands.
"""
def getSingleTrajectoryCost(mdp,initState,state_VFA_dict,trjDemandList):
    
    #--------------------------------------------------------------------------
    # Define some auxiliary variables and abbreviated functions
    cumCost                         = 0
    curState                        = initState
    R                               = mdp.myRound
    getBestAction                   = mdp.getBestAction
    getCostWithDemand               = mdp.getCostWithDemand
    getSingleNextStatesWithDemand   = mdp.getSingleNextStatesWithDemand
    discount                        = mdp.discount
    
    #--------------------------------------------------------------------------
    # Iterate over stages of the given trajectory
    for trajLen in range(mdp.trajLen): 
       npState      = curState              # numpy state
       npState[0]   = R(npState[0])         # round reached state
       bytesState   = npState.tobytes()     # code state to bytes
       
       #-----------------------------------------------------------------------
       # If for state (bytesState) we already computed optimal action, then we
       # retive such action from dictionary (mapping); otherwise, we compute 
       # the best action via (getBestAction).
       if bytesState in mapping:            
           optAct               = mapping[bytesState]     
       else:           
           optAct               = getBestAction(curState,state_VFA_dict)
           mapping[bytesState]  = optAct
       
       #-----------------------------------------------------------------------
       # Given optimal action and current state, update the next state and 
       # compute the immediate cost
       nextCost      = getCostWithDemand(curState,optAct,trjDemandList[trajLen])
       cumCost      += pow(discount,trajLen)*nextCost
       curState      = getSingleNextStatesWithDemand(curState,optAct,trjDemandList[trajLen]) 
    return cumCost

"""
    The following function estimates the cost of greedy policy (upper bound),
    given a VFA.
"""
def simulateGreedyPolicy(mdp,bf,splPAths):
    #--------------------------------------------------------------------------
    # Create a dictionary of state-value.
    pool            = Pool(mdp.CPU_CORE)
    valueBFList     = pool.map(bf.getVFA,mdp.fullStates)
    pool.close()
    pool.join()
    dicts           = {mdp.fullStates[i].tobytes(): 
                            valueBFList[i] for i in range(len(valueBFList))}
    g               = partial(getSingleTrajectoryCost,mdp,
                              asarray(mdp.initDistRV_GreedyPol.rvs()),dicts) 
    
    #--------------------------------------------------------------------------
    # For multiple sample-path, run a pool and compute cost policy on different
    # paths.
    pool            = Pool(mdp.CPU_CORE)
    costList        = pool.map(g,splPAths)
    pool.close()
    pool.join()
    
    #--------------------------------------------------------------------------
    # Calculate the mean and standard error of computed costs.
    meanCost        = mean(costList)
    sError          = ((std(costList)/sqrt(len(costList)))/meanCost)*100
    mapping.clear()
    return meanCost,sError
