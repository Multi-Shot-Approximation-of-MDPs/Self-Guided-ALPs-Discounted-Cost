"""
###############################################################################
# Created: Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
#                           | http://business.uic.edu/faculty/parshan-pakiman
#                          
# Licensing Information: The MIT License
###############################################################################
"""
from multiprocessing import Pool
import numpy as np
from math import log,pi,sqrt
from sampyl import Metropolis

"""
    Lambda is a fixed parameter defined in Lemma EC.3 of Lin, Qihang, Selvaprabu
    Nadarajah, and Negar Soheili. "Revisiting approximate linear programming: 
    Constraint-violation learning with applications to inventory control and energy
    storage." Management Science (2019). Specific to PIC, we can compute Lambda
    using the following function.
"""
def getLambda(mdp,bf,lbSetting,ALP_opt_obj_val):
    
    #--------------------------------------------------------------------------
    # Please see these constants for each instance of PIC, e.g., MDP/
    # PerishableInventory/Instances/INS_1.py
    LNS_const_log_prob  = lbSetting['LNS_const_log_prob']
    LNS_const_log_gamma = lbSetting['LNS_const_log_gamma']
    LNS_R_Q             = lbSetting['LNS_R_Q']
    
    #--------------------------------------------------------------------------
    # Given weights of a VFA, we compute 1-norm of its weights.
    norm_coefs = np.linalg.norm(bf.optCoef[1:len(bf.optCoef)],1)
    
    #--------------------------------------------------------------------------
    # Lipschitz constant of function Y; Please see Online Supplement of Pakiman
    # et al. 2019.
    L           = 2*norm_coefs
    L          += (mdp.c_p + mdp.c_h + mdp.c_b + mdp.c_d + mdp.c_l)
    L           = L/(1-mdp.discount)
    LNS_const   = LNS_const_log_prob+  LNS_const_log_gamma -  L*LNS_R_Q                                
    Lambda      = abs(1/(LNS_const + mdp.dimX + mdp.dimU))
    return Lambda, LNS_const
    
 
"""
     Please see the Online Supplement of Pakiman t al. 2019. This function 
     computes a lower bound on the optimal cost ba sampling approach.
"""   
def get_LNS_LB(mdp,bf,initSateList, lbSetting,ALP_opt_obj_val):
    #--------------------------------------------------------------------------
    # Define auxiliary parameters and functions.
    Lambda, LNS_const       = getLambda(mdp,bf,lbSetting,ALP_opt_obj_val)
    discount                = mdp.discount
    isItAStateFeasible      = mdp.isItAStateFeasible
    isItAActionFeasible     = mdp.isItAActionFeasible
    getBatchOfNextStates    = mdp.getBatchOfNextStates
    getExpectedCost         = mdp.getExpectedCost
    getVFA                  = bf.getVFA
    splNumber               = len(mdp.fixedListDemand)
    initStatesNumber        = len(initSateList)
    
    #--------------------------------------------------------------------------
    # We randomly pick some state-action pairs as initial states of an MC  that
    # is used for generating the lower bound.
    proposedStates          = mdp.getSamplesFromStateSpace(200)  # 50
    proposedAction          = mdp.getSamplesFromActionSpace(200) # 50
    
    #--------------------------------------------------------------------------
    # The following function assigns a mass to each state-action pair, given a 
    # VFA, by measuring how much a state-action pair is violating ALP constraints.
    global saddleFunction
    def saddleFunction(state,action):     
        #----------------------------------------------------------------------
        # Check if a state-action pair is feasible to a PIC instance.
        if isItAStateFeasible(state) and isItAActionFeasible(action):
            nxtStateList = getBatchOfNextStates(state,action)
            expectedBF   = discount*(sum(map(getVFA, nxtStateList))/splNumber)
            expectedBFOnInitStates = sum(map(getVFA, initSateList))/initStatesNumber
            val = ((1/(1-discount))*(getExpectedCost(state,action) - (getVFA(state) - expectedBF))) + \
                                         expectedBFOnInitStates
            return val
        
        #----------------------------------------------------------------------
        # If a state-action pair is infeasible to a PIC instance, then assign 
        # infinity value, e.g., no violation in the ALP constraint! This ensures,
        # with small probability, we may sample this infeasible pair.
        else:
            return float('inf')

    #----------------------------------------------------------------------
    # Logp balances saddleFunction via constant -1/Lambda.
    global logp
    def logp(state,action):
        return -saddleFunction(state,action)/Lambda  
    
    #----------------------------------------------------------------------
    # Starting from "initState", the following function, runs Metropolis Hasting
    # algorithm and samples from function "logp". We burn the first 500 samples.
    global sampleFromInitState       
    def sampleFromInitState(initState):
        return Metropolis(logp,initState).sample(num=650,burn=500,progress_bar=False)
    
    #----------------------------------------------------------------------
    # Evaluate the value of saddleFunction over proposed state-action pairs.
    pool = Pool(mdp.CPU_CORE)
    vals = pool.starmap(saddleFunction,zip(proposedStates,proposedAction))
    pool.close()
    pool.join()
    
    #----------------------------------------------------------------------
    # Sort value ofsaddleFunction on the proposed state-action pairs and 
    # select those pairs with maximum saddleFunction value.
    numInitStates   = mdp.CPU_CORE
    vals            = np.argsort(np.asarray(vals),kind='heapsort')    
    proposedStates  = [proposedStates[vals[_]] for _ in range(numInitStates)]
    proposedAction  = [proposedAction[vals[_]] for _ in range(numInitStates)]
    start           = [{'state':proposedStates[_],'action':proposedAction[_]} for _ in range(numInitStates)]
    
    #----------------------------------------------------------------------
    # Sort Metropolis Hasting Algorithm from those points with maximum 
    # saddleFunction values that are stored in the list start.
    pool    = Pool(mdp.CPU_CORE)
    chain   = pool.map(sampleFromInitState,start)
    pool.close()
    pool.join()
    

    #----------------------------------------------------------------------
    # Set the state-action pairs visited in the MC.
    proposedStates  = np.reshape(np.asarray([np.asarray(chain[_].state) for _ in range(len(start))]),
                               newshape=(len(start)*len(chain[0].state),mdp.dimX))
    proposedAction  = np.reshape(np.asarray([chain[_].action for _ in range(len(start))]),
                               newshape=(len(start)*len(chain[0].action),mdp.dimU))        
    stateList       =  proposedStates
    actionList      =  proposedAction 
    
    #----------------------------------------------------------------------
    # Evaluate function saddleFunction on the state-action pairs visited in the MC
    # and stored in stateList and actionList.
    pool = Pool(mdp.CPU_CORE)
    LB = pool.starmap(saddleFunction,zip(stateList,actionList))
    pool.close()
    pool.join()
    
    #----------------------------------------------------------------------
    # The average value of saddleFunction plus some constants that are provided.
    # below gives a lower bound estimate. Also, compute the standard error.
    meanLB  =  np.mean(LB)
    seLB    =  float(((np.std(LB)/sqrt(len(LB)))/meanLB))*100
    meanLB += (mdp.dimX + mdp.dimU)*Lambda*log(Lambda)
    meanLB += Lambda*LNS_const 
    return meanLB,seLB
