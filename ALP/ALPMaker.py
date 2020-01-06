"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
from numpy import load,save,asarray
from multiprocessing import Pool
from functools import partial

"""
Generate  ALP constraints.
"""
def makeALP(mdp,bf,iPath):
    #--------------------------------------------------------------------------
    # Initialize to NULL the ALP constraint matrix, constraints right-hand-side,
    # and objective coefficents.
    constraintMatrix = None
    RHSVector        = None
    objectiveVector  = None
    
    #--------------------------------------------------------------------------
    # Sample coefficients of random Fourier bases and save the sampled parameters.  
    bf.setRandBasisCoefList()
    save(iPath+"/SAMPLED_STATE_ACTION_BASIS/FourierIntercept", bf.intercept_list,allow_pickle=True) 
    save(iPath+"/SAMPLED_STATE_ACTION_BASIS/FourierTheta", bf.theta_list, allow_pickle=True)
    
    #--------------------------------------------------------------------------
    # Load sampled states, actions, and initial states.
    stateList  = load(iPath+'/SAMPLED_STATE_ACTION_BASIS/stateList.npy')
    actionList = load(iPath+'/SAMPLED_STATE_ACTION_BASIS/actionList.npy')
    initList   = load(iPath+'/SAMPLED_STATE_ACTION_BASIS/initialStateSamples.npy')  
    ALPNumConstraint = len(stateList)
    
    #--------------------------------------------------------------------------
    # The ALP objective function is the sample average approximation of basis 
    # functions evaluation on the set of sampled initial states.
    objectiveVector  = sum(map(bf.evalBasisList,initList))/len(initList)
    
    #--------------------------------------------------------------------------
    # Auxilary redefinition of constants and functions.
    discount                = mdp.discount
    evalBasisList           = bf.evalBasisList
    expectedBasisList       = bf.expectedBasisList
    getBatchOfNextStates    = mdp.getBatchOfNextStates
    getExpectedCost         = mdp.getExpectedCost
    
    #--------------------------------------------------------------------------
    # The following function receives a state-action pair and creates a 
    # constraint of ALP. This function is defined globally to be used in a
    # multithread process.
    global makeALPSingleConstriant
    def makeALPSingleConstriant(state,action): 
        return evalBasisList(state), discount*expectedBasisList(getBatchOfNextStates(state,action)),\
                                getExpectedCost(state,action), state   
    
    #--------------------------------------------------------------------------
    # Create ALP constraints in a pool.
    pool = Pool(mdp.CPU_CORE)
    X    = pool.starmap(makeALPSingleConstriant,zip(stateList,actionList))
    pool.close()
    pool.join()
    
    #--------------------------------------------------------------------------
    # Variable "X" has three components:
    #   1) X[i][0]: vector of the value of basis functions list at the current
    #               state.
    #   2) X[i][1]: vector of the expected value of basis functions list at the 
    #               next state.
    #   3) X[i][2]: ALP RHS values.
    constraintMatrix    = [X[i][0][:]-X[i][1][:]  for i in range(ALPNumConstraint)]
    RHSVector           = [X[i][2]  for i in range(ALPNumConstraint)]
    expVFA              = [X[i][1][:]  for i in range(ALPNumConstraint)]
    
    #--------------------------------------------------------------------------
    # Save ALP constraints.
    save(iPath+"/ALP_COMPONENTS/ALPConstMatrix", asarray(constraintMatrix),allow_pickle=True)  
    save(iPath+"/ALP_COMPONENTS/ALPRHSVector", asarray(RHSVector), allow_pickle=True)
    save(iPath+"/ALP_COMPONENTS/ALPobjectiveVector", objectiveVector, allow_pickle=True)
    save(iPath+"/ALP_COMPONENTS/exp_VFA", expVFA, allow_pickle=True)

    return True
