"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
import numpy as np
from numpy import save
from multiprocessing import Pool
from utils import dropZeros
"""
    --------------------------------------------------------------------------
    The following function samples ALP constraints 
    Specific to Perishable Inventory Control
    --------------------------------------------------------------------------
"""
def getStateActionSamples(mdp,CSSetup,iPath):
    #--------------------------------------------------------------------------
    # Call related function from object "mdp" to sample state, action, demand, and 
    # initial state from appropriate distributions specified in "CSSetup" 
    # (please see dictionary info['cs'] in "MDP/PerishableInventory/Instances/INS_1"
    # to see how parameters of constraint sampling should be configured.) 
    # The constraints are saved in folder 'SAMPLED_STATE_ACTION_BASIS".
    stateList   = mdp.getSamplesFromStateSpace(CSSetup['CS_num'])
    actionList  = mdp.getSamplesFromActionSpace(CSSetup['CS_num'])   
    initList    = mdp.getSamplesFromInitalDist(True,CSSetup['SA_numSamples'])
    mdp.setBatchOfExoInfo()
    save(iPath+"/SAMPLED_STATE_ACTION_BASIS/stateList",     
             stateList,              allow_pickle=True) 
    save(iPath+"/SAMPLED_STATE_ACTION_BASIS/actionList",    
             actionList,            allow_pickle=True)
    save(iPath+"/SAMPLED_STATE_ACTION_BASIS/initialStateSamples",
             initList,              allow_pickle=True)  
    save(iPath+"/SAMPLED_STATE_ACTION_BASIS/demandList",
             mdp.fixedListDemand,   allow_pickle=True)  

"""
--------------------------------------------------------------------------
The following function samples ALP constraints 
Specific to Generalized Joint Replenishment
--------------------------------------------------------------------------
"""
def getASample(givenMDP,i):
    #----------------------------------------------------------------------
    # Sample a state uniformly
    s = [dropZeros(np.random.uniform(low=0.0,high=givenMDP['invUppBounds'][_])) for _ in range(givenMDP['dimX'])]
    s[np.random.randint(givenMDP['dimX'])]=0.0
    s = np.round(s,5)
    
    #----------------------------------------------------------------------
    # Choosing a uniform action, if a sample is close to zero, make it zero
    a = [dropZeros(np.random.uniform(low=0.0,high=givenMDP['invUppBounds'][_]))  for _ in range(givenMDP['dimX'])]
    a = np.round(a,5)
    
    #----------------------------------------------------------------------
    # A heuristic for generating a random action
    g = np.random.uniform(0.0,1.0)
    if g> 0.5:
        a[np.random.randint(givenMDP['dimX'])] = 0.0
    if g> 0.7:
        a[np.random.randint(givenMDP['dimX'])] = 0.0
   
    #----------------------------------------------------------------------
    # Return state-action pair if the action is feasible
    if sum(a) <= givenMDP['maxOrder'] and all(np.add(s,a) <= givenMDP['invUppBounds']):
        return s,a
   
    #----------------------------------------------------------------------
    # Iterating over k in the while loop  to make an infeasible state-action 
    # pairs as a feasible one.
    k=0
    while sum(a) > givenMDP['maxOrder'] or any(np.add(s,a) > givenMDP['invUppBounds']) :
        a = np.divide(a,2)
        a = [dropZeros(a[i]) for i in range(givenMDP['dimX'])]
        a = np.round(a,5)
       
        k+=1
        if k>100:
            break

    #----------------------------------------------------------------------
    # Return state-action pair if a feasiblepair is found     
    if sum(a) <= givenMDP['maxOrder'] and all(np.add(s,a) <= givenMDP['invUppBounds']):
        return s,a
    

"""
Perform constraint sampling
"""
def sampleConstraints(givenMDP,numConstr,trial,thrd):
    #----------------------------------------------------------------------
    # In a pool, sample GJR constraints, and make transition times
    pool = Pool(thrd)
    vals = pool.starmap(getASample,zip([givenMDP for _ in range(numConstr)],range(numConstr)))
    pool.close()
    pool.join()
    transTime_local = lambda s,a: np.min([(s[_]+a[_])/givenMDP['consumptionRate'][_] 
                                              for _ in range(givenMDP['dimX'])])
   
    #----------------------------------------------------------------------
    # Generate next states given sampled state-action pairs
    def getNextState_local(s,a):
        Z       = np.asarray([(s[_]+a[_])/givenMDP['consumptionRate'][_] 
                              for _ in range(givenMDP['dimX'])])
        Zmin    = Z.min()
        ns      = np.add(np.add(s,a),-transTime_local(s,a)*givenMDP['consumptionRate'])
       
        for _ in range(len(ns)):
            if abs(Z[_] - Zmin)<1e-4:
                ns[_] = 0.0
        return np.round(ns,5)
   
    #----------------------------------------------------------------------
    # Set up state-action-state triples and save the lists        
    MVC_StateList  = [vals[_][0] for _ in range(len(vals))]
    MVC_ActionList = [vals[_][1] for _ in range(len(vals))]
    MVC_NStateList = [getNextState_local(vals[_][0],vals[_][1]) for _ in range(len(vals))]
    np.save('Output/GJR/'+givenMDP['mdp_name']+'/SampleConstraints/TRIAL_'+str(trial)+'/MVC_StateList.npy' , np.asarray(MVC_StateList), allow_pickle=False)
    np.save('Output/GJR/'+givenMDP['mdp_name']+'/SampleConstraints/TRIAL_'+str(trial)+'/MVC_ActionList.npy', np.asarray(MVC_ActionList), allow_pickle=False)
    np.save('Output/GJR/'+givenMDP['mdp_name']+'/SampleConstraints/TRIAL_'+str(trial)+'/MVC_NStateList.npy', np.asarray(MVC_NStateList), allow_pickle=False)        
    return MVC_StateList,MVC_ActionList,MVC_NStateList
