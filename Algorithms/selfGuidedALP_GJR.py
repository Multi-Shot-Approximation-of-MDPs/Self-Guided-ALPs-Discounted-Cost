"""
###############################################################################
# Created: Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
#                           | http://business.uic.edu/faculty/parshan-pakiman
#                          
# Licensing Information: The MIT License
###############################################################################
"""
import numpy as np
import textwrap
import gurobipy as gb
from gurobipy import *
import time
from BasisFunction.stumpBasisFunctions import StumpBasis
from utils import printWithPickle,powerset,whereIsElementInPowerSet
from multiprocessing import Pool
from os import unlink

""" Global Gurobi Variables """
INFINITY      = gb.GRB.INFINITY
CONTINUOUS    = gb.GRB.CONTINUOUS
BINARY        = gb.GRB.BINARY
MAXIMIZE      = gb.GRB.MAXIMIZE
MINIMIZE      = gb.GRB.MINIMIZE
quicksum      = gb.quicksum
LinExpr       = gb.LinExpr

class SelfGuidedALP:
    
    def __init__(self,expInfo):
        self.mdp            = expInfo['mdp']['type'](expInfo['mdp'])
        self.bf             = expInfo['bf']
        self.ub             = expInfo['ub']
        self.dimX           = self.mdp.dimX
        self.numThreads     = self.mdp.Threads 
        self.getCost        = self.mdp.getExpectedCost
        self.transTime      = self.mdp.transTime
        self.powSet         = powerset(range(0,self.dimX))
        self.powSetLen      = len(self.powSet)
        self.RNG_DIM_X      = range(self.dimX)
        self.RNG_POW_SET    = range(self.powSetLen)
         

    """
        This function Implements FALP and FGLP models.
    """
    def ALP_GJR(self,
                BF,                                 # Basis function object (for future use)
                BF_Number,                          # The current number of bases    
                stateList,                          # List of states that ALP constraints should be enforced
                actionList,                         # List of actions that ALP constraints should be enforced
                MVC_NStateList,                     # List of next states corresponding to stateList and actionList 
                MVC_sgnStateList,                   # Value signum functions on the stateList
                MVC_sgnNStateList,                  # Value signum functions on the MVC_NStateList
                model= None,                        # Existing ALP Model if there is any
                basisItr = 0,                       # Basis functions counter
                enforceSGConstr=False,              # Enforce self-guiding constraints? (for future use)   
                Perv_VFA_Coef = None,               # Optimal weights of random stumps form the last iterate   
                linCoef_Old=None,                   # Optimal weights of affine VFA form the previous iterate  
                intercept_Old = None,               # Optimal intercept form the last iterate  
                BF_Perv_index_list=None,            # Parameter of random stumps form the previous iterate
                BF_Perv_threshold_list=None):       # Parameter of random stumps form the previous iterate

        #----------------------------------------------------------------------
        # If we did not create any ALP model, then create it for the first time!
        RNG_BF_NUMBER = range(BF_Number)
        if model is None:
            #------------------------------------------------------------------
            # Compute the immediate cost as well as the next transition time from
            # a state-action pair where the pairs are given in two lists stateList
            # and actionList.
            pool = Pool(self.numThreads)
            COST= pool.starmap(self.getCost, zip(stateList,actionList))
            pool.close()
            pool.join()
            pool = Pool(self.numThreads)
            TRAN_TIME= pool.starmap(self.transTime, zip(stateList,actionList))
            pool.close()
            pool.join()
           
            #------------------------------------------------------------------
            # Reconciling the previous VFA. If basisItr=10, then we only have 
            # the affine VFA,
            if basisItr == 10:
                prevVFA_func    = lambda _: float(intercept_Old -  np.dot(stateList[_],linCoef_Old))
                prevVFA         = list(map(prevVFA_func, range(len(stateList))))
            
            #------------------------------------------------------------------
            # Otherwise, we have a VFA consisting of the affine plus a weighted
            # sum of basis functions.    
            if basisItr >= 20:
                #--------------------------------------------------------------
                # Set up random stumps used in the last iteration
                prevBasis            = StumpBasis(self.bf)
                prevBasis.BF_number  = len(Perv_VFA_Coef)
                prevBasis.setSampledParms(index      = BF_Perv_index_list,
                                          threshold  = BF_Perv_threshold_list)                
                #--------------------------------------------------------------
                # Calculate VFA in all states
                prevVFA_func    = lambda _: float(intercept_Old -  
                                                  np.dot(stateList[_],linCoef_Old) - 
                                                  np.dot(prevBasis.evalBasisList(stateList[_]),Perv_VFA_Coef))
                prevVFA         = list(map(prevVFA_func, range(len(stateList))))


            #------------------------------------------------------------------
            # Creating an ALP Gurobi model
            ALP = gb.Model('ALP')
            ALP.setParam('OutputFlag',      False)
            ALP.setParam('LogFile',         self.mdp.Adrr+'/groubiLogFile.log')
            unlink('gurobi.log')
            ALP.setParam('Threads',         self.numThreads)
            ALP.setParam('FeasibilityTol',  1e-9)
            ALP.setParam('NumericFocus',    3)

         
            #------------------------------------------------------------------
            # ALP Variables, e.g., bases weights
            intercept  = ALP.addVar(ub =  INFINITY, lb  = -INFINITY, vtype= CONTINUOUS,
                                    name = 'intercept')
            linCoefVar = ALP.addVars(self.RNG_DIM_X, 
                                     ub = [ INFINITY for _ in self.RNG_DIM_X],
                                     lb = [-INFINITY for _ in self.RNG_DIM_X],
                                     vtype=CONTINUOUS,
                                     name = 'linCoefVar')
            
            
            #------------------------------------------------------------------
            # ALP Objective for average-cost semi-MDP defining GJR
            linCoef   = [linCoefVar[i] for i in self.RNG_DIM_X]
            ALP.setObjective(intercept + LinExpr(self.mdp.consumptionRate,linCoef), MAXIMIZE)  
    
            #------------------------------------------------------------------
            # ALP constraints: using affine VFA
            if basisItr == 0 :
                ALP.addConstrs(TRAN_TIME[itr]*intercept \
                                      +  LinExpr(actionList[itr],linCoef)  <= \
                                        COST[itr] for itr in range(len(stateList)))    
               
                #--------------------------------------------------------------
                # Solve ALP & return optimal weights
                ALP.update()
                ALP.optimize()
                return intercept.X, \
                       [linCoef[_].X for _ in self.RNG_DIM_X], \
                       [None for  _ in RNG_BF_NUMBER],\
                       ALP.objVal, \
                       ALP        
                       
            #------------------------------------------------------------------
            # ALP constraints: using affine plus random stumps             
            if basisItr > 0 :    
                #--------------------------------------------------------------
                # ALP Variables: weights of random stumps
                BF_CoefVar = ALP.addVars(RNG_BF_NUMBER, ub    = [ INFINITY for _ in RNG_BF_NUMBER],
                                                        lb    = [-INFINITY for _ in RNG_BF_NUMBER],
                                                        vtype = CONTINUOUS,
                                                        name  = 'BF_CoefVar')
                BF_Coef     = [BF_CoefVar[i] for i in RNG_BF_NUMBER]    
       
                #--------------------------------------------------------------
                # ALP constraints: using affine plus random stump bases
                ALP.addConstrs(TRAN_TIME[itr]*intercept \
                                       +  LinExpr(actionList[itr],linCoef) + \
                                    LinExpr(MVC_sgnNStateList[itr]-MVC_sgnStateList[itr],BF_Coef) <= \
                                        COST[itr] for itr in range(len(stateList)))    
                    
                #--------------------------------------------------------------
                # ALP constraints: self-guiding constraints
                """ 
                    Remove self-guiding constraints to obtain the FALP model 
                """
                if basisItr == 10:
                    ALP.addConstrs( prevVFA[itr]   <= intercept  - LinExpr(stateList[itr],linCoef)    \
                                                  for itr in range(len(stateList)))  
                if basisItr >= 20:
                    ALP.addConstrs( prevVFA[itr]   <= intercept  - LinExpr(stateList[itr],linCoef) -LinExpr(MVC_sgnStateList[itr],BF_Coef)  \
                                                  for itr in range(len(stateList)))
    
                #--------------------------------------------------------------
                # Solve ALP & return optimal weights
                ALP.update()
                ALP.optimize()

                return intercept.X, \
                       [linCoef[_].X for _ in self.RNG_DIM_X], \
                       [BF_Coef[_].X for  _ in RNG_BF_NUMBER],\
                       ALP.objVal, \
                       ALP                              
                       
        #----------------------------------------------------------------------
        # If we already solved an ALP and we have a Gorubi ALP model with a fewer
        # number of random bases than the current iterate number of bases, then
        # use the already created Gorubi model.   
        else:
            #------------------------------------------------------------------
            # Load variables of the Gorubi model
            vars        = model.getVars()
            intercept   =  vars[0]
            linCoef     = [vars[_+1] for _ in self.RNG_DIM_X]
            BF_Coef     = [vars[_+1+len(self.RNG_DIM_X)] for _ in RNG_BF_NUMBER]  
            
            #------------------------------------------------------------------
            # Adjust parameters of Gorubi
            model.setParam('NumericFocus',3)
            model.setParam('FeasibilityTol',1e-9)
           
            #------------------------------------------------------------------
            # Focus on the most violated state-action pair
            mostRecentViolatedState  = stateList[0]
            mostRecentViolatedAction = actionList[0]
            valueOfStumpState        = MVC_sgnStateList[0]
            valueOfStumpNextState    = MVC_sgnNStateList[0]
            
            #------------------------------------------------------------------
            # Add ALP constraint associated with the most violated state-action pair
            model.addConstr(self.transTime(mostRecentViolatedState,mostRecentViolatedAction)*intercept +\
                            LinExpr(mostRecentViolatedAction,linCoef) + \
                            LinExpr(valueOfStumpNextState,BF_Coef) <=  LinExpr(valueOfStumpState,BF_Coef)+\
                                self.getCost(mostRecentViolatedState,mostRecentViolatedAction))

            #--------------------------------------------------------------
            # ALP constraints: self-guiding constraints
            """ 
                Remove self-guiding constraints to obtain the FALP model 
            """
            if basisItr == 10:
                model.addConstr(  float(intercept_Old -  np.dot(mostRecentViolatedState,linCoef_Old))   <= \
                                    intercept  - LinExpr(mostRecentViolatedState,linCoef) )  
            if basisItr >= 20:
                model.addConstr(  float(intercept_Old -  np.dot(mostRecentViolatedState,linCoef_Old))   <= \
                                    intercept  - LinExpr(mostRecentViolatedState,linCoef)  \
                                            -LinExpr(valueOfStumpState,BF_Coef) )
                    
            #------------------------------------------------------------------
            # Solve ALP & return optimal weights                           
            model.update()      
            model.optimize()   
            return intercept.X, \
                   [linCoef[_].X for _ in self.RNG_DIM_X], \
                   [BF_Coef[_].X for  _ in RNG_BF_NUMBER],\
                   model.objVal, \
                   model
                                 
    """
        The following function checks if ALP with current sampled state-action 
        pair is bounded or not.
    """       
    def isSampledALPFeasible(self, MVC_StateList,MVC_ActionList,MVC_NStateList):
        #----------------------------------------------------------------------
        # If ALP can be solved without any unboundedness error, then return true;
        # otherwise, return false.
        try:
            self.ALP_GJR(BF                 = None,
                         BF_Number          = 0,
                         stateList          = MVC_StateList,
                         actionList         = MVC_ActionList,
                         MVC_NStateList     = MVC_NStateList,
                         MVC_sgnStateList   = [],
                         MVC_sgnNStateList  = [],
                         model              = None)
            return True
        except:
            return False
       
    """
    The following function iteratively performs following steps
        1. Sampling random stumps
        2. Solving ALP
        3. Generating constraints (solving separation problem)
    """            
    def SG_ALP(self, expInfo,BF,BF_Setup,indexList,thresholdList,trial,thrd):
        #----------------------------------------------------------------------
        # Set up output 
        resTable        = np.zeros(shape = (BF_Setup['BF_num'],12))
        timeGapTable    = np.zeros(shape = (BF_Setup['BF_num'],4))
        runTimeLimit    = expInfo['misc']['runTime']
        
        #----------------------------------------------------------------------
        # Counter on the number of times that bases are sampled. Cumulative runtime.
        basisCounter    = 0
        cumTime         = 0
        
        #----------------------------------------------------------------------
        # Initialization 
        UB          = UB_I      =                       float('inf')
        I_LB        = LB        = LB_I_LB   =           -float('inf')
        CYC_LEN     = I_GAP     = GAP       =   UB_LB =  0
        
        
        #----------------------------------------------------------------------
        # Save parameters of the current random stumps and load state-action pairs
        np.savetxt(self.mdp.Adrr+'/BF_index_list.csv',np.asarray(BF.index_list),delimiter=',')
        np.savetxt(self.mdp.Adrr+'/threshold_list.csv',np.asarray(BF.threshold_list),delimiter=',')
        MVC_StateList  = np.load('Output/GJR/'+self.mdp.mdpName+'/SampleConstraints/TRIAL_'+str(trial)+'/MVC_StateList.npy')
        MVC_ActionList = np.load('Output/GJR/'+self.mdp.mdpName+'/SampleConstraints/TRIAL_'+str(trial)+'/MVC_ActionList.npy')
        MVC_NStateList = np.load('Output/GJR/'+self.mdp.mdpName+'/SampleConstraints/TRIAL_'+str(trial)+'/MVC_NStateList.npy')
        MVC_StateList  = list(MVC_StateList)    
        MVC_ActionList = list(MVC_ActionList)
        MVC_NStateList = list(MVC_NStateList)
        
        #----------------------------------------------------------------------
        # Some loggers to pickle algorithm performance
        logger      = open(self.mdp.Adrr+ '/summary_trial_'+str(trial)+".txt","w+")  
        timeLogger  = open(self.mdp.Adrr+ '/time_gap_list_'+str(trial)+".txt","w+")        
        timeLogger.write('{:^5} | {:^15} | {:^15} | {:^15} |'.format('B', 'GAP (%)', 'Iter-RT (s)', 'Cum-RT (s)'))
        timeLogger.write("\n")
        
        #----------------------------------------------------------------------
        # This set tracks index of those bases accepted, e.g., they obtain 
        # non-zero weights
        acceptedBasesIdx            = set()
        
        #----------------------------------------------------------------------
        # Tracking previous iterate basis functions optimal weights and parameters
        BF_Perv_threshold_list      = None
        BF_Perv_index_list          = None
        Perv_VFA_Coef               = None  # Optimal weights of random stumps form the previous iterate
        linCoef_Old                 = None  # Optimal weights of affine VFA form the previous iterate  
        intercept_Old               = None  # Optimal weights of intercept form the previous iterate  
    
        #----------------------------------------------------------------------
        # Iteratively sample random stump bases
        for basisItr in np.arange(0,BF_Setup['BF_num'],10):
           
            #------------------------------------------------------------------
            # Add a batch (10) of new bases to the already accepted random bases
            if basisItr>0:
                indexOfBasesToBeUsed = set.union(acceptedBasesIdx,set(range(basisItr-10,basisItr)))    
            else:
                indexOfBasesToBeUsed = []
            
            #------------------------------------------------------------------
            # Set the number of basis functions and their parameters
            BF.BF_number = len(indexOfBasesToBeUsed)
            BF.setSampledParms( index      = indexList[list(indexOfBasesToBeUsed)],
                                threshold  = thresholdList[list(indexOfBasesToBeUsed)])
           
            #------------------------------------------------------------------
            # Set time and load initial state-action pairs
            start_time =  time.time()
            MVC_sgnStateList,\
                MVC_sgnNStateList = self.loadStateActionsForFeasibleALP(MVC_StateList,\
                                                        MVC_NStateList,BF)
           
            #------------------------------------------------------------------
            # Run an iteration of self-guided ALPs
            flag,               I_LB,           LB,              UB_I,       UB, \
            CYC_LEN,            I_GAP,          GAP,             LB_I_LB,    UB_LB, \
            effectiveBasis,     MVC_StateList,  MVC_ActionList,  MVC_NStateList, \
            MVC_sgnStateList,   MVC_sgnNStateList,               intercept_Old, \
            linCoef_Old,        Perv_VFA_Coef,  BF_Perv_index_list, \
            BF_Perv_threshold_list    = \
                self.selfGuidedALP_OneIteration(   BF,
                                                   len(indexOfBasesToBeUsed),
                                                   basisItr,
                                                   logger,
                                                   UB,I_LB,
                                                   UB_I,cumTime,
                                                   MVC_StateList,
                                                   MVC_ActionList,
                                                   MVC_NStateList,
                                                   MVC_sgnStateList,
                                                   MVC_sgnNStateList, 
                                                   intercept_Old,
                                                   linCoef_Old,
                                                   Perv_VFA_Coef,
                                                   BF_Perv_index_list,
                                                   BF_Perv_threshold_list)
           
            #------------------------------------------------------------------ 
            # Update the runtime
            runTime = time.time() - start_time
            cumTime += runTime
           
            #------------------------------------------------------------------ 
            # Check the stoping criterion for runtime 
            if cumTime >= runTimeLimit:
                #--------------------------------------------------------------
                # Print and log the results
                printWithPickle('\nTime Out!\n The terminal optimality gap is {} % and computed in {} (m).'.format(GAP,cumTime),logger)
                timeLogger.write("Time Out!\n")
                timeLogger.flush()
                logger.flush()
               
                #--------------------------------------------------------------
                # Store the algorithm output
                resTable        = resTable[~ np.all(resTable==0, axis=1)]
                timeGapTable    = timeGapTable[~ np.all(timeGapTable==0, axis=1)]
                np.savetxt(self.mdp.Adrr+'/RESULTS_TABLE_SG.csv',resTable,delimiter=',',
                                    header='T,B,I-LB,LB,I-UB,UB,C-LEN,I-GAP(%),GAP(%),LB-IMP(%),UB-IMP(%),GAP-IMP(%)')
                np.savetxt(self.mdp.Adrr+'/TIME_GAP_TABLE_SG.csv',timeGapTable,delimiter=',',
                                    header='B,GAP (%),Iter-RT(s),Cum-RT(s)')
       
                #--------------------------------------------------------------
                # Return the results
                return True,resTable
               
               
            #------------------------------------------------------------------
            # Update time-gap table
            timeGapTable[basisCounter,0] = BF.BF_number
            timeGapTable[basisCounter,2] = runTime
            timeGapTable[basisCounter,3] = cumTime
           
            #------------------------------------------------------------------
            # If it is the initial iteration, use the I_GAP. Print the number of
            # bases, gap, and runtime.
            if GAP == None and flag in set([0,1]):
                GAP = I_GAP
                timeLogger.write('{:>5d} | {:>15.2f} | {:>15.2f} | {:>15.2f} |'.format(BF.BF_number,I_GAP,runTime,cumTime))
                timeGapTable[basisCounter,1] = I_GAP
            else:
                timeLogger.write('{:>5d} | {:>15.2f} | {:>15.2f} | {:>15.2f} |'.format(BF.BF_number,GAP,runTime,cumTime))
                timeGapTable[basisCounter,1] = GAP
            timeLogger.write("\n")  
            timeLogger.flush()
             
            #------------------------------------------------------------------
            # If there were a numerical issue, return false;
            if flag == 0:
                return False,None
           
            #------------------------------------------------------------------
            # Resample bases if there were any issue
            elif flag == 1:
                continue    
           
            #------------------------------------------------------------------
            # If everything seems fine, store results and continue sampling
            else:
                acceptedBasesIdx = set([list(indexOfBasesToBeUsed)[i] for i in effectiveBasis] ) #acceptedBasesIdx.union()
                resTable[basisCounter,0]        = trial
                resTable[basisCounter,1]        = BF.BF_number
                resTable[basisCounter,2]        = I_LB
                resTable[basisCounter,3]        = LB
                resTable[basisCounter,4]        = UB_I
                resTable[basisCounter,5]        = UB
                
                if CYC_LEN == None:
                    resTable[basisCounter,6]    = resTable[basisCounter-1,6]
                else:
                    resTable[basisCounter,6]    = CYC_LEN
                resTable[basisCounter,7]        = I_GAP
                resTable[basisCounter,8]        = GAP
                resTable[basisCounter,9]        = (1-I_LB/LB)*100
                resTable[basisCounter,10]       = (1-UB/UB_I)*100
               
                if (not GAP == None) and  (I_GAP == 0 or I_GAP == 1):
                    resTable[basisCounter,11]   = I_GAP - GAP
                else:
                    resTable[basisCounter,11]   = None
                basisCounter+=1
                
                #--------------------------------------------------------------
                # Check the convergence
                if not GAP == None:
                    if GAP <= 2.00:
                        print('The algorithm is converged with the terminal optimality gap of ({:>8.3} %) in ({:>8.3} m).'.format(GAP,cumTime/60))
                        break
             
            logger.flush()
        
        #----------------------------------------------------------------------
        # Save the output and return the results
        logger.close()
        timeLogger.close()
        resTable        = resTable[~ np.all(resTable==0, axis=1)]
        timeGapTable    = timeGapTable[~ np.all(timeGapTable==0, axis=1)]
        np.savetxt(self.mdp.Adrr+'/RESULTS_TABLE_SG.csv',resTable,delimiter=',',
                            header='T,B,I-LB,LB,I-UB,UB,C-LEN,I-GAP(%),GAP(%),LB-IMP(%),UB-IMP(%),GAP-IMP(%)')
        np.savetxt(self.mdp.Adrr+'/TIME_GAP_TABLE_SG.csv',timeGapTable,delimiter=',',
                            header='B,GAP (%),Iter-RT(s),Cum-RT(s)')
      
        return True,resTable 
    
    
    def loadStateActionsForFeasibleALP(self,MVC_StateList,MVC_NStateList,BF):
        #----------------------------------------------------------------------
        # Evaluate random stumps on the states and next states that are in the
        # MVC_StateList MVC_NStateList
        pool = Pool(self.numThreads)
        MVC_sgnStateList = pool.map(BF.evalBasisList, MVC_StateList)
        pool.close()
        pool.join()
        pool = Pool(self.numThreads)
        MVC_sgnNStateList = pool.map(BF.evalBasisList, MVC_NStateList)
        pool.close()
        pool.join()
       
       
        return list(MVC_sgnStateList),list(MVC_sgnNStateList)    



    """ Run Row Generation with Self-guided ALP On Realized Basis Functions """      
    def selfGuidedALP_OneIteration(self,
                                   BF,
                                   numBasis,
                                   basisItr,
                                   logger,
                                   prevUB,
                                   initialLB,
                                   initialUB,
                                   timeElapsed,
                                   MVC_StateList,
                                   MVC_ActionList,
                                   MVC_NStateList,
                                   MVC_sgnStateList,
                                   MVC_sgnNStateList,
                                   intercept_Old,
                                   linCoef_Old,
                                   Perv_VFA_Coef, 
                                   BF_Perv_index_list,
                                   BF_Perv_threshold_list):
       
       
        """ SET UP ALP CONSTRIANT """
        rowGenCounter           = 0
        ALP                     = None
        s                       = None
        a                       = None
        MVCval                  = None
        NS                      = None
        sgnS                    = None
        sgnNS                   = None
        cyclicPolFound          = False
        cycleLength             = 4000
        # If basis functions are ''ill''
        prevState               = np.zeros(self.mdp.dimX)
        prevAction              = np.zeros(self.mdp.dimX)
        illCondBasisCount       = 0
       
        #----------------------------------------------------------------------
        # Iterate and perform constraint generation
        while True:  
            #------------------------------------------------------------------
            # Basis         : No
            # Generated row : No
            if rowGenCounter ==0 and basisItr ==0:
                #--------------------------------------------------------------
                # Solve ALP to get VFA
                intercept,  linCoef, BF_Coef, \
                            ALPval, ALP   = self.ALP_GJR(BF,
                                                         numBasis,
                                                         MVC_StateList,
                                                         MVC_ActionList,
                                                         MVC_NStateList,
                                                         MVC_sgnStateList, 
                                                         MVC_sgnNStateList,
                                                         None,
                                                         basisItr,
                                                         False,
                                                         None,
                                                         None,
                                                         None,
                                                         None)
                            
            #------------------------------------------------------------------
            # Basis         : Yes
            # Generated row : No
            elif rowGenCounter ==0 and basisItr > 0:
                #--------------------------------------------------------------
                # Solve ALP to get VFA: use the already created Gorubi model
                intercept,  linCoef, BF_Coef, \
                             ALPval, ALP   = self.ALP_GJR(BF,numBasis,
                                                          MVC_StateList,
                                                          MVC_ActionList,
                                                          MVC_NStateList,
                                                          MVC_sgnStateList,
                                                          MVC_sgnNStateList,
                                                          None,
                                                          basisItr,
                                                          True,
                                                          Perv_VFA_Coef,
                                                          linCoef_Old,
                                                          intercept_Old,
                                                          BF_Perv_index_list,
                                                          BF_Perv_threshold_list)
                
            #------------------------------------------------------------------
            # Generated row : Yes
            else:
                #--------------------------------------------------------------
                # Basis         : Yes
                if basisItr >= 10:  
                    #----------------------------------------------------------
                    # Solve ALP to get VFA: use the already created Gorubi model
                    intercept,  linCoef, BF_Coef,\
                                ALPval, ALP = self.ALP_GJR(BF,numBasis,[s],[a],
                                                           [NS], [sgnS],[sgnNS],
                                                           ALP,basisItr,True,
                                                           Perv_VFA_Coef,
                                                           linCoef_Old,
                                                           intercept_Old, 
                                                           BF_Perv_index_list,
                                                           BF_Perv_threshold_list)
                    
                #--------------------------------------------------------------
                # Basis         : NO
                else:
                    intercept, linCoef, BF_Coef, ALPval,ALP   = self.ALP_GJR(BF,
                                                                             0,
                                                                             [s],
                                                                             [a],
                                                                             [NS],
                                                                             [sgnS], 
                                                                             [sgnNS],
                                                                             ALP,
                                                                             basisItr,
                                                                             False,
                                                                             None,
                                                                             None,
                                                                             None,
                                                                             None)
            #------------------------------------------------------------------
            # Next, perform constraint generation using the computed VFA
            s, a, MVCval,NS,sgnS,sgnNS = self.getMostViolatedConstraint(intercept,
                                                                        linCoef,
                                                                        BF_Coef,
                                                                        BF)
            
            #------------------------------------------------------------------  
            """
                If the most violating state-action pair of the previous and 
                current iterations are identical, then there is an issue. In fact,
                row generation could not cut off the previous infeasible solution.
                If this issue happens, Gurobi parameters need to be adjusted.
            """
            if np.linalg.norm(s-prevState,np.inf)<1e-50 and \
               np.linalg.norm(a-prevAction,np.inf)<1e-50:  
                    illCondBasisCount   +=1    
            if illCondBasisCount >= 100:
                illCondBasisCount       = 0
                msg = 'WARNING! Row generation could not cut off the previous infeasible solution. Gurobi parameters need to be adjusted for addressing this issue.'
                print('\n\n'+textwrap.TextWrapper(width=50).fill(text=msg)+'\n\n')
                
                #--------------------------------------------------------------
                # The flag shows if row generation was successful or not.
                if basisItr == 0 :
                    flag = 0
                else:
                    flag = 1
                    msg ='We continue solving this problem with a new set of sampled bases'
                    print('\n\n'+textwrap.TextWrapper(width=50).fill(text=msg)+'\n\n')
               
                #--------------------------------------------------------------                    
                # Return the results
                return flag,     initialLB,      ALPval,     initialUB,      prevUB,\
                        None,    (1-initialLB/initialUB)*100,                None,\
                        ALPval/initialLB,        None,       [], MVC_StateList,\
                        MVC_ActionList,          MVC_NStateList, MVC_sgnStateList,\
                        MVC_sgnNStateList,       intercept_Old,  linCoef_Old,\
                        Perv_VFA_Coef,           BF_Perv_index_list,\
                        BF_Perv_threshold_list  
             
    
            #------------------------------------------------------------------    
            # If row generation was successful, then add the newly generated 
            # state-actions.
            MVC_StateList.append(s)
            MVC_ActionList.append(a)
            MVC_NStateList.append(NS)
            MVC_sgnStateList.append(sgnS)
            MVC_sgnNStateList.append(sgnNS)
            
            #------------------------------------------------------------------
            # Store previous state and action
            prevState       = s
            prevAction      = a
           
            #------------------------------------------------------------------
            # Print a header for results
            if basisItr ==0 and rowGenCounter ==0:
                #print('{:{fill}^{w}}'.format(' Self-guided ALP 2019 \t ',   fill=' ',w=148))
                #print('\n \n{:{fill}^{w}}'.format(self.mdp.mdpName,         fill=' ',w=148))
                #print('\n{} | {:{fill}^{w}}|'.format(''.ljust(21), 'ALP',   fill=' ',w=99))
               
                printWithPickle('{:^3} | {:^15} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^5} | {:^5} | {:^5} | {:^8} |'.format(
                        'B','VIOL','I-LB','LB','I-UB','UB','I-GAP','GAP','CYC?','C-LEN','MAX-B','T(s)'),logger)
                printWithPickle('----------------------------------------------------------------------------------------------------------------------------',logger)
           
            #------------------------------------------------------------------
            # Break the loop if constraint violation is negligible
            isALPConverged   = 0.001*prevUB
            if basisItr ==0:
                isALPConverged = 1e-3
            if MVCval>=-isALPConverged:
                
                printWithPickle(str('{:>3d} | {:>15.5f} | {} |'.format(numBasis,(MVCval   ),''.ljust(98))),logger)
                break
            
            #------------------------------------------------------------------
            # Every 50 iteration of the row generation, print results to a user.
            elif rowGenCounter%50 ==0:
                printWithPickle('{:>3d} | {:>15.5f} | {} |'.format(numBasis,(MVCval),''.ljust(98)),logger)  
            
            rowGenCounter+=1  

            
        #----------------------------------------------------------------------
        # Only use the random stumps that have a significant weight and drop the
        # rest of them.
        effectiveBasis          = []
        effectiveBasisIndex     = []
        nonEffectiveBasis       = []
        for i in range(numBasis):
            if abs(BF_Coef[i]) < 1e-3:
                nonEffectiveBasis.append(i)
            else:
                effectiveBasis.append(i)
                effectiveBasisIndex.append(i)
                
        #----------------------------------------------------------------------
        # Store previous VFA
        Perv_VFA_Coef               = np.asarray([BF_Coef[i] for i in effectiveBasisIndex])
        linCoef_Old                 = linCoef
        intercept_Old               = intercept
        BF_Perv_index_list          = np.asarray([BF.index_list[i] for i in effectiveBasisIndex])
        BF_Perv_threshold_list      = np.asarray([BF.threshold_list [i] for i in effectiveBasisIndex])
       
        #----------------------------------------------------------------------
        # Pickle the results
        printWithPickle('{:>3d} | {:^15} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>5} | {:>5d} | {:>5d} | {:>8.0f} |'.format(
                    numBasis,''.ljust(15),initialLB,   ALPval,   initialUB,   prevUB, \
                    (1-initialLB/initialUB)*100,       (1-ALPval/prevUB)*100,
                      cyclicPolFound,   cycleLength, basisItr,timeElapsed),logger)  
       
        """
                Simulate Policy to Get Upper Bound
        """        
        #----------------------------------------------------------------------
        # We simulate policy less frequent than updating the lower bound. Always
        # simulate policy in the first iteration.
        if (numBasis == 0) or (basisItr % self.ub['oftenUpdateUB'] == 0 ):
            #----------------------------------------------------------------------
            # Start with a random state
            s               = [np.random.uniform(low=0.0,
                                                 high=self.mdp.invUppBounds[_])
                                   for _ in range(self.mdp.dimX)]
            s[np.random.randint(self.mdp.dimX)] = 0.0
            s               = np.asarray(s)
            initialState    = s    
            
            #----------------------------------------------------------------------
            # Compute the upper bound
            cyclicPolFound,cycleLength,cost = self.getUpperBound(state_ini=initialState, \
                                                                 BF=BF,
                                                                 BF_Number=numBasis,
                                                                 intercept=intercept,\
                                                                 linCoef=linCoef,
                                                                 BF_Coef=BF_Coef)
                
            #------------------------------------------------------------------
            # Save the policy cost           
            if  cost > prevUB:
                cost = prevUB
              
            #------------------------------------------------------------------
            # Save the initial bounds
            if basisItr==0:
                initialLB   =ALPval
                initialUB   =cost
                prevUB      =cost
                
            #------------------------------------------------------------------
            # Print bounds obtained in the current iterate
            printWithPickle('{:>3d} | {:^15} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>5} | {:>5d} | {:>5d} | {:8.0f} |'.format(
                    numBasis,''.ljust(15), initialLB,   ALPval,   initialUB,   prevUB, \
                    (1-initialLB/initialUB)*100,       (1-ALPval/prevUB)*100,
                    cyclicPolFound,   cycleLength, basisItr,timeElapsed),logger)
            printWithPickle('----------------------------------------------------------------------------------------------------------------------------',logger)
            

            #------------------------------------------------------------------
            # Return the results
            return 2,   initialLB,      ALPval,         initialUB,      prevUB,   \
                    cycleLength,        (1-initialLB/initialUB)*100,    (1-ALPval/prevUB)*100,\
                    ALPval/initialLB,   prevUB/ALPval,  effectiveBasis, MVC_StateList, \
                    MVC_ActionList,     MVC_NStateList, MVC_sgnStateList,  \
                    MVC_sgnNStateList,  intercept_Old,  linCoef_Old, \
                    Perv_VFA_Coef,      BF_Perv_index_list, BF_Perv_threshold_list  
    
        #----------------------------------------------------------------------
        # Return the results in the case of not updating the policy cost
        else:
            printWithPickle('----------------------------------------------------------------------------------------------------------------------------',logger)
            return 2,   initialLB,       ALPval,            initialUB,      prevUB,\
                    None,                (1-initialLB/initialUB)*100,    (1-ALPval/prevUB)*100,\
                    ALPval/initialLB,    None,              effectiveBasis,\
                    MVC_StateList,       MVC_ActionList,    MVC_NStateList, \
                    MVC_sgnStateList,   MVC_sgnNStateList,  intercept_Old,\
                    linCoef_Old,        Perv_VFA_Coef,      BF_Perv_index_list,\
                    BF_Perv_threshold_list    
           

    """ 
        Finding Most Violated Constraint For a Given VFA
        Please see Electronic Companion EC.5 of Pakiman et al.
        Please also see:
            Adelman, Daniel, and Diego Klabjan. "Computing near-optimal policies
            in generalized joint replenishment." INFORMS Journal on Computing 24,
            no. 1 (2012): 148-164.
    """  
    def getMostViolatedConstraint(self,intercept, linCoef, BF_Coef,BF):
        #----------------------------------------------------------------------
        # Fix the number of basis functions
        BF_Number       = len(BF_Coef)
        RNG_BF_NUMBER   = range(BF_Number)
        
        #----------------------------------------------------------------------
        # Configuring the Gorubi model
        MVC = gb.Model('MVC')
        MVC.setParam('OutputFlag',False)
        MVC.setParam('LogFile',self.mdp.Adrr+'/groubiLogFile.log')
        unlink('gurobi.log')
        MVC.setParam('Threads',self.numThreads)
        MVC.setParam('MIPGap' , 0.01)
        MVC.setParam('FeasibilityTol',1e-9)
        MVC.setParam('IntFeasTol',1e-9)
        MVC.setParam('NumericFocus',3)

        #----------------------------------------------------------------------
        # Transition time decision variable            
        t      = MVC.addVar(ub      = INFINITY,
                            lb      = 0,
                            vtype   = CONTINUOUS)
        
        #----------------------------------------------------------------------
        # Most violating action decision variable          
        act    = MVC.addVars(self.RNG_DIM_X,
                             ub     = [self.mdp.invUppBounds[_] for _ in self.RNG_DIM_X],
                             lb     = [0        for _ in self.RNG_DIM_X],
                             vtype  = CONTINUOUS)
        
        #----------------------------------------------------------------------
        # Most violating state decision variable          
        state  = MVC.addVars(self.RNG_DIM_X, 
                             ub     = [self.mdp.invUppBounds[_] for _ in self.RNG_DIM_X],
                             lb     = [0        for _ in self.RNG_DIM_X],
                             vtype  = CONTINUOUS)  
        
        #----------------------------------------------------------------------
        # The next state decision variable corresponding to state and act   
        nState = MVC.addVars(self.RNG_DIM_X, 
                             ub     = [self.mdp.invUppBounds[_] for _ in self.RNG_DIM_X],
                             lb     = [0        for _ in self.RNG_DIM_X],
                             vtype  = CONTINUOUS)
        
        #----------------------------------------------------------------------
        # Auxiliary decision variables that are modeling the MDP transition 
        # function and replenishment time
        Y      = MVC.addVars(self.RNG_POW_SET,  vtype   = BINARY)
        U      = MVC.addVars(self.RNG_DIM_X,    vtype   = BINARY)        
        Up     = MVC.addVars(self.RNG_DIM_X,    vtype   = BINARY)              
        R      = MVC.addVars(self.RNG_DIM_X,    vtype   = BINARY)              
        
        #----------------------------------------------------------------------
        """ 1) Set Objective Function: Linear Part   """
        MVC.setObjective(quicksum(self.mdp.getFixCost(self.powSet[i])*Y[i] 
                                  for i in self.RNG_POW_SET) - \
                          intercept*t - quicksum(linCoef[i]*act[i]  for i in self.RNG_DIM_X),
                          MINIMIZE)
        #----------------------------------------------------------------------
        """ 2) Set Objective Function: Random Stumps : Picewise Linear Objective """
        if BF_Number > 0:
            #------------------------------------------------------------------
            # An upper bound on the value of integer decision variables
            upBoundInProd =  np.max(self.mdp.invUppBounds) + BF.bandWidth_UB
         
            #------------------------------------------------------------------
            # 4 points that are defining the PWL approximation of sign function
            X            = [-2*BF.margin,-BF.margin,BF.margin,2*BF.margin]
            
            #------------------------------------------------------------------
            # Value of the random stump bases on the current and the next state
            sgnState     = MVC.addVars(RNG_BF_NUMBER,
                                       ub    = [ upBoundInProd for _ in RNG_BF_NUMBER],
                                       lb    = [-upBoundInProd for _ in RNG_BF_NUMBER],
                                       vtype = CONTINUOUS)
            sgnNState    = MVC.addVars(RNG_BF_NUMBER,
                                       ub    = [ upBoundInProd for _ in RNG_BF_NUMBER],
                                       lb    = [-upBoundInProd for _ in RNG_BF_NUMBER],
                                       vtype = CONTINUOUS)
            
            #------------------------------------------------------------------
            # Unit: A random unit vector definifn an stump basis, e.g.,
            #       e(0,0,...0,1,0,...,0).
            # THR:  Threshold of random stumps
            UNIT    = BF.rndUnitVect
            THR     = BF.threshold_list
            
            #------------------------------------------------------------------
            # Gorubi functions
            PWL     = MVC.setPWLObj
            CTR     = MVC.addConstr
            
            #------------------------------------------------------------------
            # Adding the random bases
            for i in RNG_BF_NUMBER:
                #--------------------------------------------------------------
                # Value of signum on the current state [-sgn(x) = sgn(-x)]
                PWL(sgnState[i], x=X, y=[-BF_Coef[i],-BF_Coef[i],BF_Coef[i],BF_Coef[i]])
                CTR(sgnState[i] + THR[i] == LinExpr(UNIT[i],[state[j]  for j in self.RNG_DIM_X]))
                
                #--------------------------------------------------------------
                # Value of signum on the next state [-sgn(x) = sgn(-x)]
                PWL(sgnNState[i],x=X,y=[-BF_Coef[i],-BF_Coef[i],BF_Coef[i],BF_Coef[i]])
                CTR(sgnNState[i] + LinExpr(UNIT[i],[nState[j] for j in self.RNG_DIM_X]) ==  THR[i])
     
     
        #----------------------------------------------------------------------
        # Subset Constraints
        MVC.addConstr(quicksum(Y[_] for _ in self.RNG_POW_SET) == 1.0)
        
        MVC.addConstrs(R[i] == quicksum(Y[j] for j in whereIsElementInPowerSet(self.powSet,i)) \
                                    for i in self.RNG_DIM_X)
     
        MVC.addConstrs(act[i] <= self.mdp.invUppBounds[i]*R[i] for i in self.RNG_DIM_X)
        
        #----------------------------------------------------------------------
        # State-actions Constraints
        MVC.addConstrs(nState[i]  + self.mdp.consumptionRate[i]*t == state[i] + act[i] \
                                for i in self.RNG_DIM_X)
         
        MVC.addConstrs(state[i] + act[i] <= self.mdp.invUppBounds[i] for i in self.RNG_DIM_X)
         
        MVC.addConstr(quicksum(act[i] for i in self.RNG_DIM_X) <= self.mdp.maxOrder)
        
        #----------------------------------------------------------------------
        # Just-in-time Constraints
        MVC.addConstrs(state[i] + U[i]*self.mdp.invUppBounds[i] <= self.mdp.invUppBounds[i] for i in self.RNG_DIM_X)
        
        MVC.addConstr(quicksum(U[i] for i in self.RNG_DIM_X) >= 1.0) 
         
        MVC.addConstrs(nState[i] +self.mdp.invUppBounds[i]*Up[i]
                                  <= self.mdp.invUppBounds[i] for i in self.RNG_DIM_X)  
     
        MVC.addConstr(quicksum(Up[i] for i in self.RNG_DIM_X) >= 1.0)
     
        MVC.addConstrs(U[i] <= R[i] for i in self.RNG_DIM_X)  
        
        #----------------------------------------------------------------------
        # Optimize this MILP
        MVC.update()
        MVC.optimize()

        #----------------------------------------------------------------------
        # Make Output Ready
        MVC_state    = np.asarray([round(state[i].X,5)      for i in self.RNG_DIM_X])
        MVC_NState   = np.asarray([round(nState[i].X,5)     for i in self.RNG_DIM_X])
        MVC_action   = np.asarray([round(act[i].X,5)        for i in self.RNG_DIM_X])
        sgnOptState  = BF.evalBasisList(MVC_state)
        sgnOptNState = BF.evalBasisList(MVC_NState)

        return MVC_state, MVC_action, MVC.objVal,MVC_NState,sgnOptState,sgnOptNState
       

    """ 
        Computing Greedy Actions w.r.t. A Computed VFA 
    """      
    def getGreedyAction(self,state_ini, BF ,BF_Number,intercept, linCoef, BF_Coef):
        #----------------------------------------------------------------------
        # Configuring the Gurobi model
        RNG_BF_NUMBER = range(BF_Number)
        PD = gb.Model('PD')
        PD.setParam('OutputFlag',False)
        PD.setParam('LogFile',self.mdp.Adrr+'/groubiLogFile.log')
        unlink('gurobi.log')
        PD.setParam('Threads',self.numThreads)      
        PD.setParam('FeasibilityTol',1e-5)
        PD.setParam('IntFeasTol',1e-5)
        PD.setParam('NumericFocus',3)
    
        #----------------------------------------------------------------------
        # Defining some ranges and some list of indices
        horizon             = self.ub['roleHorizon']
        rangeHorizon        = range(horizon)
        rangeHorizon_1      = range(horizon-1)
        idx                 = [(r,d) for r in rangeHorizon   for d in self.RNG_DIM_X]
        idx_1               = [(r,d) for r in rangeHorizon_1 for d in self.RNG_DIM_X]
        psIdx               = [(r,i) for r in rangeHorizon_1 for i in self.RNG_POW_SET]
    
        #----------------------------------------------------------------------
        # Time upper bound
        upT    = INFINITY
        
        #----------------------------------------------------------------------
        # Vector of transition time decision variables
        t      = PD.addVars(rangeHorizon_1, 
                            ub      = [upT for _ in rangeHorizon_1],
                            lb      = [0   for _ in rangeHorizon_1],
                            vtype   = CONTINUOUS)
        
        #----------------------------------------------------------------------
        # Vector of action decision variables
        act    = PD.addVars(idx_1,
                            ub      = [self.mdp.invUppBounds[d] for (r,d) in idx_1],
                            lb      = [0        for _ in idx_1],
                            vtype   = CONTINUOUS)

        #----------------------------------------------------------------------
        # Vector of next state decision variables             
        nState = PD.addVars(idx,   
                            ub      = [self.mdp.invUppBounds[d] for (r,d) in idx],
                            lb      = [0 for _ in idx],
                            vtype   = CONTINUOUS)
        
        #----------------------------------------------------------------------
        # Auxiliary decision variables that are modeling the MDP transition 
        # function and replenishment time     
        Y      = PD.addVars(psIdx, vtype = BINARY)
        U      = PD.addVars(idx,   vtype = BINARY)    
        R      = PD.addVars(idx_1, vtype = BINARY)              
       
        #----------------------------------------------------------------------
        # Vector decision variables coding value of random stumps on the next state  
        upBoundInProd = np.max(self.mdp.invUppBounds) + BF.bandWidth_UB
        sgnNState     = PD.addVars(RNG_BF_NUMBER, ub = [ upBoundInProd for _ in RNG_BF_NUMBER],
                                                  lb = [-upBoundInProd for _ in RNG_BF_NUMBER],
                                                  vtype=CONTINUOUS)  
        
        #----------------------------------------------------------------------
        """ 1) Set Objective Function: Linear Part   """
        PD.setObjective(quicksum(
                            quicksum(self.mdp.getFixCost(self.powSet[i])*Y[r,i] for i in self.RNG_POW_SET) - \
                            intercept*t[r] - quicksum(linCoef[i]*act[r,i] for i in self.RNG_DIM_X) \
                                for r in rangeHorizon_1), MINIMIZE)

        #----------------------------------------------------------------------
        # Last state of time horizon (e.g., K-step greedy policy) variable
        varLastState = [nState[horizon-1,i] for i in self.RNG_DIM_X]
       
        #----------------------------------------------------------------------
        # 4 points that are defining the PWL approximation of sign function
        X = [-2*BF.margin,-BF.margin,BF.margin,2*BF.margin]
        
        #----------------------------------------------------------------------
        """ 2) Set Objective Function: Random Stumps : Picewise Linear Objective """
        for i in RNG_BF_NUMBER:
              PD.setPWLObj(sgnNState[i],x=X,y=[-BF_Coef[i],-BF_Coef[i],BF_Coef[i],BF_Coef[i]])
              PD.addConstr(sgnNState[i] + BF.threshold_list[i]== gb.LinExpr(BF.rndUnitVect[i],varLastState))
        PD.update()
        
        #----------------------------------------------------------------------
        # Subset Constraints 
        PD.addConstrs(quicksum(Y[r,i] for i in self.RNG_POW_SET) == 1.0 for r in rangeHorizon_1)
        
        PD.addConstrs(R[r,i] == quicksum(Y[r,j] for 
                                         j in whereIsElementInPowerSet(self.powSet,i)) for \
                          i in self.RNG_DIM_X for r in rangeHorizon_1)
             
        PD.addConstrs(act[r,i] <= self.mdp.invUppBounds[i]*R[r,i] for i in self.RNG_DIM_X for \
                                              r in rangeHorizon_1)
       
        #----------------------------------------------------------------------
        # State-actions Constraints
        PD.addConstrs(nState[0,i] == float(state_ini[i]) for i in self.RNG_DIM_X)
        PD.addConstrs(nState[r+1,i] + self.mdp.consumptionRate[i]*t[r] == nState[r,i] + act[r,i] \
                                      for i in self.RNG_DIM_X for r in rangeHorizon_1)
       
        PD.addConstrs(nState[r,i] + act[r,i] <= self.mdp.invUppBounds[i] for i in
                                        self.RNG_DIM_X for r in rangeHorizon_1)
             
        PD.addConstrs((quicksum(act[r,i] for i in self.RNG_DIM_X)) <= self.mdp.maxOrder for r in rangeHorizon_1)
           
        #----------------------------------------------------------------------
        # Just-in-time Constraints 
        PD.addConstrs(nState[r,i] +U[r,i]*self.mdp.invUppBounds[i] <= self.mdp.invUppBounds[i] \
                          for r in rangeHorizon for i in self.RNG_DIM_X)
        PD.addConstrs(quicksum(U[r,i] for i in self.RNG_DIM_X) >= 1.0 for r in rangeHorizon)
    
        PD.addConstrs(U[r,i] <= R[r,i] for i in self.RNG_DIM_X for r in rangeHorizon_1)  
       
        #----------------------------------------------------------------------
        # Optimize this MILP
        PD.update()            
        PD.optimize()
        return [act[0,i].X for i in self.RNG_DIM_X]
    
    
    """ 
        Simulating policy and computing upper bound
    """      
    def getUpperBound(self,state_ini, BF,BF_Number ,intercept, linCoef, BF_Coef):
        #----------------------------------------------------------------------
        # List of state, action, and transition time 
        visitedStates           = []
        visitedActions          = []
        visitedTimes            = []
        
        #----------------------------------------------------------------------
        # Initial state
        state                   = state_ini
        
        #----------------------------------------------------------------------
        # Does a cyclic policy found? Save index of the beginning of the cycle
        cyclicPolFound          = False
        cycleStartStateIndex    = 0
        
        #----------------------------------------------------------------------
        # Total time and cost 
        cumCost                 = 0
        totalTime               = 0
        
        #----------------------------------------------------------------------
        # Iterate over the trajectory length
        for traj in range(self.ub['trajLen']):
            #------------------------------------------------------------------
            # Get the optimal action 
            optAct  = self.getGreedyAction(state, BF,BF_Number ,intercept, linCoef, BF_Coef)
            
            #------------------------------------------------------------------
            # Get the transition time 
            t       = self.mdp.transTime(state,optAct)
            
            #------------------------------------------------------------------
            # Store state, action, and transition time
            visitedStates.append(state)
            visitedActions.append(optAct)
            visitedTimes.append(t)  
            
            #------------------------------------------------------------------
            state   =   np.add(np.add(state,optAct), -t*self.mdp.consumptionRate)
                            
            #------------------------------------------------------------------
            # Is there any cycle?
            ifCycle = [np.linalg.norm(state-x,ord=np.inf)<1e-14 for x in visitedStates]
            if  any(ifCycle):
                cyclicPolFound          = True  
                cycleStartStateIndex    = ifCycle.index(True)
                break
        
        #----------------------------------------------------------------------
        # Iterate over the cycle and compute the cost of the cycle
        for _ in range(cycleStartStateIndex,len(visitedStates)):
            totalTime += visitedTimes[_]
            cumCost   += self.mdp.getExpectedCost(visitedStates[_],visitedActions[_])
        
        #----------------------------------------------------------------------
        # Set the average time
        if abs(totalTime) <1e-6:
            AveCost = float('inf')
        else:
            AveCost = (cumCost/totalTime)

        #----------------------------------------------------------------------
        # Return the results 
        return cyclicPolFound,  len(visitedStates) - cycleStartStateIndex    , AveCost

     





