"""
###############################################################################
# Created: Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
#                           | http://business.uic.edu/faculty/parshan-pakiman
#                          
# Licensing Information: The MIT License
###############################################################################
"""
import time
import numpy as np
import gurobipy as gb
from BasisFunction.hatBasisFunctions import HatBasis
from scipy.stats import randint
from itertools import chain, combinations
from gurobipy import *
from utils import printWithPickle,powerset,whereIsElementInPowerSet,arreqclose_in_list, dropZeros,AK_Header
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

"""
    This following class implements the ALgorithm in
    Adelman, Daniel, and Diego Klabjan. "Computing near-optimal policies in
    generalized joint replenishment." INFORMS Journal on Computing 24.1 (2012):
    148-164.
"""
class GJR_Benchmark():

    """
    The constructor of the class that initializes an instance encapsulated in
    expInfo.
    """
    def __init__(self,expInfo):
        self.mdp            = expInfo['mdp']['type'](expInfo['mdp'])
        self.bf             = expInfo['bf']
        self.ub             = expInfo['ub']
        self.runTime        = expInfo['misc']['runTime']
        self.dimX           = self.mdp.dimX
        self.numThreads     = self.mdp.Threads 
        self.transTime      = self.mdp.transTime
        self.powSet         = powerset(range(0,self.dimX))
        self.powSetLen      = len(self.powSet)
        self.RNG_DIM_X      = range(self.dimX)
        self.RNG_POW_SET    = range(self.powSetLen)
        self.OMEGA          = np.ceil(np.sqrt(self.dimX*np.linalg.norm(self.mdp.invUppBounds,ord=2)))
    
    def AK_GJR_Algorithm(self,trial):
        
        #----------------------------------------------------------------------
        # Defining the ridge basis functions
        hatSetting              = self.initHatSetting()    
        MVC_StateList, \
            MVC_ActionList, \
                MVC_NStateList  = self.loadStateActionsForFeasibleALP(trial)
        
        #----------------------------------------------------------------------
        # Unit and pair ridge vectors defined in AK 2012.        
        numB_Unit               = 0
        numB_Pair               = 0
        numB_Unit_init          = 0
        numB_Pair_init          = 0
        numNewRidgeAdded        = 0
        basis_Itr               = 0
        basisNum                = 0
        cost                    = float('inf')
        cycleLength             = float('inf')
        ALP_MODEL               = None
        BF                      = None
        useVFA                  = False    
        bestUB                  = float('inf')
        initialLB               = -float('inf')
        initialUB               = float('inf')
        runTimeLimit            = self.runTime
        isDualCyclic            = False
        
        #----------------------------------------------------------------------
        for i in hatSetting['indexUnitVec']:
            numB_Unit += hatSetting['numBreakPts'][i]
        for i in  list(set(range(hatSetting['numRidgeVec']))- set( hatSetting['indexUnitVec'])):
            numB_Pair += hatSetting['numBreakPts'][i]
            
        numB_Unit_init          = numB_Unit
        numB_Pair_init          = numB_Pair
            
        s = [np.random.uniform(low=0.0,high=self.mdp.invUppBounds[_]) for _ in range(self.dimX)]
        s[np.random.randint(self.dimX)]=0.0
        s = np.asarray(s)
        initialState = s      
            
        
        """ Setup Loggers """
        
        logger = open(self.mdp.Adrr+ '/summary_trial_'+str(trial)+".txt","w+")
        timeLogger = open(self.mdp.Adrr+ '/time_gap_list_'+str(trial)+".txt","w+")        
        timeLogger.write('{:^5} | {:^15} | {:^15} | {:^15} |'.format('B', 'GAP (%)', 'Iter-RT (s)', 'Cum-RT (s)'))
        timeLogger.write("\n") 
        
        resTable = np.zeros(shape = (5000,14))
        timeGapTable  = np.zeros(shape = (5000,4))
        cumTime       = 0    
        
        
        AK_Header(self.mdp.mdpName,"Generalized Joint Replenishment",148)
        printWithPickle('{:^3} | {:^15} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^5} | {:^5} | {:^5} | {:^5} | {:^5} | {:^5} | {:^8} |'.format(
                                'B','VIOL','I-LB','LB','I-UB','UB','I-GAP','GAP','CYC?','C-LEN','MAX-B','R','#b U', '#b P', 'T(s)'),logger)
        printWithPickle('-'*148,logger)
        # print('{:{fill}^{w}}'.format('Adelman & Klabjan 2012 \t ',fill=' ',w=148))
        # print('\n \n{:{fill}^{w}}'.format(mdp['mdp_name'],fill=' ',w=148))
    
        
        
        stillAddConstr = False
        """ Iterate Until an Eps-Opt Policy Will Be Find"""
        while True:
            I =0
            # Track Ill Condition Situation
            prevS = np.asarray([-float('inf') for _ in range(self.mdp.dimX)])
            prevA = np.asarray([-float('inf') for _ in range(self.mdp.dimX)])
            illCondBasisCount=0
            
            start_time =  time.time()
            """ Generate Rows """
            while True:
                
                if BF == None:
                    useVFA = False
                else:
                    useVFA = True
                    
                
                """ SOLVE ALP TO GET VFA """
                if not stillAddConstr:
                    #print('1')
                    intercept , linCoef , BF_Coef , ALPval, dual,ALP_MODEL = \
                        self.ALP_GJR(BF,\
                            stateList=MVC_StateList, actionList= MVC_ActionList, nStateList =\
                                MVC_NStateList, useVFA = useVFA, basisItr = basis_Itr,\
                                model = None,addedS = None, addedA= None)  
                else:
    
                    intercept , linCoef , BF_Coef , ALPval, dual,ALP_MODEL = \
                        self.ALP_GJR(BF,\
                            stateList=MVC_StateList, actionList= MVC_ActionList, nStateList =\
                                MVC_NStateList, useVFA = useVFA, basisItr = basis_Itr,
                                model=ALP_MODEL,addedS=prevS,addedA=prevA)                  
                    
                 
                    
     
                """ COMPUTE CONSTRAINT VIOLATION """
                s, a, MVCval,NS   = self.getMostViolatedConstraint(intercept, linCoef, BF_Coef,BF)  
    
                if basis_Itr ==0:
                    isALPConverged   = 1e-3
                else:
                    isALPConverged   = 0.001*bestUB
                
                
                # Constraint Violation 
                if  MVCval >= -isALPConverged:
                    stillAddConstr = False
                    printWithPickle('{:>3d} | {:>15.5f} | {} |'.format(basisNum,(MVCval),''.ljust(122)),logger)
                    
                    support, Q, flowViolation, isDualCyclic, costDual,cycleLengthDual = self.dualALP_GJR(dual,MVC_StateList,  MVC_ActionList, MVC_NStateList )
                    
                    
                    if isDualCyclic:
                        cost = costDual
                        cycleLength = cycleLengthDual
                            
                        cyclicPolFound,cycleLength,cost =  self.getUpperBound(state_ini=initialState, \
                                                                     BF=BF ,intercept=intercept,\
                                                                     linCoef=linCoef, BF_Coef=BF_Coef)
    
                        print(cost,costDual)
                       
                        bestUB = min(bestUB,cost)
                        
                        if basis_Itr == 0 :
                            initialLB = ALPval
                            initialUB = cost
                        
        
                        printWithPickle('{:>3d} | {:^15} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>5} | {:>5d} | {:>5d} | {:>5d} | {:>5d} | {:>5d} | {:>8.0f} |'.format(
                                    basisNum,''.ljust(15),initialLB,   ALPval,   initialUB,   bestUB, \
                                    (1-initialLB/initialUB)*100,       (1-ALPval/bestUB)*100,
                                    isDualCyclic,   cycleLength, basis_Itr,numNewRidgeAdded,numB_Unit,\
                                                            numB_Pair,cumTime),logger) 
                        printWithPickle('-'*148,logger)
                        
                        
                        logger.flush()
                        
                        basisNum=0
                        for j in range(hatSetting['numRidgeVec']):
                            basisNum += len(hatSetting['breakPoints'][j])
                        break
    
                    
                    else:
    
                        if basis_Itr%15 == 0 :
                      
                            cyclicPolFound,cycleLength,cost =  self.getUpperBound(state_ini=initialState, \
                                                                     BF=BF ,intercept=intercept,\
                                                                     linCoef=linCoef, BF_Coef=BF_Coef)
                        
                        bestUB = min(bestUB,cost)
                        
                        if basis_Itr == 0 :
                            initialLB = ALPval
                            initialUB = cost
                        
                        
    
                        printWithPickle('{:>3d} | {:^15} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>5} | {:>5d} | {:>5d} | {:>5d} | {:>5d} | {:>5d} | {:>8.0f} |'.format(
                                    basisNum,''.ljust(15),initialLB,   ALPval,   initialUB,   bestUB, \
                                    (1-initialLB/initialUB)*100,       (1-ALPval/bestUB)*100,
                                    isDualCyclic,   cycleLength, basis_Itr,numNewRidgeAdded,numB_Unit,\
                                                            numB_Pair,cumTime),logger)  
                        printWithPickle('-'*148,logger)
                    
    
                        
                    resTable[basis_Itr,0]   = trial
                    resTable[basis_Itr,1]   = basisNum
                    resTable[basis_Itr,2]   = initialLB
                    resTable[basis_Itr,3]   = ALPval
                    resTable[basis_Itr,4]   = initialUB
                    resTable[basis_Itr,5]   = cost
                    resTable[basis_Itr,6]   = cycleLength
                    resTable[basis_Itr,7]   = (1-initialLB/initialUB)*100
                    resTable[basis_Itr,8]   = (1-ALPval/cost)*100
                    resTable[basis_Itr,9]   = (1-initialLB/ALPval)*100
                    resTable[basis_Itr,10]  = (1-cost/initialUB)*100
                    resTable[basis_Itr,11]  = resTable[basis_Itr,7] - resTable[basis_Itr,8]
                    resTable[basis_Itr,12]  = numB_Unit 
                    resTable[basis_Itr,13]  = numB_Pair 
                    break
                
                # Violation Is Large
                else:
                    if I%50==0:
                        printWithPickle('{:>3d} | {:>15.5f} | {} |'.format(basisNum,(MVCval),''.ljust(122)),logger)           
                        
                    if np.linalg.norm(s-prevS,ord=np.inf) < 1e-50 and np.linalg.norm(a-prevA,ord=np.inf) < 1e-50:
                        #printWithPickle('{:>3d} | {:>15.5f} | {} |'.format(basisNum,(MVCval),''.ljust(122)),logger) 
                        illCondBasisCount +=1
                        I = np.random.randint(self.mdp.dimX)
                                            
                        
                         
                        if illCondBasisCount >=100:
                            printWithPickle('------> ALP DOES NOT CUT OFF MVC ACTION <------',logger)                        
                            break
                            #return outPut(-1,logger,timeLogger,resTable,timeGapTable)
                    
                    prevS = s
                    prevA = a
                
                    MVC_StateList.append(s)
                    MVC_ActionList.append(a)
                    MVC_NStateList.append(NS)
                       
                    stillAddConstr = True  
                    
                I+=1
                logger.flush()
                
        
            basisNum=0
            for j in range(hatSetting['numRidgeVec']):
                basisNum += len(hatSetting['breakPoints'][j])-4
            
            """ Colecting Time """
            GAP = (1-ALPval/bestUB)*100
            runTime = time.time() - start_time
            cumTime += runTime
            
            
            
            if cumTime >= runTimeLimit:
                printWithPickle('TIME OUT! gap = {} and runtime = {}'.format(GAP,cumTime),logger)
            
                timeLogger.write("Time Out!\n")
                return self.outPut(0,logger,timeLogger,resTable,timeGapTable)      
            
            
            timeLogger.write('{:>5d} | {:>15.2f} | {:>15.2f} | {:>15.2f} |'.format(basisNum,GAP,runTime,cumTime))
            timeLogger.write("\n")  
            timeLogger.flush()
            
            
            
        
            timeGapTable[basis_Itr,0] = basisNum
            timeGapTable[basis_Itr,1] = GAP
            timeGapTable[basis_Itr,2] = runTime
            timeGapTable[basis_Itr,3] = cumTime
            
            # 2% Optimal Policy Is Found
            if  GAP<= 2.00 or isDualCyclic:
                printWithPickle('{:>3d} | {:^15} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>5} | {:>5d} | {:>5d} | {:>5d} | {:>5d} | {:>5d} | {:>8.0f} |'.format(
                                    hatSetting['numRidgeVec'],''.ljust(15),initialLB,   ALPval,   initialUB,   bestUB, \
                                    (1-initialLB/initialUB)*100,       (1-ALPval/bestUB)*100,
                                    isDualCyclic,   cycleLength, basis_Itr,numNewRidgeAdded,numB_Unit,\
                                                            numB_Pair,cumTime),logger)  
                
                printWithPickle('The algorithm is converged with the terminal optimality gap of ({:>8.3} %) in ({:>8.3} m).'.format(GAP,cumTime/60),logger)
                return self.outPut(0,logger,timeLogger,resTable,timeGapTable)
            
            # More Bases Is Needed
            else: 
                basis_Itr +=1
    #            if basis_Itr == 3:
    #                sys.exit()
                
                
                hatSetting,BF, numNewRidge = self.getNewHatBases(hatSetting,basis_Itr,support, Q, flowViolation)
                numNewRidgeAdded += numNewRidge    
                
                stillAddConstr = False
                
                numB_Unit = -numB_Unit_init
                numB_Pair = -numB_Pair_init
                for i in hatSetting['indexUnitVec']:
                    numB_Unit += hatSetting['numBreakPts'][i]
                for i in  list(set(range(hatSetting['numRidgeVec']))- set( hatSetting['indexUnitVec'])):
                    numB_Pair += hatSetting['numBreakPts'][i]



    def initHatSetting(self):
        hatSetting      = {}
        unitVecs        = [self.getUnitVec(_) for  _ in range(self.dimX)]
        pairVecs        =  self.getPairVec()
        
        hatSetting.update({'dimX'           : self.dimX})
        hatSetting.update({'diamX'          : np.linalg.norm(self.mdp.invUppBounds,2)})
        hatSetting.update({'BF_num'         : None})
        hatSetting.update({'ridgeVector'    : unitVecs+pairVecs })     
        hatSetting.update({'numRidgeVec'    : len(hatSetting['ridgeVector'])})
        hatSetting.update({'indexUnitVec'   : range(self.dimX)})
        hatSetting.update({'indexPairVec'   : range(self.dimX, self.dimX+len(pairVecs))})
        hatSetting.update({'breakPoints'    : [self.getInitBreakPts(hatSetting,j) for j in range(len(hatSetting['ridgeVector']))]})
        hatSetting.update({'numBreakPts'    : [len(hatSetting['breakPoints'][_])-2 for _ in range(hatSetting['numRidgeVec'])] })
        
        return hatSetting

    """ 
    Ridge basis functions defined in [AK12] algorithm
    """
    def getUnitVec(self,i):
            z = np.zeros(self.dimX)
            z[i] = 1.0
            return z
    
    def getPairVec(self):
        z = []
        for i in range(self.dimX):
            for j  in range(i+1, self.dimX):
                zero = np.zeros(self.dimX)
                zero[i] =   1/self.mdp.consumptionRate[i]
                zero[j] =  -1/self.mdp.consumptionRate[j]
                z.append(zero)
        return z 


    # Initializing breaking points of ridge bases based on the [AK12]
    def getInitBreakPts(self,hatSetting,j):
        breakPoint  = gb.Model('left')
        breakPoint.setParam('OutputFlag',False)
        breakPoint.setParam('LogFile','Output/GJR/groubiLogFile.log')
        unlink('gurobi.log')
        allBreakPts = []
        
        """
            The following problem is defined on page 160 [AK12]:
            This corresponds with two hat functions, one centered at the leftmost point of
            the domain and the other centered at the rightmost
            point of the domain.
        """    
        r       = hatSetting['ridgeVector'][j]
        x       = [breakPoint.addVar(lb = 0.0,vtype = CONTINUOUS) for _ in range(len(r))]
        xMin    = breakPoint.addVar(lb = 0.0,vtype = CONTINUOUS)
        breakPoint.setObjective(gb.LinExpr(r,x),MINIMIZE)
        breakPoint.addConstrs(x[i] <= self.mdp.invUppBounds[i] for i in range(len(r)))
        breakPoint.addGenConstrMin(xMin, x)
        breakPoint.addConstr(xMin == 0)
        breakPoint.optimize()
        
        mini = breakPoint.objVal
        
        breakPoint.setObjective(gb.LinExpr(r,x),MAXIMIZE)
        breakPoint.optimize()
        maxi = breakPoint.objVal
            
        # b_left
        allBreakPts.append(-self.OMEGA)
        #  middle 
        allBreakPts.append(mini)
        allBreakPts.append(maxi)
        #b_right
        allBreakPts.append(self.OMEGA)
        
        allBreakPts = list(np.sort(allBreakPts, kind = 'mergesort'))
        return allBreakPts












# # Define minor/major cost functions
# def getFixCost(itemSubset):    
#     minorCost = mdp['minorFixCost']
#     majorCost = mdp['majorFixCost']
#     if len(itemSubset)==0:
#         return 0
#     return majorCost + sum(minorCost[_] for _ in itemSubset)

# def getCost(state,action):    
#     idx = []
#     for i in range(len(action)):
#         if not action[i]==0:
#             idx.append(i)
#     c = getFixCost(tuple(idx))       
#     return c

# # Transition time of semi-MDP  
# def transTime(s,a):
#     return np.min([(s[_]+a[_])/mdp['consumptionRate'][_] for _ in range(mdp['dimX'])])

# # Transition function 
# def getNextState(s,a):
#     Z = np.asarray([(s[_]+a[_])/mdp['consumptionRate'][_] for _ in range(mdp['dimX'])])
#     Zmin = Z.min()
#     ns =np.add(np.add(s,a),-transTime(s,a)*mdp['consumptionRate'])
    
#     # Avoiding cancellation error: one of ns coordinates is zero.    
#     for _ in range(len(ns)):
#         if abs(Z[_] - Zmin) < 1e-4:
#             ns[_] = 0.0
            
#     return np.round(ns,5)


           




    """ Finding the most violated constraint for a given VFA """   
    def getMostViolatedConstraint(self,intercept, linCoef, BF_Coef,BF):
            
             MVC = gb.Model('MVC')
             MVC.setParam('OutputFlag',False)
             MVC.setParam('LogFile','Output/GJR/groubiLogFile.log')
             unlink('gurobi.log')
             MVC.setParam('Threads',self.numThreads) 
             #MVC.setParam('MIPGap'   , 0.05) 
             MVC.setParam('FeasibilityTol',1e-9)
             MVC.setParam('IntFeasTol',1e-9)
             MVC.setParam('NumericFocus',3)
             
             
             """ Variables    """
            
             t      = MVC.addVar(ub = INFINITY,lb = 0,vtype=CONTINUOUS) 
             
             act    = MVC.addVars(self.RNG_DIM_X, ub = [self.mdp.invUppBounds[_] for _ in self.RNG_DIM_X],
                                                     lb = [0        for _ in self.RNG_DIM_X],
                                             vtype=CONTINUOUS)
                                  
             state  = MVC.addVars(self.RNG_DIM_X, ub = [self.mdp.invUppBounds[_] for _ in self.RNG_DIM_X],
                                             lb = [0        for _ in self.RNG_DIM_X],
                                             vtype=CONTINUOUS)  
             
             nState = MVC.addVars(self.RNG_DIM_X, ub = [self.mdp.invUppBounds[_] for _ in self.RNG_DIM_X],
                                             lb = [0        for _ in self.RNG_DIM_X],
                                             vtype=CONTINUOUS)
             
             Y      = MVC.addVars(self.RNG_POW_SET,vtype   = BINARY) 
             
             U      = MVC.addVars(self.RNG_DIM_X,vtype     = BINARY)        
             
             Up     = MVC.addVars(self.RNG_DIM_X,vtype     = BINARY)               
             
             R      = MVC.addVars(self.RNG_DIM_X,vtype     = BINARY)              
             
             
             """ 1) Set Objective Function: Linear Part   """
             MVC.setObjective(quicksum(self.mdp.getFixCost(self.powSet[i])*Y[i] for i in self.RNG_POW_SET) - \
                                  intercept*t - quicksum(linCoef[i]*act[i] for i in self.RNG_DIM_X),
                                  MINIMIZE)
            
             addVar     = MVC.addVar
             linExpr    = gb.LinExpr
             PWL        = MVC.setPWLObj
             addConstrs = MVC.addConstrs        
             Svec  = [state[i]  for i in self.RNG_DIM_X]
             NSvec = [nState[i] for i in self.RNG_DIM_X]
                        
    
             if not BF == None:
                 """ 2) Set Objective Function: Stump Basis Functions -- Picewise Linear Objective """
                
                 Hat_State  = [addVar(ub =    INFINITY,
                                      lb =   -INFINITY,
                                      vtype = CONTINUOUS)  for j in range(BF.numRidgeVec)]
        
                 Hat_NState = [addVar(ub =    INFINITY,
                                      lb =   -INFINITY,
                                      vtype = CONTINUOUS)  for j in range(BF.numRidgeVec)]
                 
                 for j in range(BF.numRidgeVec): 
                    VFA = [0.0]
                    for i in range(BF.numBreakPts[j]+2):
                        if i <= BF.numBreakPts[j]-1:
                            if i ==0  or i ==  BF.numBreakPts[j]-1:
                                VFA.append(0.0)
                            elif abs(BF_Coef[j][i]) < 1e-4:
                                VFA.append(0.0)
                            else:
                                VFA.append(BF_Coef[j][i])
                    VFA.append(0.0)
    
                    
                    PWL(Hat_State[j], x=BF.breakPoints[j]    ,y= VFA)                    
                    PWL(Hat_NState[j],x=BF.breakPoints[j]    ,y= [-VFA[i] for i in range(len(VFA))])
                    
                 addConstrs(Hat_State[j]  == linExpr( BF.ridgeVector[j],Svec) for j in range(BF.numRidgeVec))
                 addConstrs(Hat_NState[j] == linExpr( BF.ridgeVector[j],NSvec) for j in range(BF.numRidgeVec))
                    
    
    
             """ Subset Constraints """
             MVC.addConstr(quicksum(Y[_] for _ in self.RNG_POW_SET) == 1.0)
             
             MVC.addConstrs(R[i] == quicksum(Y[j] for j in whereIsElementInPowerSet(self.powSet,i)) \
                                            for i in self.RNG_DIM_X)
             
             MVC.addConstrs(act[i] <= self.mdp.invUppBounds[i]*R[i] for i in self.RNG_DIM_X)
                 
             """ State-actions Constraints """
             MVC.addConstrs(nState[i]  + self.mdp.consumptionRate[i]*t == state[i] + act[i] \
                                    for i in self.RNG_DIM_X)
             
             MVC.addConstrs(state[i] + act[i] <= self.mdp.invUppBounds[i] for i in self.RNG_DIM_X)
             
             MVC.addConstr(quicksum(act[i] for i in self.RNG_DIM_X) <= self.mdp.maxOrder)
             
             """ Just-in-time Constraints """
             ### MVC.addConstrs(state[i] <= mdp['invUppBounds'][i]*(1-U[i]) for i in RNG_DIM_X)
             MVC.addConstrs(state[i] + U[i]*self.mdp.invUppBounds[i] <= self.mdp.invUppBounds [i] for i in self.RNG_DIM_X)
             
             MVC.addConstr(quicksum(U[i] for i in self.RNG_DIM_X) >= 1.0)
    
             ### MVC.addConstrs((nState[i] <= mdp['invUppBounds'][i]*(1-Up[i])) for i in RNG_DIM_X)  
             MVC.addConstrs(nState[i] +self.mdp.invUppBounds[i]*Up[i]
                                     <= self.mdp.invUppBounds[i] for i in self.RNG_DIM_X)  
             
             MVC.addConstr(quicksum(Up[i] for i in self.RNG_DIM_X) >= 1.0)
             
             MVC.addConstrs(U[i] <= R[i] for i in self.RNG_DIM_X)  
             
             """ Optimize The Program """
             MVC.update()
             MVC.optimize()
             
             """ Make Output Ready """
             MVC_state    = np.round([dropZeros(state[i].X )     for i in self.RNG_DIM_X],5)
             MVC_action   = np.round([dropZeros(act[i].X)         for i in self.RNG_DIM_X],5)
             MVC_NState   = self.mdp.getNextState(MVC_state,MVC_action) #np.round([dropZeros(nState[i].X)         for i in RNG_DIM_X],5)#
             
             return MVC_state, MVC_action, MVC.objVal, MVC_NState


    """ ALP Model in AK 12 """   
    def ALP_GJR(self,BF,  stateList, actionList, nStateList, 
                    useVFA = True, basisItr = 0, model = None,addedS = None, addedA= None): 
    
        if model == None:  
            
            getCost= self.mdp.getExpectedCost
            pool = Pool(self.numThreads)
            COST= pool.starmap(getCost, zip(stateList,actionList))
            pool.close()
            pool.join()
            
            transTime = self.mdp.transTime
            pool = Pool(self.numThreads)
            TRAN_TIME= pool.starmap(transTime, zip(stateList,actionList))
            pool.close()
            pool.join()
            
            ALP = gb.Model('ALP')
            ALP.setParam('OutputFlag',False)
            ALP.setParam('LogFile','Output/GJR/groubiLogFile.log')
            unlink('gurobi.log')
            ALP.setParam('Threads',self.numThreads) 
            ALP.setParam('NumericFocus',3)
            ALP.setParam('FeasibilityTol',1e-9)
            
        
            """ Variables """
            intercept  = ALP.addVar(ub =  INFINITY, lb  = -INFINITY, vtype= CONTINUOUS,
                                    name = 'intercept')
            linCoefVar = ALP.addVars(self.RNG_DIM_X, ub = [ INFINITY for _ in self.RNG_DIM_X],
                                                lb = [-INFINITY for _ in self.RNG_DIM_X],
                                                vtype=CONTINUOUS,
                                                name = 'linCoefVar')
            
            linCoef   = [linCoefVar[i] for i in self.RNG_DIM_X]
            ALP.setObjective(intercept + gb.LinExpr(self.mdp.consumptionRate,linCoef), MAXIMIZE)  
                
    
            # Consider Model with basis functions other than affine
            if useVFA:
                numRidgeVec  = BF.numRidgeVec
                numBreakPts  = BF.numBreakPts
                RNG_BF_NUMBER  = [(j,i) for j in range(numRidgeVec) for i in range(numBreakPts[j])]
                
                BF_CoefVar = [[ALP.addVar(ub    =  INFINITY,  lb    = -INFINITY ,
                                          vtype = CONTINUOUS, name  = 'BF_CoefVar') \
                                                for i in range(numBreakPts[j])]
                                                    for j in range(numRidgeVec)]
            
                
                for j in range(numRidgeVec):
                    ALP.addConstr(BF_CoefVar[j][0]  == 0.0)
                    ALP.addConstr(BF_CoefVar[j][numBreakPts[j]-1]  == 0.0)
        
                """ ALP constraints """    
                delta=BF.deltaHat
                dual = ALP.addConstrs(TRAN_TIME[itr]*intercept +\
                                        gb.LinExpr(actionList[itr],linCoef) +\
                                        quicksum(BF_CoefVar[j][i]*delta(j,i,nStateList[itr],stateList[itr]) for (j,i) in RNG_BF_NUMBER) \
                                        <= COST[itr]  for itr in range(len(stateList))) 
        
                """ Solve ALP & Return Opt Coefs """
                ALP.update() 
                ALP.optimize()
                optVal = ALP.objVal
                
                DUAL_VAL = [dual[i].getAttr('Pi')     for i in  range(len(stateList))]
                
                
                        
                return intercept.X, \
                             [linCoef[_].X for _ in self.RNG_DIM_X], \
                                 [[BF_CoefVar[j][i].X for i in range(numBreakPts[j])] for j in range(numRidgeVec)],\
                                     optVal, DUAL_VAL,ALP               
            else:       
                    
                """ ALP constraints """
        
                dual = ALP.addConstrs(TRAN_TIME[itr]*intercept +\
                                       gb.LinExpr(actionList[itr],linCoef) \
                                         <=  COST[itr] for itr in range(len(stateList)))  
                
                """ Solve ALP & Return Opt Coefs """          
                ALP.update()       
                ALP.optimize() 
                intercept.X 
                
                DUAL_VAL = [(dual[i].getAttr('Pi'))     for i in  range(len(stateList))]
                
                
                return intercept.X, \
                        [linCoef[_].X for _ in self.RNG_DIM_X], \
                        [ ],\
                        ALP.objVal, DUAL_VAL,ALP
        else:
            model.setParam('FeasibilityTol',1e-9)
            model.setParam('NumericFocus',3)
            if not BF == None:
                numRidgeVec  = BF.numRidgeVec
                numBreakPts  = BF.numBreakPts
                RNG_BF_NUMBER  = [(j,i) for j in range(numRidgeVec) for i in range(numBreakPts[j])]
                
                vars = model.getVars()
                
                intercept =  vars[0]
                linCoef   = [vars[_+1] for _ in self.RNG_DIM_X]
                
                BF_Coef   = [vars[_+1+self.dimX] for _ in range(len(RNG_BF_NUMBER))]  
                BF_CoefVar = [[None for i in range(numBreakPts[j])] for j in range(numRidgeVec)]
                
                k = 0
                for j in range(numRidgeVec):
                    for i in range(numBreakPts[j]):
                        BF_CoefVar[j][i] = BF_Coef[k]
                        k+=1
                        
                
                #print(BF_CoefVar)
                
                
                for j in range(numRidgeVec):
                    model.addConstr(BF_CoefVar[j][0]  == 0.0)
                    model.addConstr(BF_CoefVar[j][numBreakPts[j]-1]  == 0.0)
            
                addedNs = self.mdp.getNextState(addedS,addedA)
                """ ALP constraints """    
                delta=BF.deltaHat
                model.addConstr(self.mdp.transTime(addedS,addedA)*intercept +\
                                        gb.LinExpr(addedA,linCoef) +\
                                        quicksum(BF_CoefVar[j][i]*delta(j,i,addedNs,addedS) for (j,i) in RNG_BF_NUMBER) \
                                        <= self.mdp.getExpectedCost(addedS,addedA)) 
                
                model.update() 
                model.optimize()
                optVal = model.objVal
                dual = model.getConstrs()
                DUAL_VAL = [dual[i].getAttr('Pi')     for i in  range(len(stateList))]
                
                return intercept.X, \
                         [linCoef[_].X for _ in self.RNG_DIM_X], \
                             [[BF_CoefVar[j][i].X for i in range(numBreakPts[j])] for j in range(numRidgeVec)],\
                                 optVal, DUAL_VAL,model 
                
            else:
                vars = model.getVars()
                intercept =  vars[0]
                linCoef   = [vars[_+1] for _ in self.RNG_DIM_X]
       
                """ ALP constraints """    
                model.addConstr(self.transTime(addedS,addedA)*intercept +\
                                        gb.LinExpr(addedA,linCoef) 
                                        <= self.mdp.getExpectedCost(addedS,addedA)) 
                model.update() 
                model.optimize()
                optVal = model.objVal
                dual = model.getConstrs()
                DUAL_VAL = [dual[i].getAttr('Pi')     for i in  range(len(stateList))]
    
                return intercept.X, \
                        [linCoef[_].X for _ in self.RNG_DIM_X], \
                        [ ],\
                        optVal, DUAL_VAL,model
            

        
        
              
                                 
                                 
        

    """ Dual ALP Model  """   
    def dualALP_GJR(self,Dual, stateList, actionList, nStateList ):
        """  Support Set of Z """
        support = []
        for (i,x) in enumerate(Dual):
            if x >  1e-8:
                support.append([i, list(stateList[i]), list(actionList[i]), Dual[i]])   
    
        """ States Visited Under Dual Solution """
        Q = []
        for (i,s,a,z_s_a) in support:
            s = np.asarray(s)
            a = np.asarray(a)
            nS= np.asarray(self.mdp.getNextState(s,a))
            if not arreqclose_in_list(s,Q):
                Q.append(s)
            if not arreqclose_in_list(nS,Q):
                Q.append(nS)
        
        """ Flow Imbalance Calculation    """
        flowViolation = []
        for x in Q:
            flow_out  = 0.0
            flow_in   = 0.0
            for (d,s,a,z_s_a) in support: 
                if np.allclose(x,np.asarray(s),atol=1e-8):
                    flow_in  += z_s_a  
                    
                if np.allclose(x,self.mdp.getNextState(s,a),atol=1e-8):
                    flow_out += z_s_a
    
            flowViolation.append([x,abs(flow_in-flow_out)])
            
        flowViolation = sorted(flowViolation,key = lambda x: x[1],reverse=True)
        Q = [flowViolation[i][0] for i in range(len(Q))]
        
    
        """ Cyclic Schedule Detection """
        def isCyclic(lst):
            l = len(lst)
            nS = self.mdp.getNextState(np.asarray(lst[l-1][1]),np.asarray(lst[l-1][2]))
            
            if not np.allclose(np.asarray(lst[0][1]),nS,atol = 1e-4):
                return False
            
            for i in range(0,l-1):
                ns=np.asarray(self.mdp.getNextState(lst[i][1],lst[i][2]))
                if not np.allclose(ns,np.asarray(lst[i+1][1]) ,atol = 1e-4):
                    return False
                
            return True    
        
        
        # Cycle detetion 
        cycLen = float('inf')  
        isDualCyclic = False
        I = -1
        for (i,y,a,z_y_a) in support:
            J = -1
            I += 1
            for (j,x,u,z_x_u) in support:
                J +=1
                if j > i:
                    subSupport = [support[_] for _ in range(I,J)]
                    if  len(subSupport) >  1:
    
                        if isCyclic(subSupport) :
                            isDualCyclic = True
                            cycLen = len(subSupport)
                            break   
            if isDualCyclic:
                break
        
        # Cost calculation
        aveCost       = float('inf')  
        if isDualCyclic:
            totalTime=0.0
            cumCost = 0.0
            for (i,x,b,z_y_a) in subSupport:
                cumCost += self.mdp.getCost(x,b)
                totalTime += self.mdp.transTime(x,b)                    
    
            aveCost = cumCost/totalTime        
            
        
        return support, Q, flowViolation, isDualCyclic, aveCost,cycLen
    
    
    
    def getUniqueMaps(self,hatSetting,Q,q,idxSet):
        U = []
        for i in idxSet:
            r = hatSetting['ridgeVector'][i]
            val = dropZeros(np.dot(r,Q[q]))
            flag = True
            for k in range(len(Q)):
                if not q == k:
                    vval = dropZeros(np.dot(Q[k],r))
                    if abs(val - vval) <1e-6:
                        flag = False
                      
            if flag:
                 U.append(i)             
        return U
     
    def getNewHatBases(self,hatSetting,basis_Itr,support, Q, flowViolationSorted):    
        
        U_unit       = None
        U_non_unit   = None
        ridgeNeeded  = 0
        lenQ         = len(Q)
        
        
        K = 0
        for q in range(lenQ):            
    
            if K>=9:
                break
            
            E = hatSetting['indexUnitVec']
            O = list(set(range(hatSetting['numRidgeVec']))- set(E))  
            
            U_unit     = self.getUniqueMaps(hatSetting,Q,q,E)
            U_non_unit = self.getUniqueMaps(hatSetting,Q,q,O)
            
            if not U_unit == []:
                idx = U_unit[randint(0,len(U_unit)).rvs(1)[0]]
                prevBrPts = list(hatSetting['breakPoints'][idx])
                
                br = round(np.dot(hatSetting['ridgeVector'][idx],Q[q]),5)
                delta = [abs(br - _) for _ in prevBrPts]
                
                if min(delta)>0.1:
                    prevBrPts.append(br)
                    prevBrPts = np.unique(prevBrPts)
                    prevBrPts = list(np.sort(prevBrPts, kind = 'mergesort'))
                    hatSetting['breakPoints'][idx]     = prevBrPts        
                    hatSetting.update({'numBreakPts'   : [len(hatSetting['breakPoints'][_])-2 for _ in range(hatSetting['numRidgeVec'])] })
                    K+=1
                
            elif not U_non_unit == []:
                idx = U_non_unit[randint(0,len(U_non_unit)).rvs(1)[0]]
                prevBrPts = list(hatSetting['breakPoints'][idx])
                
                br = round(np.dot(hatSetting['ridgeVector'][idx],Q[q]),5)
                delta = [abs(br - _) for _ in prevBrPts]
               
                if min(delta)>0.1:
                    prevBrPts.append(br)
                    prevBrPts = np.unique(prevBrPts)
                    prevBrPts = list(np.sort(prevBrPts, kind = 'mergesort'))
                    hatSetting['breakPoints'][idx]     = prevBrPts
                    hatSetting.update({'numBreakPts'   : [len(hatSetting['breakPoints'][_])-2 for _ in range(hatSetting['numRidgeVec'])] }) 
                    K+=1
            
            else:
                
                ridgeNeeded+=1
    
                E = hatSetting['indexUnitVec']
                O = list(set(range(hatSetting['numRidgeVec']))- set(E))
                
                """ Page 160 Adelman & Klabjan 2012"""
                idx = list(set(range(lenQ)) - set([q]))
                
                ridgeModel = gb.Model('findRidge')
                ridgeModel.setParam('OutputFlag',False)
                ridgeModel.setParam('LogFile','Output/GJR/groubiLogFile.log')
                unlink('gurobi.log')
                ridgeModel.setParam('NumericFocus',3)
                ridgeModel.setParam('FeasibilityTol',1e-9)
                ridgeModel.setParam('MIPGap',0.00)
                
                theta = ridgeModel.addVar(lb = 0.0,
                                          ub = INFINITY,
                                          vtype = CONTINUOUS)
                
                alpha = ridgeModel.addVars(idx,
                                           lb=[-INFINITY for _ in idx],
                                           ub=[ INFINITY for _ in idx],
                                           vtype=CONTINUOUS)
                
                alphaAbs = ridgeModel.addVars(idx,
                                           lb=[0.0       for _ in idx],
                                           ub=[ INFINITY for _ in idx],
                                           vtype=CONTINUOUS)
                
                beta  = ridgeModel.addVars(idx,
                                           lb=[-INFINITY for _ in idx],
                                           ub=[ INFINITY for _ in idx],
                                           vtype=CONTINUOUS)
                
                betaAbs = ridgeModel.addVars(idx,
                                           lb=[ 0.0      for _ in idx],
                                           ub=[ INFINITY for _ in idx],
                                           vtype=CONTINUOUS)
                
                maxAbs = ridgeModel.addVars(idx,
                                           lb=[ 0.0      for _ in idx],
                                           ub=[ INFINITY for _ in idx],
                                           vtype=CONTINUOUS)
                
                
                rVec  = [ridgeModel.addVar(lb=-1.0,ub=1.0,vtype=CONTINUOUS) for _ in range(self.mdp.dimX)]
                
                ridgeModel.setObjective(theta,MAXIMIZE)
                
                fixVec=np.asarray(Q[q])
                for qq in idx:
                    
                    ridgeModel.addConstr(theta   <= alpha[qq] +beta[qq])
                    ridgeModel.addConstr(gb.LinExpr(np.asarray(Q[qq])-fixVec,rVec) \
                                                 <= alpha[qq] - beta[qq])
                    # alpha*beta = 0
                    ridgeModel.addConstr(alphaAbs[qq] == gb.abs_(alpha[qq]))
                    ridgeModel.addConstr(betaAbs[qq]  == gb.abs_(beta[qq] ))
                    ridgeModel.addConstr(maxAbs[qq]   == gb.max_(alphaAbs[qq],betaAbs[qq]))
                    ridgeModel.addConstr(alphaAbs[qq] + betaAbs[qq] <= maxAbs[qq])
                    
                ridgeModel.update()
                ridgeModel.optimize()
                
                optRidge = np.round([rVec[_].X for _ in range(self.mdp.dimX)],5)
                
                # Page 157 Adelman & Klabjan 
                ALPHA = []
                ZETA  = []
                val = np.dot(optRidge,Q[q])
                for d in Q:
                    if np.around(np.dot(optRidge,d),5)   < val +1e-2 :
                            ALPHA.append(np.round(np.dot(optRidge,d),5) )
                    if np.around(np.dot(optRidge,d),5)    +1e-2 > val :
                            ZETA.append(np.around(np.dot(optRidge,d),5) )
                        
                Br_center= val
                
            
                if ALPHA == [] and ZETA  == []:
                    Break = [-self.OMEGA,    Br_center,  self.OMEGA]
                    
                elif ALPHA == [] and (not ZETA  == []):
                    
                    if abs(Br_center -   min(ZETA)) > 0.1:
                        Break = [-self.OMEGA,    Br_center,  min(ZETA),  self.OMEGA]
                    
                    else:
                        Break = [-self.OMEGA,    Br_center,  self.OMEGA]
                        
                elif (not ALPHA == []) and (not ZETA  == []):
                    
                    if abs(max(ALPHA) -  Br_center) > 0.1 and abs(min(ZETA) -  Br_center)  > 0.1:
                        Break = [-self.OMEGA,    max(ALPHA) , Br_center,  min(ZETA), self.OMEGA]
                        
                    elif abs(max(ALPHA) -  Br_center) < 0.1 and abs(min(ZETA) -  Br_center) >0.1:
                        Break = [-self.OMEGA, Br_center , min(ZETA) , self.OMEGA]
                    
                    elif abs(max(ALPHA) -  Br_center) > 0.1 and abs(min(ZETA) -  Br_center) <0.1:
                            Break = [-self.OMEGA,  max(ALPHA) , Br_center, self.OMEGA]
                            
                    else:
                        Break = [-self.OMEGA,    Br_center,  self.OMEGA]
                        
                elif (not ALPHA  == []) and ZETA  == []:
                    if abs(Br_center -   max(ALPHA)) > 0.1:
                        Break = [-self.OMEGA, np.around(max(ALPHA)) ,   Br_center,  self.OMEGA]
                    else:
                        Break = [-self.OMEGA,    Br_center,  self.OMEGA]
                
    
                
                prevRidge = hatSetting['ridgeVector']
                if not arreqclose_in_list(np.asarray(optRidge),prevRidge):
                    
                    prevRidge.append(np.asarray(optRidge))
                    
                    Break = list(np.round(np.sort(Break, kind = 'mergesort'),5))
                    K+=1
               
                    hatSetting.update({'ridgeVector'    : prevRidge})
                    hatSetting.update({'numRidgeVec'    : len(hatSetting['ridgeVector'])})
                                
                    hatSetting['breakPoints'].append(Break)
                    hatSetting.update({'numBreakPts'   : [len(hatSetting['breakPoints'][_])-2 for _ in range(hatSetting['numRidgeVec'])] })  
        
                    E = hatSetting['indexUnitVec']
                    O = list(set(range(hatSetting['numRidgeVec']))- set(E))
                    
                    if self.getUniqueMaps(hatSetting,Q,q,E) == [] and self.getUniqueMaps(hatSetting,Q,q,O)==[]:
                         print('------> Z* did not cut off & ridge has been skipped <------',optRidge)
                     
    
            
        return  hatSetting, HatBasis(hatSetting), ridgeNeeded



    """ Computing Greedy Actions w.r.t. A Computed VFA """       
    def getGreedyAction(self,state_ini, BF ,intercept, linCoef, BF_Coef):
        
        PD = gb.Model('PD')
        PD.setParam('OutputFlag',False)
        PD.setParam('LogFile','Output/GJR/groubiLogFile.log')
        unlink('gurobi.log')
        PD.setParam('MIPGap',0.01)
        PD.setParam('Threads',self.numThreads)
        PD.setParam('FeasibilityTol',1e-9)
        PD.setParam('NumericFocus',3)     
        
        
        
        H     = self.ub['roleHorizon']
        rH    = range(H)
        rH_1  = range(H-1)
        
        idx   = [(r,d) for r in rH for d in         self.RNG_DIM_X]
        idx_1 = [(r,d) for r in rH_1 for d in       self.RNG_DIM_X]
        psIdx = [(r,i) for r in rH_1 for i in       self.RNG_POW_SET]
    
        """ Variables """
        upT    = INFINITY
        t      = PD.addVars(rH_1, ub = [upT for _ in rH_1],
                                  lb = [0        for _ in rH_1],
                                  vtype = CONTINUOUS) 
             
        act    = PD.addVars(idx_1, ub = [self.mdp.invUppBounds[d] for (r,d) in idx_1],
                                   lb = [0        for _ in idx_1],
                                   vtype=CONTINUOUS)
             
        nState = PD.addVars(idx,   ub = [self.mdp.invUppBounds[d] for (r,d) in idx],
                                   lb = [0 for _ in idx],
                                   vtype=CONTINUOUS)
        
        Y      = PD.addVars(psIdx, vtype = BINARY) 
             
        U      = PD.addVars(idx,   vtype = BINARY)     
             
        R      = PD.addVars(idx_1, vtype = BINARY)              
    
      
        """ Set Objective """    
        PD.setObjective(quicksum(
                            quicksum(self.mdp.getFixCost(self.powSet[i])*Y[r,i] for i in self.RNG_POW_SET) - \
                            intercept*t[r] - quicksum(linCoef[i]*act[r,i] for i in self.RNG_DIM_X) \
                                for r in rH_1), MINIMIZE)
       
        varLastState = [nState[H-1,_] for _ in self.RNG_DIM_X]
        
        
        if not BF == None:
             """ 2) Set Objective Function: Stump Basis Functions -- Picewise Linear Objective """
             addVar     = PD.addVar
             linExpr    = gb.LinExpr
             PWL        = PD.setPWLObj
             addConstrs = PD.addConstrs     
        
             Hat_NState = [addVar(ub =    INFINITY,
                                  lb =   -INFINITY,
                                  vtype = CONTINUOUS)  for j in range(BF.numRidgeVec)]
             
             for j in range(BF.numRidgeVec):
                B     = BF.breakPoints[j]    
                VFA = [0.0]
                for i in range(BF.numBreakPts[j]+2):
                    if i <= BF.numBreakPts[j]-1:
                        if abs(BF_Coef[j][i]) < 1e-4:
                            VFA.append(0.0)
                        else:
                            VFA.append(-BF_Coef[j][i])
                VFA.append(0.0)
                PWL(Hat_NState[j],x=B,y=VFA)
                
             addConstrs(Hat_NState[j] == linExpr(BF.ridgeVector[j],varLastState) for j in range(BF.numRidgeVec))
        
    
    #    if not BF == None:
    #        upBound1 = np.linalg.norm(mdp['invUppBounds'],2)
    #        upBound2 = upBound1*max([np.linalg.norm(BF.ridgeVector[j],ord = np.inf) for j in range(BF.numRidgeVec)])
    #                    
    #        upBound = INFINITY  #max(upBound1,upBound2)
    #        
    #        Hat_NState = [None for _ in range(BF.numRidgeVec)]
    #        for j in range(BF.numRidgeVec):
    #            Hat_NState[j] = [None for _ in range(BF.numBreakPts[j])]
    #            B     = BF.breakPoints[j]
    #            Ridge = BF.ridgeVector[j]
    #            for i in range(BF.numBreakPts[j]):                
    #                Hat_NState[j][i] = PD.addVar(ub  =   upBound,
    #                                              lb =  -upBound,
    #                                              vtype = CONTINUOUS)             
    #                
    #                X = [-upBound,B[i],B[i+1],B[i+2],upBound]
    #                PD.setPWLObj(Hat_NState[j][i],x=X,y=  [0,0, -BF_Coef[j][i],0,0])
    #                PD.addConstr(Hat_NState[j][i] == gb.LinExpr(Ridge,varLastState))
        
        PD.update()
             
        """ Subset Constraints """
        PD.addConstrs(quicksum(Y[r,i] for i in self.RNG_POW_SET) == 1.0 for r in rH_1)
             
        PD.addConstrs(R[r,i] == quicksum(Y[r,j] for j in whereIsElementInPowerSet(self.powSet,i)) for \
                                         i in self.RNG_DIM_X for r in rH_1)
             
        PD.addConstrs(act[r,i] <= self.mdp.invUppBounds[i]*R[r,i] for i in self.RNG_DIM_X for \
                                             r in rH_1)
        
        """ State-actions Constraints """
        PD.addConstrs(nState[0,i] == float(state_ini[i]) for i in self.RNG_DIM_X) 
        PD.addConstrs(nState[r+1,i] +self.mdp.consumptionRate[i]*t[r] == nState[r,i] + act[r,i] \
                                     for i in self.RNG_DIM_X for r in rH_1)
        
        PD.addConstrs(nState[r,i] + act[r,i] <= self.mdp.invUppBounds[i] for i in
                                        self.RNG_DIM_X for r in rH_1)
             
        PD.addConstrs((quicksum(act[r,i] for i in self.RNG_DIM_X)) <= self.mdp.maxOrder for r in rH_1)
             
        """ Just-in-time Constraints """
        PD.addConstrs(nState[r,i] +U[r,i]*self.mdp.invUppBounds[i] <= self.mdp.invUppBounds[i] \
                          for r in rH for i in self.RNG_DIM_X)
        PD.addConstrs(quicksum(U[r,i] for i in self.RNG_DIM_X) >= 1.0 for r in rH)
    
        PD.addConstrs(U[r,i] <= R[r,i] for i in self.RNG_DIM_X for r in rH_1)  
        
        """ Solve Greedy Problem & Return Opt Act """
        PD.update()             
        PD.optimize()
        
        return [act[0,i].X for i in self.RNG_DIM_X]


    """ Finding Most Violated Constraint For a Given VFA """       
    def getUpperBound(self,state_ini, BF ,intercept, linCoef, BF_Coef):
        visitedStates   = []
        visitedActions  = []
        visitedTimes    = []
        state = state_ini
        cyclicPolFound = False
        cumCost = 0
        totalTime = 0
        cycleStartStateIndex = 0
        
        for traj in range(self.ub['trajLen']): 
            
            optAct = self.getGreedyAction(state, BF ,intercept, linCoef, BF_Coef)
            t = self.mdp.transTime(state,optAct)
            visitedStates.append(state)
            visitedActions.append(optAct)
            visitedTimes.append(t)   
            state=np.add(np.add(state,optAct), -t*self.mdp.consumptionRate) 
                             
            R = [np.linalg.norm(state-x,ord=np.inf)<1e-14 for x in visitedStates]
            if  any(R):
                cyclicPolFound = True  
                cycleStartStateIndex = R.index(True)
                break
            
        for _ in range(cycleStartStateIndex,len(visitedStates)):
            totalTime += visitedTimes[_]
            cumCost   += self.mdp.getExpectedCost(visitedStates[_],visitedActions[_])
        
        
        if abs(totalTime) <1e-6:
            AveCost = float('inf')
        else:
            AveCost = (cumCost/totalTime)
        return cyclicPolFound,  len(visitedStates) - cycleStartStateIndex    , AveCost
    
    
    
            
    def loadStateActionsForFeasibleALP(self,trial):
        
        MVC_StateList  = np.load('Output/GJR/'+self.mdp.mdpName+'/SampleConstraints/TRIAL_'+str(trial)+'/MVC_StateList.npy')
        MVC_ActionList = np.load('Output/GJR/'+self.mdp.mdpName+'/SampleConstraints/TRIAL_'+str(trial)+'/MVC_ActionList.npy')
        MVC_NStateList = np.load('Output/GJR/'+self.mdp.mdpName+'/SampleConstraints/TRIAL_'+str(trial)+'/MVC_NStateList.npy')
        
        return list(MVC_StateList), list(MVC_ActionList),list(MVC_NStateList)
    
    
    
    
    def outPut(self,error,logger,timeLogger,resTable,timeGapTable):
        
        if error == 0:
            logger.close() 
            timeLogger.close()
            
            resTable = resTable[~ np.all(resTable==0, axis=1)]     
            timeGapTable = timeGapTable[~ np.all(timeGapTable==0, axis=1)]
            
            np.savetxt(self.mdp.Adrr+'/TIME_GAP_TABLE_AK.csv',timeGapTable,delimiter=',',
                                  header='B,GAP (%),Iter-RT(s),Cum-RT(s)')
        
        
            resTable= resTable[~ np.all(resTable==0, axis=1)]
            np.savetxt(self.mdp.Adrr+'/RESULTS_TABLE_SG.csv',resTable,delimiter=',',
                                  header='T,B,I-LB,LB,I-UB,UB,C-LEN,I-GAP(%),GAP(%),LB-IMP(%),UB-IMP(%),GAP-IMP(%), Unit, Pair')
            return 0, resTable
        else:
            return error, None






                
                
            

    


