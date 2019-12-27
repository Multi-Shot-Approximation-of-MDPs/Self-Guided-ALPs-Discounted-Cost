"""
###############################################################################
# Created: Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
#                           | http://business.uic.edu/faculty/parshan-pakiman
#                          
# Licensing Information: The MIT License
###############################################################################
"""
import gurobipy as grb
from os import unlink

"""
    This function solves FALP with random basis functions.
"""
def solve_randomized_ALP(CPU_CORE,iPath,BF_number,objectiveVector,constraintMatrix,RHSVector):
    
    #--------------------------------------------------------------------------
    # Setting up the Gurobi model and parameters.
    ALP = grb.Model("SG-ALP")
    ALP.setParam('OutputFlag',False)
    ALP.setParam('LogFile',iPath+'/groubiLogFile.log')
    unlink('gurobi.log')
    ALP.setParam('FeasibilityTol',1e-9)
    ALP.setParam('Presolve',1)
    ALP.setParam('Threads',CPU_CORE)   
    
    #--------------------------------------------------------------------------
    # An auxiliary range and an add variable function.
    Rng         = range(BF_number)
    ALPvar      = ALP.addVars(Rng,
                         lb     = -grb.GRB.INFINITY,
                         ub     =  grb.GRB.INFINITY,
                         vtype  =  grb.GRB.CONTINUOUS)
    varToList   = [ALPvar[b] for b in Rng]
    add         = ALP.addConstr
    lexp        = grb.LinExpr
    
    #--------------------------------------------------------------------------
    # Setup FALP objective function.
    ALP.setObjective(grb.LinExpr(objectiveVector,varToList),grb.GRB.MAXIMIZE)
    
    #--------------------------------------------------------------------------
    # Add FALP constraints.
    for c in range(len(constraintMatrix)):
        add( lexp(constraintMatrix[c,:],varToList) <= RHSVector[c])
        
    #--------------------------------------------------------------------------
    # Solve FALP & return Gurobi model, optimal value, and optimal solution.
    ALP.optimize()
    return ALP, ALP.objVal, [ALPvar[b].X for b in Rng]

"""
    This function resolves an already created Gurobi model with a different 
    objective function.
"""
def resolve_randomized_ALP(ALP,objVec):
    #--------------------------------------------------------------------------
    # Load variables of the existing model
    ALPvar = ALP.getVars()
    varToList = [ALPvar[b] for b in range(len(ALPvar))]
    
    #--------------------------------------------------------------------------
    # Set the new objective function & reoptimize.
    ALP.setObjective(grb.LinExpr(objVec,varToList),grb.GRB.MAXIMIZE)
    ALP.optimize()
    
    #--------------------------------------------------------------------------
    # Solve ALP & return Gurobi model, optimal value, and optimal solution.
    return ALP, ALP.objVal, [ALPvar[b].X for b in range(len(ALPvar))]   

"""
    This function solves an FGLP.
"""
def solve_Feature_Guided_ALP(ALP,ALPNumConstraint,curVFAEval,selfGuidedLB,VFA_UB,objVec = None):
    #--------------------------------------------------------------------------
    # Setting up the Gurobi model and parameters.   
    ALP.setParam('FeasibilityTol',1e-9)
    ALP.setParam('Presolve',1)
    
    #--------------------------------------------------------------------------
    # Loading variables of an existing model, e.g. FALP without self-guiding
    # constrains.
    ALPvar      = ALP.getVars()
    varToList   = [ALPvar[b] for b in range(len(ALPvar))]
    
    #--------------------------------------------------------------------------
    # Some auxiliaryfunctions.
    lexp        = grb.LinExpr
    add         = ALP.addConstr
    
    #--------------------------------------------------------------------------
    # Add self-guiding constrains.
    for c in range(ALPNumConstraint):    
        
        #-----------------------------------------------------------------------
        # Consider constraints:
        #     V_cur <= c(s,a) + gamma*E[V_cur]  ---> VFA_UB
        #     V_cur >= V_prev      
        # If VFA_UB[c] > selfGuidedLB[c]: add sself-guiding constrains.
        if selfGuidedLB[c] >=0 and VFA_UB[c] >=0:
            if(VFA_UB[c] > selfGuidedLB[c]):
                add(lexp(curVFAEval[c],varToList) >= selfGuidedLB[c])  
        elif selfGuidedLB[c] <=0 and VFA_UB[c] <=0:
            if(VFA_UB[c] > selfGuidedLB[c]):
                add(lexp(curVFAEval[c],varToList) >= selfGuidedLB[c]) 
        elif  VFA_UB[c] >0 and selfGuidedLB[c] <0:
                add(lexp(curVFAEval[c],varToList) >= selfGuidedLB[c])  
                
    #--------------------------------------------------------------------------
    # Set the FGLP objective if there is any. If not, use the already objective in Gurobi model ALP.
    if objVec is not None:
        ALP.setObjective(grb.LinExpr(objVec,varToList),grb.GRB.MAXIMIZE)    
    
    #--------------------------------------------------------------------------
    # Solve ALP & return optimal value and optimal solution.
    ALP.optimize()
    return  ALP.objVal, [ALPvar[b].X for b in range(len(ALPvar))]   