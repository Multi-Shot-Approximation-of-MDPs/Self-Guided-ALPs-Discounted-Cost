# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/ 
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""
import gurobipy as gb
from gurobipy import GRB
from Wrapper.linProgWrapper import lin_prog_wrapper
import numpy as np
from numba import jit
from numpy import asarray
import time
from utils import prune_similar_columns

class gurobi_LP_wrapper(lin_prog_wrapper):
    #-------------------------------------------------------------------------------
    # Gurobi wrapper implementation
    #-------------------------------------------------------------------------------

    def __init__(self,solver_conf):
        #-------------------------------------------------------------------------------
        # Initialization
        #-------------------------------------------------------------------------------
        super().__init__(solver_conf)
        self.ALP            = gb.Model()
        self.num_cpu_core   = self.solver_conf['num_cpu_core']
        self.ALP.setParam('OutputFlag',False)
        self.ALP.setParam('LogFile','Output/groubi_log_file.log')
        self.ALP.setParam('Threads',self.num_cpu_core) 
        self.ALP.setParam('Seed',           333)
        self.ALP.setParam('FeasibilityTol', 1e-5)
        self.ALP.setParam('OptimalityTol',  1e-6)
        self.ALP.setParam('Method',         2)
        self.ALP.setParam('Crossover',      0)
        self.ALP_var                = None 
        self.dual_var_value         = None
        self.ALP_matrix             = None
        self.num_new_basis          = None
        self.SG_feasbile_sln_prev   = None 
        self.FALP_feasbile_sln_prev = None 
        
    def re_initialize_solver(self):
        #-------------------------------------------------------------------------------
        # Rest Gurobi model
        #-------------------------------------------------------------------------------
        self.ALP.remove(self.ALP.getConstrs())
        self.obj_coef               = None
       
    def prepare(self):
        #-------------------------------------------------------------------------------
        # Prepare Gurobi model before being solved
        #-------------------------------------------------------------------------------
        self.ALP.update()
    
    def optimize(self):
        #-------------------------------------------------------------------------------
        # Optimize Gurobi model
        #-------------------------------------------------------------------------------
        self.ALP.optimize()
        if self.ALP.status == GRB.OPTIMAL:
            self.FALP_feasbile_sln_prev = asarray(self.ALP.X)    
          
    def get_num_var_constr(self):
        #-------------------------------------------------------------------------------
        # Get # of variables and constraints of a model
        #-------------------------------------------------------------------------------
        return self.ALP.NumVars, self.ALP.NumConstrs
    
    
    def set_up_variables(self,num_basis):
        #-------------------------------------------------------------------------------
        # Set up variables of Gurobi model
        #-------------------------------------------------------------------------------
        self.ALP_var = self.ALP.addMVar(
                               shape  = num_basis,
                               lb     = -GRB.INFINITY,
                               ub     =  GRB.INFINITY,
                               vtype  =  GRB.CONTINUOUS)
        self.ALP.update()


    def add_new_variables(self,new_num_basis):
        #-------------------------------------------------------------------------------
        # Add new variables to an exisiting Gurobi model
        #-------------------------------------------------------------------------------
        new_ALP_var         = self.ALP.addMVar(
                                       shape  = new_num_basis,
                                       lb     = -GRB.INFINITY,
                                       ub     =  GRB.INFINITY,
                                       vtype  =  GRB.CONTINUOUS)
 
        self.num_new_basis  = new_num_basis
        for _ in range(new_num_basis):
            new_ALP_var[_].start = 0.0
        
        if self.ALP_var is None:
            self.ALP_var = new_ALP_var
        else:
            for _,val in enumerate(self.ALP_var.X):
                self.ALP_var[_].start = val
                
            self.ALP_var = gb.MVar(self.ALP_var.tolist() + new_ALP_var.tolist())
        
        self.ALP.update()
            
    def add_FALP_constraint(self,new_ALP_columns,ALP_RHS):
        #-------------------------------------------------------------------------------
        # Add new constraints to an exisiting Gurobi model
        #-------------------------------------------------------------------------------
        new_ALP_columns[np.abs(new_ALP_columns) < 1e-5] = 0.0

        if self.ALP_matrix is None:
            self.ALP_matrix = new_ALP_columns @ self.ALP_var
        else:            
            self.ALP_matrix = self.ALP_matrix  + new_ALP_columns @ self.ALP_var[range( self.ALP.NumVars - self.num_new_basis, self.ALP.NumVars)]

        self.ALP.addConstr( self.ALP_matrix <=  ALP_RHS)
        self.ALP.update()


    def incorporate_self_guiding_constraint(self,self_guiding_constr_matrix,RHS_self_guiding_constr):
        #-------------------------------------------------------------------------------
        # Add guiding constraints to an exisiting Gurobi model
        #-------------------------------------------------------------------------------
        self.ALP.setParam('OptimalityTol',  1e-8)
        self_guiding_constr_matrix[np.abs(self_guiding_constr_matrix) < 1e-5] = 0.0
        if not self.SG_feasbile_sln_prev is None:
            
            for _ in range(len(self.ALP_var.tolist())):
                if _ < len(self.SG_feasbile_sln_prev):
                    self.ALP_var[_].start = self.SG_feasbile_sln_prev[_]
                else:
                    self.ALP_var[_].start = 0.0
        

        RHS_self_guiding_constr = np.asarray(RHS_self_guiding_constr)
        allowable_violation     = .02*np.abs(RHS_self_guiding_constr) 
        self.ALP.addConstr(self_guiding_constr_matrix @ self.ALP_var + allowable_violation >=  RHS_self_guiding_constr)
        self.ALP.update()
        self.ALP.optimize()
        self.SG_feasbile_sln_prev = asarray(self.ALP.X)    
        return self.get_optimal_solution(), self.get_optimal_value() #self.obj_coef@primal_var_value


    def set_objective(self,obj_coef,is_maximum = True):
        #-------------------------------------------------------------------------------
        # Set the objective function of Gurobi model
        #-------------------------------------------------------------------------------
        self.obj_coef = obj_coef
        if is_maximum:
            self.ALP.setObjective(obj_coef@self.ALP_var,GRB.MAXIMIZE)
        else:
            self.ALP.setObjective(obj_coef@self.ALP_var,GRB.MINIMIZE)

    def get_optimal_value(self):
        #-------------------------------------------------------------------------------
        # Get optimal objective value of Gurobi model
        #-------------------------------------------------------------------------------
        return self.ALP.objVal
    
    def get_optimal_solution(self):
        #-------------------------------------------------------------------------------
        # Get optimal solution of Gurobi model
        #-------------------------------------------------------------------------------
        return asarray(self.ALP.X)
    
    def get_status(self):
        #-------------------------------------------------------------------------------
        # Get Gurobi status after optimization
        #-------------------------------------------------------------------------------
        status = self.ALP.status
        
        if status == GRB.INF_OR_UNBD:
            return "INF_OR_UNBD"
        elif status == GRB.UNBOUNDED:
            return "UNBOUNDED"
        elif status == GRB.INFEASIBLE:
            return "INFEASIBLE"
        elif status == GRB.OPTIMAL:
            return "OPTIMAL"
        else:
            return "UNKNOWN"
    
        
    def infeasbile_report(self):
        #-------------------------------------------------------------------------------
        # Store some info if Gurobi model is infeasbile
        #-------------------------------------------------------------------------------
        self.ALP.computeIIS()
        self.ALP.write("Output/gurobi_infeasbile_report.ilp")
