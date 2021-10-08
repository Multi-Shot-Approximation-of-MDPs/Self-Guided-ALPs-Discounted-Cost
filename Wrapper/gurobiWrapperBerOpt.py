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
from scipy.linalg import svd
import numpy as np
import gc


class gurobi_LP_wrapper(lin_prog_wrapper):
    #-------------------------------------------------------------------------------
    # Gurobi wrapper implementation for finite-time FALP
    #-------------------------------------------------------------------------------
    def __init__(self,solver_conf):
        #-------------------------------------------------------------------------------
        # Initialization
        #-------------------------------------------------------------------------------
        super().__init__(solver_conf)
        self.num_cpu_core                       = self.solver_conf['num_cpu_core']
        self.num_stages                         = self.solver_conf['num_stages'] 
        self.basis_func_batch_size              = self.solver_conf['batch_size'] 
        self.ALP                                = gb.Model()
        self.ALP.setParam('OutputFlag',         False)
        self.ALP.setParam('LogFile',            'Output/groubi_log_file.log')
        self.ALP.setParam('Threads',            self.num_cpu_core) 
        self.ALP.setParam('Seed',               333)
        self.ALP_var                            = [None for _ in range(self.num_stages-1)]
        self.cur_VFA                            = [None for _ in range(self.num_stages-1)]
        self.expct_next_VFA                     = [None for _ in range(self.num_stages-1)]
        self.counter                            = 0
        self.abs_val_upp_bound                  = self.solver_conf['abs_val_upp_bound']
        self.self_guide_constr                  = []
        self.box_constr_list_1                  = []
        self.box_constr_list_2                  = []
        
    
    def set_objective(self,obj_coef,is_maximum = True,warm_start=None):
        #-------------------------------------------------------------------------------
        # Set the objective function of Gurobi model
        #-------------------------------------------------------------------------------
        if not warm_start is None:
            for t in range(self.num_stages-1):
                for _,val in enumerate(warm_start[:,t]):
                    self.ALP_var[t][_].start = val

        if is_maximum:
            self.ALP.setObjective(obj_coef@self.ALP_var[0],GRB.MAXIMIZE)
        else:
            self.ALP.setObjective(obj_coef@self.ALP_var[0],GRB.MINIMIZE)

    
    def get_optimal_value(self):
        #-------------------------------------------------------------------------------
        # Get optimal objective value of Gurobi model
        #-------------------------------------------------------------------------------
        return self.ALP.objVal
    
    def get_optimal_solution(self):
        #-------------------------------------------------------------------------------
        # Get optimal solution of Gurobi model
        #-------------------------------------------------------------------------------
        return asarray([asarray(self.ALP_var[t].X) for t in range(self.num_stages-1)]).T

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

    def re_initialize_solver(self):
        #-------------------------------------------------------------------------------
        # Rest Gurobi model
        #-------------------------------------------------------------------------------
        self.ALP.remove(self.ALP.getConstrs())
        self.self_guide_constr = []
        self.ALP.update()
        self.counter+=1
        gc.collect()
      
    def prepare(self):
        #-------------------------------------------------------------------------------
        # Prepare Gurobi model before being solved
        #-------------------------------------------------------------------------------
        self.ALP.update()
    
    def optimize(self,num_times_basis_added=0):
        #-------------------------------------------------------------------------------
        # Optimize Gurobi model
        #-------------------------------------------------------------------------------
        self.ALP.setParam('Method',             2)
        self.ALP.setParam('BarConvTol',         1e-6)
        self.ALP.setParam('Crossover',          0)
        self.ALP.setParam('OptimalityTol',      1e-8)
        self.ALP.setParam('FeasibilityTol',     1e-6)

        for t in range(self.num_stages-1):
            x = self.ALP_var[t]
            x = x[1:len(x.tolist())]        
            self.ALP.addConstr( x  <=  self.abs_val_upp_bound)
            self.ALP.addConstr(-x  <=  self.abs_val_upp_bound)
    
        self.ALP.update()    
        self.ALP.optimize()
        
    def get_num_var_constr(self):
        #-------------------------------------------------------------------------------
        # Get # of variables and constraints of a model
        #-------------------------------------------------------------------------------
        return self.ALP.NumVars, self.ALP.NumConstrs
    
    
    def set_up_variables(self,num_basis,stage):
        #-------------------------------------------------------------------------------
        # Set up variables of Gurobi model
        #-------------------------------------------------------------------------------
        self.ALP_var[stage]     = self.ALP.addMVar(shape   =  num_basis,
                                                    lb     = -GRB.INFINITY,
                                                    ub     =  GRB.INFINITY,
                                                    vtype  =  GRB.CONTINUOUS)


    def add_new_variables(self,new_num_basis,stage):
        #-------------------------------------------------------------------------------
        # Add new variables to an exisiting Gurobi model
        #-------------------------------------------------------------------------------
        assert stage <= self.num_stages-2
        new_ALP_var         = self.ALP.addMVar( shape  =  new_num_basis,
                                                lb     = -GRB.INFINITY,
                                                ub     =  GRB.INFINITY,
                                                vtype  =  GRB.CONTINUOUS)
        
        for _ in range(new_num_basis):
            new_ALP_var[_].start = 0.0
        if self.ALP_var[stage] is None:
            self.ALP_var[stage] = new_ALP_var
        else:   
            self.ALP_var[stage] = gb.MVar(self.ALP_var[stage].tolist() + new_ALP_var.tolist())
        
        self.ALP.update()
            
    
    def set_terminal_val(self,terminal_val):
        #-------------------------------------------------------------------------------
        # Set the objective function of Gurobi model
        #-------------------------------------------------------------------------------
        self.terminal_val = terminal_val

    def add_FALP_constr(self,new_cur_VFA, new_expct_next_VFA,ALP_RHS,stage,warm_start = None):
        #-------------------------------------------------------------------------------
        # Add new constraints to an exisiting Gurobi model
        #-------------------------------------------------------------------------------
        assert stage <= self.num_stages-2
        if not new_expct_next_VFA is None:
            knocked_out         = np.where(~new_cur_VFA.any(axis=1))[0]
            new_cur_VFA         = np.delete(new_cur_VFA, knocked_out, axis = 0)
            new_expct_next_VFA  = np.delete(new_expct_next_VFA, knocked_out, axis = 0)
            ALP_RHS             = np.delete(ALP_RHS, knocked_out, axis = 0)
            terminal            = np.delete(self.terminal_val, knocked_out, axis = 0)
            
        else:
            
            knocked_out         = np.where(~new_cur_VFA.any(axis=1))[0]
            new_cur_VFA         = np.delete(new_cur_VFA, knocked_out, axis = 0)
            ALP_RHS             = np.delete(ALP_RHS, knocked_out, axis = 0)
            terminal            = np.delete(self.terminal_val, knocked_out, axis = 0)
       
        if self.cur_VFA[stage] is None:
            self.cur_VFA[stage]             = new_cur_VFA @ self.ALP_var[stage]             
            if stage == self.num_stages-2:
                self.expct_next_VFA[stage]  = terminal
            else:
                self.expct_next_VFA[stage]  = new_expct_next_VFA @ self.ALP_var[stage+1]  
        
        else:
            self.cur_VFA[stage]             = self.cur_VFA[stage] + new_cur_VFA @ self.ALP_var[stage][range(self.counter*self.basis_func_batch_size, 
                                                                                                               (1+self.counter)*self.basis_func_batch_size)]    
            
            if stage == self.num_stages-2:
                self.expct_next_VFA[stage]      =  terminal
            else:
                self.expct_next_VFA[stage]      =  self.expct_next_VFA[stage]  + new_expct_next_VFA @ self.ALP_var[stage+1][range(self.counter*self.basis_func_batch_size, 
                                                                                                                   (1+self.counter)*self.basis_func_batch_size)]

            
        if stage == self.num_stages-2:
            self.ALP.addConstr(self.cur_VFA[stage]  >=  np.maximum(ALP_RHS, self.expct_next_VFA[stage]))
        else:
            self.ALP.addConstr(self.cur_VFA[stage]  >= ALP_RHS)
            self.ALP.addConstr(self.cur_VFA[stage]  >= self.expct_next_VFA[stage])

        self.ALP.update()

    def update_RHS(self,new_ALP_RHS):
        #-------------------------------------------------------------------------------
        # Update right-hand-side of FALP constriants
        #-------------------------------------------------------------------------------
        for _, constr in enumerate(self.ALP.getConstrs()):
            constr.RHS = new_ALP_RHS[_]
        self.ALP.update()

    def incorporate_self_guiding_constraint(self,self_guiding_constr_matrix,RHS_self_guiding_constr,stage): 
        #-------------------------------------------------------------------------------
        # Add guiding constraints to an exisiting Gurobi model
        #-------------------------------------------------------------------------------
        assert stage <= self.num_stages-2
        allowable_violation         = .02*np.abs(RHS_self_guiding_constr)  
        nonzero_loc                 = np.nonzero(RHS_self_guiding_constr)
        LHS                         = self_guiding_constr_matrix[nonzero_loc,:]
        LHS                         = np.squeeze(LHS, axis=0)@self.ALP_var[stage] #
        RHS                         = RHS_self_guiding_constr[nonzero_loc] + allowable_violation[nonzero_loc]
                
        constr = self.ALP.addConstr(LHS <= RHS,name='SG')
        self.self_guide_constr.append(constr)
        self.ALP.update()
        gc.collect()
                
    def infeasbile_report(self):
        #-------------------------------------------------------------------------------
        # Store some info if Gurobi model is infeasbile
        #-------------------------------------------------------------------------------
        self.ALP.computeIIS()
        self.ALP.write("Output/gurobi_infeasbile_report.ilp") 
