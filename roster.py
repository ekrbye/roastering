# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 18:22:20 2023

@author: wongy
"""

from default import *
from restraint import RestraintCpModel, RestraintCpSolver, RestraintCallback

from itertools import product
import pickle
import pandas as pd
from transpose import Transposer, Sheet


class Roster():
    def __init__(self):
        self.rRM = None
        self.rRS = None
        self.dict_dep = None
        self.dict_backsolution = {}
        self.dict_loggedfunctions = {}
        self.int_status = None
        self.bool_loadfunctions = False
        self.work_value = None
        
    def solve(self):
        if not self.bool_loadfunctions:
            self.logged_fuctions = self.rRM.get_funclog()
        self.status = self.rRS._solve()
        self.work_value, self.dict_dep = self.rRS.get_dictdep()
        
    def save_solution(self, filename: (None, str) =None) -> None:
        if filename is None: filename = "solution.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.work_value, f)
    
    def load_solution(self, filename: (None, str) =None) -> None:
        if filename is None: filename = "solution.pkl"
        with open(filename, "rb") as f:
            self.backsolution = pickle.load(f)
            
    
if __name__ == "__main__":
    rRM = RestraintCpModel(90, 11, 11)

    rRM.add_atleast_forall()

    for k, v in DICT_INDEXDEP_PDOMAIN_POS.items():
        rRM.add_pdomain_forall(k, v)

    for k, v in DICT_INDEXDEP_PDOMAIN_NEG.items():
        rRM.add_not_pdomain_forall(k, v, rearrange=True)
    
    rRM.add_depinterval_forall([i for i in range(rRM._int_depart)], 1, 1)
    rRM.sadd_numinaver(w=3)
    #rRM.add_restraint_fornum(5, 8, 0)
#    rRM.sadd_numinbound(lb=6, lbw=2)
    rRM.add_numindbound(lb=8, ub=15, depart=TPL_DEPART_Y12, period=TPL_Y12)
    rRM.add_numindbound(lb=8, ub=15, depart=TPL_DEPART_Y23, period=TPL_Y23)

    rRM.add_numindbound_default(lb=7, ub=30)
    rRM.set_softconstraint()
    rRS = RestraintCpSolver(rRM)

    status = rRS._solve()
    work_value, dict_dep = rRS.get_dictdep()
    transposer = Transposer()
    transposer.df_work = pd.read_excel("l2.xlsx", dtype=str)
    transposer.delete_num(88, dict_dep, (8, 9, 10, 11))
    transposer.delete_num(75, dict_dep, (8, 9, 10, 11))

    df = transposer.dep_tosheet(dict_dep)
    transposer.save_dftoxlsx(df, 'output_202308163')
    sheet = Sheet("output_202308163")
    sheet.set_colwidth()
    sheet.set_rowheight()
    sheet._set_default()
    sheet.workbook.save("output_202308163.xlsx")