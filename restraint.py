# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:46:11 2023

@author: wongy
"""

from itertools import product
from inspect import signature

from ortools.sat.python import cp_model

from dectools import RestraintsLogger

from default import *


class RestraintCpModel(cp_model.CpModel):

    constraints_logger = RestraintsLogger()
    """
    __new__方法确保该类只能有一个实例
    """
    bool_singleton = False
    def __new__(cls, int_num: int, int_depart: int, int_period: int,
                *args, **kwargs):
        if cls.bool_singleton:
            raise ValueError("CpModel can only be singleton.")
        else:
            obj = object.__new__(cls)
            cls.bool_singleton = True
            return obj
        
    def __init__(self, int_num: int, int_depart: int, int_period: int, 
                 dict_aver: dict =None, *args, **kwargs):


        super(RestraintCpModel, self).__init__(*args, **kwargs)
        self._int_num = int_num
        self._int_depart = int_depart
        self._int_period = int_period
        self.work = {}
        self.numind = {} 
        self.bool_penalty = True
        self.penalty = []
        
        for n, d, p in product(range(self._int_num), 
                               range(self._int_depart), 
                               range(self._int_period)):
            self.work[(n, d, p)] = self.NewBoolVar("work%i%i%i" % (n, d, p))
        def s(d, p):
            return [self.work[(n, d, p)] for n in range(self._int_num)]
        for d, p in product(range(self._int_depart), 
                            range(self._int_period)):
            self.numind[(d, p)] = cp_model.LinearExpr.Sum(s(d, p))

        _t = self._int_num // self._int_depart
        self.aver = {(d, p): _t 
                     for d in range(self._int_depart) 
                     for p in range(self._int_period)
                     }


        
    def add_restraint_fornum(self, num: (int, list, tuple), 
                             depart: int, period: int,
                             reverse: bool =False) -> None:
        """
        添加一个硬约束条件，对于list/tuple序列的num，使得其必须在某一个period轮转
        某一个depart（或不轮转）
        
        Parameters
        ----------
        num : (list, tuple)
            DESCRIPTION.
        depart : int
            DESCRIPTION.
        period : int
            DESCRIPTION.
        Returns
        -------
        None.
        """
        if isinstance(num, int): num = [num]
        for n in num:
            if reverse:
                self.Add(self.work[(n, depart, period)] == 0)
            else:
                self.Add(self.work[(n, depart, period)] == 1)
    
    def add_atleast_forall(self):
        """
        添加一个硬约束条件，对于所有学员n，其在每一个季度p，使其每个季度都有科室
        轮转
        Returns
        -------
        None.

        """
        for n, p in product(range(self._int_num),
                            range(self._int_period)):
            self.Add(sum(
                        [self.work[(n, d, p)] for d in range(self._int_depart)]
                        ) == 1)
    
    def add_pdomain_forall(self, 
                           depart: int, 
                           pdomain: (int, list, tuple)) -> None:
        """
        添加一个硬约束条件，对于所有学员n，其必须要在给定时间段pdomain内轮转指定
        科室depart

        Parameters
        ----------
        pdomain : (int, list, tuple)
        
        Returns
        -------
        None.

        """
        if isinstance(pdomain, int): pdomain = [pdomain]
        for n in range(self._int_num):
            self.AddBoolOr([self.work[(n, depart, p)] for p in pdomain])
    
    def add_not_pdomain_forall(self, 
                               depart: int, 
                               pdomain: (int, list, tuple), 
                               rearrange: bool=False) -> None:
        """
        添加一个硬约束条件，对于所有学员n，其一定不能在给定时间段pdomain内轮转指定
        科室depart

        Parameters
        ----------
        pdomain : (int, list, tuple)
            DESCRIPTION.
        depart : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if isinstance(pdomain, int): pdomain = [pdomain]
        for n in range(self._int_num):
            self.AddBoolAnd([self.work[(n, depart, p)].Not() for p in pdomain])
        for p in pdomain:
            self.Add(self.numind[(depart, p)] == 0)
        # 重新分配指定科室的学员人数至当前季度的其他科室
        if rearrange:
            for p in pdomain:
                _d = [d for d in range(self._int_depart) if self.aver[(d, p)] != 0]
                _t = self.aver[(depart, p)] // len(_d)
                for d in _d: self.aver[(d, p)] += _t
                self.aver[(depart, p)] = 0
                
    def add_numindbound(self, 
                        depart: (None, int, list, tuple) =None, 
                        period: (None, int, list, tuple) =None, 
                        lb: (None, int) =None, 
                        ub: (None, int) =None) -> None:
        """
        添加一个硬约束条件，对于所有学员n，在指定季度period中，给定科室depart的
        总轮科人数区间为[lb, ub]

        Parameters
        ----------
        period : (None, int, list, tuple), optional
            DESCRIPTION. The default is None.
        depart : (None, int, list, tuple), optional
            DESCRIPTION. The default is None.
        lb : (None, int), optional
            DESCRIPTION. The default is None.
        ub : (None, int), optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if isinstance(period, int): period = [period]
        if isinstance(depart, int): depart = [depart]
        if period is None: period = range(self._int_period)
        if depart is None: depart = range(self._int_depart)
        if lb is not None:

            assert isinstance(lb, int), f"lb must be int type, {type(lb)} was given"
            for d, p in product(depart, period):
                self.Add(self.numind[(d, p)] >= lb)
        if ub is not None:
            assert isinstance(ub, int), f"ub must be int type, {type(ub)} was given"
            for d, p in product(depart, period):
                self.Add(self.numind[(d, p)] <= ub)
    
    def add_numindbound_default(self, 
                                lb: (None, int) =None, 
                                ub: (None, int) =None) -> None:
        for d, p in DICT_INDEXDEP_PDOMAIN_POS.items():
            self.add_numindbound(depart=d, period=p, lb=lb, ub=ub)
    
    def add_depinterval_forall(self, depart: (None, int, list, tuple) =None, 
                               lb: int =1, ub: int =2) -> None:
        """
        添加一个硬约束条件，对于所有学员n，其轮转的给定科室depart，使其在所有季度
        period的科室轮转次数区间为[lb, ub]

        Parameters
        ----------
        depart : (None, int, list, tuple)
            DESCRIPTION.
        lb : int, optional
            DESCRIPTION. The default is 1.
        ub : int, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        None.

        """
        if depart is None: depart = range(self._int_depart)
        if isinstance(depart, int): depart = [depart]
        for n, d in product(range(self._int_num), depart):
            _sum = [self.work[(n, d, p)] for p in range(self._int_period)]
            self.AddLinearConstraint(sum(_sum), lb, ub)

    
    # -----------------------------------------------------
    #                        ***注意***
    # -----------------------------------------------------
    """
    对于下面的函数，我本身希望将目标函数定义为科室轮转人数与给定人数差值的平方和，然后
    令目标函数最小。
    但平方和涉及两个变量相乘，为非线性问题，需要使用AddMultiplicationEquality方法。
    经过大量测试，使用AddMultiplicationEquality方法来添加乘法约束会有一些莫名其妙
    的问题，该方法可能导致求解器无法找到最优解。
    
    比如以下伪代码：
    给定X,Y,Z三个由NewIntVar创建的变量，满足：
    X + Y < 3
    X + Z < 5
    Y + Z > 1
    
    其平方和为:
    model.AddMultiplicationEquality(diff_XY, [(X - Y), (X - Y)])
    model.AddMultiplicationEquality(diff_XZ, [(X - Z), (X - Z)])
    model.AddMultiplicationEquality(diff_YZ, [(Y - Z), (Y - Z)])
    其中diff_XY、diff_XZ和diff_YZ为3个中间变量用来容纳差值的平方和。
    
    现在设定目标函数 f = diff_XY + diff_XZ + diff_YZ
    最小化目标函数 model.Minimize(f)
    
    最后solver.Solve(model)
    查看solver.status，会迅速返回无最优解。但很显然，X = Y = Z = 1就是一组最优解
    
    因此，下文中所有并入损失函数self.penalty的表达式皆为线性表达式
    """

    def sadd_numinaver(self, 
                      aver: (None, list, tuple, dict) =None,
                      depart: (None, list, tuple) =None, 
                      period: (None, list, tuple) =None, 
                      boolp: (None, bool) =None, 
                      linear: bool =True, 
                      w: int =1, 
                      ) -> None: 
        """
        添加一个软约束条件，科室的轮转人数尽量接近给定的 aver(d, p)，其中d和p分别
        为给定的科室和季度

        Parameters
        ----------
        aver : int, optional
            DESCRIPTION. The default is None.
        depart : (None, list, tuple), optional
            DESCRIPTION. The default is None.
        period : (None, list, tuple), optional
            DESCRIPTION. The default is None.
        boolp : (None, bool), optional
            DESCRIPTION. The default is None.
        linear: bool =True
            
        Returns
        -------
        None.

        """
        
        if depart is None: depart = range(self._int_depart)
        if period is None: period = range(self._int_period)

        if aver is None: aver = self.aver

        delta = {}
        absdelta = {}
        _bd = {}
        for p, d in product(period, depart):
            delta[(d, p)] = self.NewIntVar(-INT_MAXDELTA_DEPART, 
                                           INT_MAXDELTA_DEPART, '')
            absdelta[(d, p)] = self.NewIntVar(0, INT_MAXDELTA_DEPART, '')
            self.Add(delta[(d, p)] == self.numind[(d, p)] - aver[(d, p)])
            _bd[(d, p)] = self.NewBoolVar('')
            self.Add(delta[(d, p)] >= 0).OnlyEnforceIf(_bd[(d, p)])
            self.Add(delta[(d, p)] < 0).OnlyEnforceIf(_bd[(d, p)].Not())
            self.Add(absdelta[(d, p)] == delta[(d, p)] * w).OnlyEnforceIf(_bd[(d, p)])
            self.Add(absdelta[(d, p)] == -delta[(d, p)] * w).OnlyEnforceIf(_bd[(d, p)].Not())
            self.penalty.append(absdelta[(d, p)])

    def sadd_numinbound(self, 
                        lb: int =None,
                        ub: int =None, 
                        depart: (None, list, tuple) =None, 
                        period: (None, list, tuple) =None, 
                        lbw: int=1,
                        ubw: int=1,
                        ) -> None:
        """
        给定

        Parameters
        ----------
        lb : int, optional
            DESCRIPTION. The default is None.
        ub : int, optional
            DESCRIPTION. The default is None.
        depart : (None, list, tuple), optional
            DESCRIPTION. The default is None.
        period : (None, list, tuple), optional
            DESCRIPTION. The default is None.
        lbw : int, optional
            DESCRIPTION. The default is 1.
        ubw : int, optional
            DESCRIPTION. The default is 1.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """

        if depart is None: depart = range(self._int_depart)
        if period is None: period = range(self._int_period)

        if lb is not None:
            assert isinstance(lb, int), f"lb must be int type, {type(lb)} was given"
            delta_lb = {}
            absdelta_lb = {}
            _blbd = {}
            for p, d in product(period, depart):
                _blbd[(d, p)] = self.NewBoolVar('')
                delta_lb[(d, p)] = self.NewIntVar(-INT_MAXDELTA_DEPART, 
                                                  INT_MAXDELTA_DEPART, '')
                absdelta_lb[(d, p)] = self.NewIntVar(0, INT_MAXDELTA_DEPART, '')
                self.Add(delta_lb[(d, p)] == self.numind[(d, p)] - lb)
                self.Add(delta_lb[(d, p)] < 0).OnlyEnforceIf(_blbd[(d, p)])
                self.Add(absdelta_lb[(d, p)] == 
                         - delta_lb[(d, p)] * lbw).OnlyEnforceIf(_blbd[(d, p)])
                self.penalty.append(absdelta_lb[(d, p)])
        if ub is not None:
            assert isinstance(ub, int), f"ub must be int type, {type(ub)} was given"
            delta_ub = {}
            absdelta_ub = {}
            _bubd = {}
            for p, d in product(period, depart):
                _bubd[(d, p)] = self.NewBoolVar('')
                delta_ub[(d, p)] = self.NewIntVar(-INT_MAXDELTA_DEPART, 
                                                  INT_MAXDELTA_DEPART, '')
                absdelta_ub[(d, p)] = self.NewIntVar(0, INT_MAXDELTA_DEPARTSQU, '')
                self.Add(delta_ub[(d, p)] == self.numind[(d, p)] - ub)
                self.Add(delta_ub[(d, p)] > 0).OnlyEnforceIf(_bubd[(d, p)])
                self.Add(absdelta_ub[(d, p)] == delta_ub[(d, p)] * ubw).OnlyEnforceIf(_bubd[(d, p)])
                self.penalty.append(absdelta_ub[(d, p)])
        
    def sadd_restraint_fornum(self, num: (int, list, tuple), depart: int, 
                              period: int, weight: int=1, reverse: bool =False, 
                              boolp: (None, bool) =None) -> None:
        """
        添加一个软约束条件，当boolp为True时，尽量满足学员num在给定季度period轮转
        给定科室depart的要求。如果满足，则penalty减去权重weight，否则penalty增加
        权重weight

        Parameters
        ----------
        num : (int, list, tuple)
            DESCRIPTION.
        depart : int
            DESCRIPTION.
        period : int
            DESCRIPTION.
        weight : int且只能为int
            如果reverse为False:
                对于当前排班work[(n,d,p)] -> 0 penalty += weight
                                         -> 1 penalty -= weight
            如果reverse为True:
                对于当前排班work[(n,d,p)] -> 0 penalty -= weight
                                         -> 1 penalty += weight
            The default is 1.
        reverse : bool, optional
            reverse为False：表示学员num希望在给定季度period轮转给定科室depart. 
            reverse为False：表示学员num不希望在给定季度period轮转给定科室depart. 
            The default is False.
        boolp : (None, bool), optional
            DESCRIPTION. The default is None.
        Returns
        -------
        None
            DESCRIPTION.

        """


        if isinstance(num, int): num = [num]
        assert isinstance(weight, int), f"weight must be int type, {type(weight)} was given"
        assert weight > 0, "weight must be over 0."
        if reverse:
            for n in num:
                _t = self.NewIntVarFromDomain(
                     cp_model.Domain.FromValues([-weight, weight], '')
                     )
                self.Add(
                         _t == weight * self.work[(n, depart, period)]
                         ).OnlyEnforceIf(self.work[(n, depart, period)])
                self.Add(
                         _t == - weight * self.work[(n, depart, period)].Not()
                         ).OnlyEnforceIf(self.work[(n, depart, period)].Not())
                self.penalty.append(_t)
        else:
            for n in num:
                _t = self.NewIntVarFromDomain(
                     cp_model.Domain.FromValues([-weight, weight], '')
                     )
                self.Add(
                         _t == - weight * self.work[(n, depart, period)]
                         ).OnlyEnforceIf(self.work[(n, depart, period)])
                self.Add(
                         _t == weight * self.work[(n, depart, period)].Not()
                         ).OnlyEnforceIf(self.work[(n, depart, period)].Not())
                self.penalty.append(_t)
        
    def set_softconstraint(self, mindesicion=False):
        """
        `SELECT_MIN_VALUE`和`model.Minimize()`之间有一些区别。`SELECT_MIN_VALUE`
        是一种值选择策略，用于在`AddDecisionStrategy`方法中指定在搜索过程中如何从
        选定变量的取值范围中选择一个值。而`model.Minimize()`是一种优化目标，用于
        指定在求解过程中需要最小化的目标函数。

        1. `SELECT_MIN_VALUE`：这是一种值选择策略，用于在`AddDecisionStrategy`方
        法中指定在搜索过程中如何从选定变量的取值范围中选择一个值。例如，如果您有一
        个整数变量`x`，其取值范围为1到5，那么`SELECT_MIN_VALUE`策略将首先尝试为`x`
        分配最小值1。这个策略可以与其他变量选择策略（如`CHOOSE_FIRST`、
        `CHOOSE_LOWEST_MIN`等）结合使用，以指定在搜索过程中如何选择变量。

        2. `model.Minimize()`：这是一种优化目标，用于指定在求解过程中需要最小化的
        目标函数。例如，如果您有一个线性规划问题，需要最小化目标函数
        `c1 * x1 + c2 * x2`，那么您可以使用`model.Minimize(c1 * x1 + c2 * x2)`
        来指定这个优化目标。在求解过程中，求解器将尝试找到一组变量取值，使得目标函
        数的值最小。

        总之，`SELECT_MIN_VALUE`和`model.Minimize()`在求解过程中扮演不同的角色。
        `SELECT_MIN_VALUE`是一种值选择策略，用于指定在搜索过程中如何从选定变量的
        取值范围中选择一个值。而`model.Minimize()`是一种优化目标，用于指定在求解
        过程中需要最小化的目标函数。这两者可以根据问题的特点和需求进行组合使用。


        如果将变量`X`设置为`c1 * x1 + c2 * x2`，然后添加约束条件，使`X`使用
        `SELECT_MIN_VALUE`策略，并不等价于`model.Minimize(c1 * x1 + c2 * x2)`。
        这两者在求解过程中扮演不同的角色。

        `SELECT_MIN_VALUE`是一种值选择策略，用于在`AddDecisionStrategy`方法中指
        定在搜索过程中如何从选定变量的取值范围中选择一个值。当使用`SELECT_MIN_VALUE`
        策略时，求解器会在搜索过程中尝试为选定变量分配最小值。这个策略可以与其他变
        量选择策略（如`CHOOSE_FIRST`、`CHOOSE_LOWEST_MIN`等）结合使用，以指定在
        搜索过程中如何选择变量。

        而`model.Minimize()`是一种优化目标，用于指定在求解过程中需要最小化的目标
        函数。当您使用`model.Minimize(c1 * x1 + c2 * x2)`时，求解器会尝试找到一
        组变量取值，使得目标函数`c1 * x1 + c2 * x2`的值最小。

        因此，将`X`设置为`c1 * x1 + c2 * x2`，然后添加约束条件，使`X`使用
        `SELECT_MIN_VALUE`策略，只会影响求解过程中变量`X`的值选择方式，而不会使求
        解器寻找最小化目标函数`c1 * x1 + c2 * x2`的解。如果希望最小化目标函数
        `c1 * x1 + c2 * x2`，应该使用`model.Minimize(c1 * x1 + c2 * x2)`。

        """
        if mindesicion:
            costvar = self.NewIntVar(INT_MINPENALTY, INT_MAXPENALTY, "costvar")
            self.Add(costvar == cp_model.LinearExpr.Sum(self.penalty))
            self.Minimize(costvar)
            self.AddDecisionStrategy([costvar], 
                                     cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)
        else:
            self.Minimize(cp_model.LinearExpr.Sum(self.penalty))

class RestraintCpSolver(cp_model.CpSolver): 
    def __init__(self, model, printer=None, *args, **kwargs):
        """
        
        Parameters
        ----------
        model : 参数为具有约束条件的cp_model.CpModel
        
        Returns
        -------
        None.

        """
        super(RestraintCpSolver, self).__init__(*args, **kwargs)
        self._model = model
        
        if SOLVER_CONFIG_MANNUAL:
            for k, v in SOLVER_PARAMETER.items(): setattr(self.parameters, k, v)

        
    def _solve(self) -> int:
        """
        派生原有cp_model.CpSolver类的cp_model.CpSolver().Solve(model)方法，并
        返回其状态
        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        #if printer is not None:
        #    return super(RestraintCpSolver, self).Solve(self._model, printer)
        #else:
        
        return super(RestraintCpSolver, self).Solve(self._model)
    

    def get_dictdep(self, work: (None, dict) =None) -> (dict, dict):
        if work is None:
            max_n = self._model._int_num - 1
            max_d = self._model._int_depart - 1
            max_p = self._model._int_period- 1
            work = self._model.work
        else:
            max_n, max_d, max_p = max(work.keys())
        work_value = {}
        dict_dep = {(d, p): []
                    for d, p in product(range(max_d + 1), range(max_p + 1))
                    }
        for n, d, p in product(range(max_n + 1), 
                               range(max_d + 1), 
                               range(max_p + 1)):
            work_value[(n, d, p)] = self.Value(work[(n, d, p)])
            if work_value[(n, d, p)]: dict_dep[(d, p)].append(n)
        return work_value, dict_dep
    
class RestraintCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, *args, **kwargs):
        super(RestraintCallback, self).__init__(*args, **kwargs)
        self._count = 0
        self._limit = INT_RESTRAINTCALLBACK_LIMIT
        self._work = {}
        self._penalty = {}
        self._solution_work = {}
        self._solution_penalty = {}
        self._solver = None
        
    def get_objvariables(self, obj: RestraintCpModel) -> None:
        if isinstance(obj, RestraintCpModel):
            # obj需要为RestraintCpModel类
            self._work = obj.work 
            self._penalty = obj.penalty
            
    def on_solution_callback(self, solver: (None, RestraintCpSolver)=None) -> None:
        if solver is None: solver = self._solver
        self._solution_work[self._limit] = solver.get_dictdep(self._work)
        self._penalty[self._limit] = cp_model.CpSolver.Value(solver._penalty)
        if self._count >= self._limit:
            self.StopSearch()
        

