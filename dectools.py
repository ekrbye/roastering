# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 19:36:24 2023

@author: wongy
"""

"""
本模块为一些常用工具
"""
from inspect import signature
from functools import wraps

class RestraintsLogger:
    """
    该类的目的是记录RestraintCpModel类的约束条件及其参数
    """
    
    def __init__(self):
        self.logged_functions = {}

    def log_function(self, func):
        """
        记录函数写成装饰器的形式
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            func_name = func.__name__
            arg_names = list(signature(func).parameters.keys())
            arg_values = list(args) + [v for v in kwargs.values()]            
            if func_name in self.logged_functions:
                self.logged_functions[func_name].append((arg_names, arg_values))
            else:
                self.logged_functions[func_name] = [(arg_names, arg_values)]
            return result
        return wrapper
    
    def print_log(self):
        """
        打印目前所记录的约束条件及其参数
        """
        for f in self.logged_functions.keys():
            print( f"Executed function '{f}':")
            for n, v in self.logged_functions[f]:
                print(f"args: {n} -> values: {v}")
                
