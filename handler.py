
"""
Created on Sat Aug 12 15:41:06 2023

@author: wongy
"""

from inspect import signature


class FunctionCache:
    def __init__(self):
        self.functions = {}
        self.logged_functions = {}

    
    def add_function(self, func, *args, **kwargs):

        func_name = func.__name__
        self.functions[func_name] = func
        arg_names = list(signature(func).parameters.keys())
        arg_values = list(args) + [v for v in kwargs.values()]            
        if func_name in self.logged_functions:
            self.logged_functions[func_name].append((arg_names, arg_values))
        else:
            self.logged_functions[func_name] = (arg_names, arg_values)
        
    def run_all(self):
        for func, params in self.logged_functions.items():
            names, values = params
            _param = (dict(zip(names, values)))
            print(_param)
            self.functions[func](**_param)
            
def f1(a, b, *, c, d):
    print(a)
    print(b)
    print(c)

        
def f2(a, b, c=3, d=4, e=10):
    print(a)
    print(b)
    print(c)
    print(e)
fc = FunctionCache()
fc.add_function(f1, 1, b=3, c=4, d=10)
fc.add_function(f2, 3, 4, e=20)
fc.run_all()