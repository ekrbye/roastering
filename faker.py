# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:19:52 2023

@author: wongy
"""

import random
import pandas as pd
import string

class Faker():
    def __init__(self, int_num=100, int_depart=10, int_period=12):
        self._int_num = int_num
        self._int_depart = int_depart
        self._int_period = int_period
    
    def generate_l_num(self, int_num=None, digit=5):
        if int_num is None: int_num = self._int_num
        # 从a-zA-Z0-9生成指定数量x的随机字符：
        g = lambda x:''.join(random.sample(string.ascii_letters, x))
        l_num = [g(digit) for _ in range(int_num)]
        return l_num
    
    def generate_l_numcsv(self, int_num=None, digit=5, 
                          **to_csv_kwargs):
        if "path_or_buf" not in to_csv_kwargs.keys():
            to_csv_kwargs["path_or_buf"] = "testing.csv"
        df = pd.DataFrame(self.generate_l_num(int_num, digit), 
                          columns=["name"])
        df.to_csv(**to_csv_kwargs)