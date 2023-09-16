# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 00:35:57 2023

@author: wongy
"""

__all__ = (
           "INT_MAXDELTA_DEPART", 
           "INT_MAXDELTA_DEPARTSQU", 
           "INT_MINPENALTY", 
           "INT_MAXPENALTY", 
           "SOLVER_CONFIG_MANNUAL", 
           "SOLVER_PARAMETER", 
           "DICT_INDEXDEP",
           "DICT_INDEXDEP_PDOMAIN_POS", 
           "DICT_INDEXDEP_PDOMAIN_NEG", 
           "INT_RESTRAINTCALLBACK_LIMIT", 
           "TPL_PERIODSUMMER", 
           "TPL_Y1", 
           "TPL_Y2", 
           "TPL_Y3", 
           "TPL_Y12", 
           "TPL_Y23", 
           "TPL_DEPART_Y12", 
           "TPL_DEPART_Y23", 
           "STR_SHEETFIELD_NAME", 
           "LIST_SHEETCOLUMNS", 
           "DICT_FONTPARAMETERS_MAINTEXT", 
           "DICT_FONTPARAMETERS_TITLETEXT", 
           "DICT_ALIGNMENTPARAMETERS_MAINTEXT", 
           "INT_SHEETCOLUMNWIDTH", 
           "INT_SHEETROWHEIGHT", 
           "___DEBUG",
           )

# ---------------------------------------------------
# SYSTEMATIC PARAMETERS
# ---------------------------------------------------
___DEBUG = False

# ---------------------------------------------------
# CPMODEL PARAMETERS
# ---------------------------------------------------
INT_MAXDELTA_DEPART = 100
INT_MAXDELTA_DEPARTSQU = 100000
INT_MAXPENALTY = 1000000
INT_MINPENALTY = -100

# ---------------------------------------------------
# CPSOLVER PARAMETERS
# ---------------------------------------------------
SOLVER_CONFIG_MANNUAL = True
SOLVER_PARAMETER = {
                    "max_time_in_seconds": 300, 
                    "enumerate_all_solutions": True
                    }
INT_RESTRAINTCALLBACK_LIMIT = 5
#----------------------------------------------------
# Configuration of department roasting
#----------------------------------------------------
DICT_INDEXDEP = {
                 0: "外院方向", 
                 1: "屈光与近视防控科/近视眼激光治疗科", 
                 2: "角膜科", 
                 3: "眼外伤科/眼科急症/感染平台", 
                 4: "白内障科", 
                 5: "斜视与弱视科", 
                 6: "青光眼/特需医疗科", 
                 7: "眼眶眼肿瘤科/眼整形科", 
                 8: "眼底外科/小儿眼底病科", 
                 9: "葡萄膜炎科/眼底内科", 
                 10: "防盲办/导师科室", 
                 }

TPL_Y1 = tuple([i for i in range(0, 3)])
TPL_Y2 = tuple([i for i in range(3, 7)])
TPL_Y3 = tuple([i for i in range(7, 11)])
TPL_Y12 = tuple([i for i in range(0, 7)])
TPL_Y23 = tuple([i for i in range(3, 11)])
TPL_PERIODSUMMER = tuple([2, 6, 10])
TPL_DEPART_Y12 = tuple([i for i in range(2, 5)])
TPL_DEPART_Y23 = tuple([i for i in range(5, 10)])
DICT_INDEXDEP_PDOMAIN_POS = {
                             0: TPL_Y1, 
                             1: TPL_Y1, 
                             2: TPL_Y12, 
                             3: TPL_Y12, 
                             4: TPL_Y12, 
                             5: TPL_Y23, 
                             6: TPL_Y23, 
                             7: TPL_Y23, 
                             8: TPL_Y23, 
                             9: TPL_Y23, 
                             10: TPL_Y3, 
                             }

DICT_INDEXDEP_PDOMAIN_NEG = {
                             0: TPL_Y23, 
                             1: TPL_Y23, 
                             2: TPL_Y3, 
                             3: TPL_Y3, 
                             4: TPL_Y3, 
                             5: TPL_Y1, 
                             6: TPL_Y1, 
                             7: TPL_Y1, 
                             8: TPL_Y1, 
                             9: TPL_Y1, 
                             10: TPL_Y12, 
                             }

#----------------------------------------------------
# Configuration of EXCEL SHEET
#----------------------------------------------------

STR_SHEETFIELD_NAME = "姓名"
LIST_SHEETCOLUMNS = tuple(["2023年第4季度", 
                           "2024年第1季度", 
                           "2024年第2季度", 
                           "2024年第3季度", 
                           "2024年第4季度", 
                           "2025年第1季度", 
                           "2025年第2季度", 
                           "2025年第3季度", 
                           "2025年第4季度", 
                           "2026年第1季度", 
                           "2026年第2季度", ])
DICT_FONTPARAMETERS_MAINTEXT = {"name": '宋体', 
                                "size": 11, 
                                "bold": False, 
                                "italic": False, }
DICT_FONTPARAMETERS_TITLETEXT = {"name": '微软雅黑', 
                                 "size": 15, 
                                 "bold": True, 
                                 "italic": False, }
INT_SHEETCOLUMNWIDTH = 30
INT_SHEETROWHEIGHT = 50

DICT_ALIGNMENTPARAMETERS_MAINTEXT = {'horizontal': 'center', 
                                     'vertical': 'center', 
                                     'wrap_text': True}