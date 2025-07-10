import numpy as np
import pickle
import os 
from typing import Dict, Any, List

equiv_dic_fpa50 = {f'{4000*k}.cfg':f'00_000_{str(1000000+k+1)[1:]}.poscar' for k in range(128)}
inv_equiv_dic_fpa50 = {val:key for key, val in equiv_dic_fpa50.items()}

equiv_dic_fpa60 = {f'{4000*k}.cfg':f'00_000_{str(1000000+k+1)[1:]}.poscar' for k in range(152)}
inv_equiv_dic_fpa60 = {val:key for key, val in equiv_dic_fpa60.items()}

equiv_dic_fpa70 = {f'{4000*k}.cfg':f'00_000_{str(1000000+k+1)[1:]}.poscar' for k in range(152)}
inv_equiv_dic_fpa70 = {val:key for key, val in equiv_dic_fpa70.items()}

print (equiv_dic_fpa70)
