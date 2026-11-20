import pandas as pd
import numpy as np
import math

EULER = 2.718281828459045

def sigmoid(val) -> float:
    e_to_val = pow(EULER, val)
    fx = e_to_val / (1 + e_to_val)
    return fx

def initialize_network(**param) -> tuple[pd.dataFrame]:

    return



while True:
    try:
        x = float(input("enter value: "))
        print(sigmoid(x))
    except ValueError:
        break
        
