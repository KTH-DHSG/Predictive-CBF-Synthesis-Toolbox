"""

    Various auxiliary functions for other classes, functions and scripts of this project.

    (c) Adrian Wiltz, 2025
    
"""

import inspect
import json
import math
import casadi as ca
import casadi as casadi
import textwrap
import types
import sys

import numpy as np

def ensure_json_extension(filename):
    """
    Ensure that the filename has the ".json" extension.
    """
    
    if not filename.endswith(".json"):
        filename += ".json"
    return filename

def next_smaller_even(number):
    # Floor the number to the nearest integer
    floored = math.floor(number)
    # If it's even, return it; otherwise, subtract 1 to make it even
    return floored if floored % 2 == 0 else floored - 1

def is_casadi_related(obj):
    """
    Determines if an object is CasADi-related.
    Expects an object as a parameter.
    This checks if the object's type originates from the CasADi module.
    """
    # Check if the object's type belongs to the CasADi module
    return obj.__class__.__module__.startswith("casadi")

def funcToJSONString(func):

    attributes = {}
    if callable(func):  # Save functions or lambdas
        if isinstance(func, types.LambdaType) and func.__name__ == "<lambda>":
            lambda_source = inspect.getsource(func)
            attributes['func'] = {
                "type": "lambda",
                "source": lambda_source[lambda_source.find("=") + 1:].strip()
            }
        else:
            attributes['func'] = {
                "type": "function",
                "name": func.__name__,
                "source": textwrap.dedent(inspect.getsource(func))
            }
    else:
        raise ValueError("The input is not a function or a lambda function.")
    
    return json.dumps(attributes, indent=4)

    
def JSONStringToFunc(json_str):

    attributes = json.loads(json_str)
    func = None
 
    key = next(iter(attributes))
    value = attributes[key]
    if key == "func":
        try:
            if isinstance(value, dict):
                if value.get("type") == "function":  # Reconstruct functions
                    exec(value["source"], globals())
                    func = eval(value["name"])
                elif value.get("type") == "lambda":  # Reconstruct lambdas
                    func = eval(value["source"])
        except Exception as e:
            raise ValueError(f"The string has the wrong format. The following error occurred: {e}")
        
    return func

def is_meshgrid(arr_list):
    # A meshgrid is a list of numpy arrays with the same shape
    if isinstance(arr_list, list) and all(isinstance(arr, np.ndarray) for arr in arr_list):
        shapes = [arr.shape for arr in arr_list]
        return all(shape == shapes[0] for shape in shapes)
    return False

