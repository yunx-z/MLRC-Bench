from .BaseMethod import BaseMethod
from .MyMethod import MyMethod

def all_method_handlers():
    """
    Returns a dictionary of all available watermark removal methods
    
    Returns:
        dict: Dictionary mapping method names to their handler classes
    """
    return {
        'base_method': BaseMethod,
        'my_method': MyMethod
    } 
