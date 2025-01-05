## TODO Add more merge methods here
from methods.MyMethod import MyMethod
from methods.DareTies import DareTies

def all_method_handlers():
    """Enumerate and Load (import) all implemented methods."""
    loaded_methods = {
        "my_method" : MyMethod,
        "dare_ties" : DareTies,
        ## TODO Add more merge methods here
    }
    
    return loaded_methods

