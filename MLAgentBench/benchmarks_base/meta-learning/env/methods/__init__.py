## TODO Add more implemented methods here
from methods.MyMethod import MyMethod

def all_method_handlers():
    """Enumerate and Load (import) all implemented methods."""
    loaded_methods = {
            "my_method" : MyMethod,
            ## TODO Add more implemented methods here
            "random" : MyMethod
            }
    
    loaded_method_dirs = {
            "my_method" : "protonet",
            ## TODO Add more implemented directories here
            "random" : "random"
            }
 

    return loaded_methods, loaded_method_dirs
