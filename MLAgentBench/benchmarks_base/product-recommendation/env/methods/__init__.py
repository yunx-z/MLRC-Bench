## TODO Add more implemented methods here
from methods.MyMethod import MyMethod
from methods.RandomMethod import RandomMethod

def all_method_handlers():
    """Enumerate and Load (import) all implemented methods."""
    loaded_methods = {
            "my_method" : MyMethod,
            "random" : RandomMethod
            ## TODO Add more implemented methods here
            }

    return loaded_methods
