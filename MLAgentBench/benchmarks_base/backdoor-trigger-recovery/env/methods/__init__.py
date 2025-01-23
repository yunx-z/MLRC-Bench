from methods.MyMethod import MyMethod

def all_method_handlers():
    """Enumerate and Load (import) all implemented methods."""
    loaded_methods = {
        "my_method": MyMethod,
        # Add more methods here if needed
    }
    return loaded_methods

