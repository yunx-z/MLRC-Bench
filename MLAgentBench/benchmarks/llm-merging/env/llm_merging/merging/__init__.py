## TODO Add more merge methods here
from llm_merging.merging.MyMerge import MyMerge

def all_merge_handlers():
    """Enumerate and Load (import) all merge methods."""
    loaded_merges = {
        "my_merge" : MyMerge,
        ## TODO Add more merge methods here
    }
    
    return loaded_merges

