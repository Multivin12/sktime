
from sktime.transformers.dictionary_based import PAA

class ShapeDTW():
    """
    
    
    
    Parameters
    ----------
    n_neighbours                : int, set k for knn (default = 1)
    shape_descriptor_function   : string, defines the function to describe the set of subsequences (default = raw)
    
    
    
    """
    def __init__(self,n_neighbours=1,shape_descriptor_function='raw',):
        pass