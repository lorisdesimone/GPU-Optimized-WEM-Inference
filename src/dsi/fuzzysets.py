"""
Fuzzy sets defined heer are capable of infers a membeship value 
on a generic array with arbitrary dimensione [N,M,...]
"""

from .backend import xp

def triang_fs(arr, a,b,c):
    """
    Parameters
    ----------
    arr -> generic arrray of shape [N,M,...]
    a -> start of triangle
    b -> mid of triangle
    c -> end of triangle

    It handle also:
    L-function case: where a=b
    R-function case: where b=c

    Return
    ------
    fuzzy set values of shape [N,M,...]
    """

    if a == b and b < c:
        
        fs = xp.where(arr<=b, 1, (c - arr) / (c - b))
    
    elif  b == c and a<b:
    
        fs = xp.where(arr>=b, 1, (arr - a) / (b - a)) # right slope
        
    elif a<b<c:
        
        left = (arr - a) / (b - a) # left slope
        right = (c - arr) / (c - b) # right slope
        fs = xp.minimum(left, right)
        
    else:
        
        raise ValueError("Must be a<b<c or a=b<c or a<b=c")

    # clip to [0,1]
    return xp.clip(fs, 0, 1)