"""
Domain division utilities for fuzzy set generation.

Each function produces an array of pivots, where the first and last values
are repeated to allow construction of L-functions and R-functions.

Example
-------
Given centers [-1, 0, 2, 3, 4], the padded pivot array becomes:
[-1, -1, 0, 2, 3, 4, 4]

This corresponds to 5 fuzzy sets.
"""

from .backend import xp
import warnings

# --- private fuincion for density division ---
def _check_for_unique_ordered_centers(to_test):
    """
    Ensure that the array of centers is strictly increasing and unique.

    If repeated or non-monotonic values are detected, a warning is issued
    and the centers are replaced with their unique sorted version.

    Parameters
    ----------
    to_test : array-like
        Array of candidate centers.

    Returns
    -------
    array-like
        Validated array of strictly increasing centers.
    """
    
    if (to_test[1:]<=to_test[:-1]).any():
        warnings.warn(
        "Domain division failed: insufficient unique values in the "
        "feature domain caused repeated quantile pivots. The effective number of "
        "fuzzy sets is now lower than the theoretical. "
        "Consider using linear_division or unique_division for this feature.",
        RuntimeWarning,
        stacklevel=2
        )
        
        return xp.unique(to_test)

    else:

        return to_test

def check_array_1D(arr):
    """
    Validate that the input is a 1D array and convert it to the backend array type.

    Parameters
    ----------
    arr : array-like
        Input array.

    Returns
    -------
    array-like
        Backend array with 1D shape.

    Raises
    ------
    ValueError
        If the array is not 1-dimensional.
    """

    if len(arr.shape) != 1:
        raise ValueError("arr must be 1D")
        
    return xp.asarray(arr)

def density_div(arr, n_of_sets):
    """
    Density-based domain division.

    Centers are computed using quantiles, so fuzzy sets are denser where
    the data distribution is denser.

    Parameters
    ----------
    arr : array-like, shape (N,)
        Input feature values.
    n_of_sets : int
        Number of fuzzy sets (centers) to generate.

    Returns
    -------
    pivots : array-like, shape (n_of_sets + 2,)
        Padded array of pivots. The first and last values are repeated
        to allow construction of L- and R-functions.
    """
    
    arr = check_array_1D(arr)

    quantiles = xp.linspace(0, 1, n_of_sets) # Quantiles
    
    centers = _check_for_unique_ordered_centers(xp.quantile(arr, quantiles)) # Centers
    
    return xp.pad(centers, (1,1), mode="edge")

def linear_div(arr, n_of_sets):
    """
    Linear domain division.

    Centers are evenly spaced between the minimum and maximum of the array.

    Parameters
    ----------
    arr : array-like, shape (N,)
        Input feature values.
    n_of_sets : int
        Number of fuzzy sets (centers).

    Returns
    -------
    pivots : array-like, shape (n_of_sets + 2,)
        Padded array of pivots. The first and last values are repeated
        to allow construction of L- and R-functions.
    """
    
    arr = check_array_1D(arr)

    # domain retrival
    mn = arr.min()
    mx = arr.max()

    # linear spaced centes
    centers = xp.linspace(mn, mx, n_of_sets)

    return xp.pad(centers, (1,1), mode="edge")

def unique_div(arr, n_of_sets):
    """
    Unique value domain division.

    Centers correspond to the unique values present in the array.
    This method is typically used for categorical or discrete features.

    Parameters
    ----------
    arr : array-like, shape (N,)
        Input feature values.
    n_of_sets : int
        Ignored. The number of fuzzy sets is determined by the number
        of unique values in the array.

    Returns
    -------
    pivots : array-like
        Padded array of pivots based on unique sorted values.
    """

    arr = check_array_1D(arr)
    
    return xp.pad(xp.unique(xp.asarray(arr)),(1,1), mode="edge")