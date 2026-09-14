# Backend
from .backend import xp, is_gpu_backend

# Regrression methods
from .regressors import DSregressor

# Fuzzy sets
from .fuzzysets import triang_fs

# Regression scores
from .metrics import mse, r2, mixed_metric, mae, rmse

# Domain division
from .domainDivision import density_div, linear_div, unique_div

__all__ = [
    #DSRegressor class
    "DSregressor",
    #Fuzzy sets
    "triang_fs",  
    #scores
    "mse", "r2","mixed_metric",
    #Domain division
    "density_div", "linear_div", "unique_div",
]