from .backend import xp

def mse(y_pred, y_true):
    """
    Mean Squared Error using backend xp.
    """
    y_pred = xp.asarray(y_pred)
    y_true = xp.asarray(y_true)

    return xp.mean((y_pred - y_true) ** 2)


def r2(y_pred, y_true):
    """
    Coefficient of determination R^2 using backend xp.
    """
    y_pred = xp.asarray(y_pred)
    y_true = xp.asarray(y_true)

    ss_res = xp.sum((y_true - y_pred) ** 2)
    ss_tot = xp.sum((y_true - xp.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot

def mixed_metric(y_pred, y_true, metric="r2"):
    """
    Score that combines regression scores with coverage of algorithm
    """

    y_pred = xp.asarray(y_pred)
    y_true = xp.asarray(y_true)

    mask = ~xp.isnan(y_pred) #Mask True when not NaN

    coverage = xp.mean(mask.astype(float))  # in [0,1]

    # No predict -> Most negative score
    if coverage == 0:
        return -xp.inf if metric == "r2" else xp.inf

    # Mix regression score and coverage
    if metric == "r2":
        score = r2(y_pred[mask], y_true[mask])
        return score * coverage

    elif metric == "mse":
        score = mse(y_pred[mask], y_true[mask])
        return score / coverage

    else:
        raise ValueError("metric must be 'r2' or 'mse'")