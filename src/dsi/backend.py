# --- Backend selection ---

try:
    import cupy as _cp
    import cupyx.scipy as _cpx

    if _cp.cuda.runtime.getDeviceCount() > 0:
        _GPU_AVAILABLE = True
        xp = _cp          # array backend (GPU)
        xpy = _cpx        # scipy-like backend (GPU)
    else:
        _GPU_AVAILABLE = False
        xp = _np          # array backend (CPU)
        xpy = _sp         # scipy backend (CPU)

except Exception:
    _GPU_AVAILABLE = False
    xp = _np
    xpy = _sp

# --- GPU test ---

def is_gpu_backend():
    '''
    Returns True if cuda devices are ready and operations runs on GPU.
    '''
    return _GPU_AVAILABLE and xp.__name__.startswith("cupy")