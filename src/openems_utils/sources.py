from typing import Sequence
import numpy as np
from numpy import pi, sqrt, log
from scipy.special import erfinv

__all__ = ['gaussian_step', 'evaluate_custom_source_string']

def gaussian_step(rise_time, center_time, dB_cutoff=6, sign=1):
    '''
    Generate a string expression for a Gaussian step (rising or falling)
    suitable for OpenEMS time-domain source definitions.

    Usage:
    ```python
    fm = 1.0e-12
    string, f_nyquist = gaussian_step(rise_time=50e-12, center_time=1e-9, dB_cutoff=3)
    max_res = C0 / f_nyquist / 20
    FDTD.SetCustomExcite(bytes(string, "utf-8"), f_nyquist, fm)
    ```

    https://cdn.hackaday.io/files/1656277086185568/gaussian_step_v11.pdf

    Gaussian step using erf() approximation 7.1.25 from Abramowitz and Stegun
    https://personal.math.ubc.ca/~cbm/aands/abramowitz_and_stegun.pdf pg. 299
    https://personal.math.ubc.ca/~cbm/aands/page_299.htm


    Parameters:
    - rise_time: 10–90% rise time in seconds
    - center_time: step midpoint in seconds
    - dB_cutoff: cutoff level in dB (default: 6 dB)
    - sign: +1 for rising edge, -1 for falling edge

    Returns:
    - string: expression for time-dependent step
    - f_nyquist: suggested Nyquist frequency in Hz
    '''
    assert sign in [1, -1], "sign must be +1 (rising) or -1 (falling)"

    C = center_time
    sigma = rise_time / (2 * erfinv(0.8))
    K = 1 / sigma
    f_nyquist = 2 * sqrt((dB_cutoff / 20) * log(10) / (pi**2 * sigma**2))

    # Gaussian step approximation based on erf() (Abramowitz and Stegun 7.1.25)
    base_expr = f'(1 - exp(-{K}*(t-{C})*{K}*(t-{C})) * ' \
                f'(0.3480242/(1+0.47047*{K}*(t-{C})) - ' \
                f'0.0958798/((1+0.47047*{K}*(t-{C}))**2) + ' \
                f'0.7478556/((1+0.47047*{K}*(t-{C}))**3)))'

    neg_expr = base_expr.replace('+0.47047', '-0.47047')

    string = f'0.5 + 0.5*({K}*(t-{C})>=0)*{base_expr} - 0.5*({K}*(t-{C})<0)*{neg_expr}'

    if sign == -1:
        string = f'1 - ({string})'

    return string, f_nyquist


def evaluate_custom_source_string(source: str, t: Sequence[float], context=None):
    '''
    Evaluate a custom time-dependent string expression at each time in `t`.

    Parameters:
    - source: A string expression using 't' (e.g., "exp(-t*t)")
    - t: A sequence of time points (list, tuple, NumPy array, etc.)

    Returns:
    - times: list of t values
    - values: list of evaluated values
    '''
    safe_globals_ = {
        'pi': np.pi, 'sqrt': np.sqrt, 'log': np.log, 'exp': np.exp
    }

    if isinstance(context, dict):
        safe_globals_.update(context)

    values = eval(source, safe_globals_, {'t': t})
    return list(t), values
