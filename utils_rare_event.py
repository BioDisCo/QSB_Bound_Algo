import scipy.linalg as sclig
import numpy as np


def round_to_significant(number, decimals=2, zero_threshold=1e-15):
    """
    Format a number in scientific notation matching LaTeX table style.
    Values smaller than zero_threshold are rounded to 0.
    """
    # Handle zero and very small numbers
    if number == 0 or abs(number) < zero_threshold:
        return '0'

    # Handle infinity
    if np.isinf(number):
        return '\\infty'

    # Get the order of magnitude
    exponent = int(np.floor(np.log10(abs(number))))

    # Scale the number to have 1 digit before decimal
    mantissa = number / (10 ** exponent)

    # Round the mantissa
    mantissa = round(mantissa, decimals)

    # Format based on the exponent
    if exponent == 0:
        # No scientific notation needed
        return str(round(number, decimals))
    elif exponent == -1:
        # For 0.1-0.9 range, might want to show as decimal
        return str(round(number, decimals))
    elif exponent > 0 and exponent <= 2:
        # For small positive exponents, show as regular number
        return str(round(number, decimals))
    else:
        # Use scientific notation
        if mantissa == 1.0 and decimals == 0:
            # Special case: exactly 10^n
            return f'10^{{{exponent}}}'
        else:
            # Format mantissa without trailing zeros
            mantissa_str = f'{mantissa:.{decimals}f}'.rstrip('0').rstrip('.')
            return f'{mantissa_str}\\cdot10^{{{exponent}}}'


def calculate_distribution(d, t, Q):
    return np.matmul(d, sclig.expm(t * Q))

