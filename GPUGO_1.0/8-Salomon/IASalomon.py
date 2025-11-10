'''
Copyright 2025 Carnegie Mellon University. All rights reserved.                

GPUGO

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

YOU MAY OBTAIN A COPY OF THE AGREEMENT AT

https://github.com/CMU-Integrated-Design-Innovation-Group/GPUGO

 

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF

THIS LICENSE AGREEMENT. IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY

NOT USE OR DOWNLOAD THE SOFTWARE.

 

IF YOU ARE INTERESTED IN A COMMERCIAL LICENSE, PLEASE CONTACT CMU'S CENTER

FOR TECHNOLOGY TRANSFER AND ENTERPRISE CREATION AT INNOVATION@CMU.EDU.

'''

'''
Python script "IASalomon.py” implements the basic interval arithmetic 
operations on GPU-based devices for interval evaluation of the Salomon 
Function [1].

As stated in the NVIDIA white paper [2], floating point data in CUDA-enabled 
GPUs can be stored in either 32-bit or 64-bit formats. This script implements 
the interval arithmetic operations in 64-bit format (double precision). Such 
implementation enables the input and output of 64-bit floating point data, 
which has around 16 significant digits. Notably, the variable n in the function 
"outRound" is set to 15 to ensure that outward rounding after each interval 
arithmetic operation returns outwardly rounded interval defined by lower and 
upper bounds with 15 significant digits. The outwardly rounded interval contains 
all possible results of an interval arithmetic operation.

Reference:
[1] Salomon, R. 1996. Re-evaluating Genetic Algorithm Performance Under 
Coordinate Rotation of Benchmark Functions: A Survey of Some Theoretical and 
Practical Aspects of Genetic Algorithms. BioSystems 39(3), 263-278.
https://doi.org/10.1016/0303-2647(96)01621-8

[2] NVIDIA Corporation. 2020. Floating Point and IEEE 754 Compliance for NVIDIA 
GPUs. 
Available at: https://docs.nvidia.com/cuda/floating-point/index.html.

'''

import math
from numba import cuda

@cuda.jit
def ru(n, decimals=0):

    '''
    The function "ru" rounds a number up to a specified number of decimal
    places.

    Parameters
    ----------
    n: float
    The number to be rounded up.
    decimals: int
    The number of decimal places to round to.

    Returns
    -------
    result: float
    The number rounded up to the specified number of decimal places.
    '''

    mult = 10 ** decimals
    result = math.ceil(n * mult) / mult
    return result

@cuda.jit
def rd(n, decimals=0):

    '''
    The function "rd" rounds a number down to a specified number of decimal 
    places.

    Parameters
    ----------
    n: float
    The number to be rounded down.
    decimals: int
    The number of decimal places to round to.

    Returns
    -------
    result: float
    The number rounded down to the specified number of decimal places.
    '''
        
    mult = 10 ** decimals
    result = math.floor(n * mult) / mult
    return result

@cuda.jit
def outRound(lower, upper, n):
    
    '''
    The function "outRound" outwardly rounds the bounds of an interval to the 
    specified number of significant digits.

    Parameters
    ----------
    lower: float
    The lower bound of the interval.
    upper: float
    The upper bound of the interval.
    n: int
    The number of significant digits specified.

    Returns
    -------
    result: float
    The lower and upper bounds of the interval rounded to the specified number 
    of significant digits.
    '''

    if lower == 0:
        lowerOut = 0
    elif lower < 0:
        lowerOut = -ru(-lower, n - 1 - math.floor(math.log10(-lower)))
    else:
        lowerOut = rd(lower, n - 1 - math.floor(math.log10(lower)))
    if upper == 0:
        upperOut = 0
    elif upper < 0:
        upperOut = -rd(-upper, n - 1 - math.floor(math.log10(-upper)))
    else:
        upperOut = ru(upper, n - 1 - math.floor(math.log10(upper)))
    return lowerOut, upperOut


# User defined number of significant digits for outward rounding
n = 15 

@cuda.jit
def add(lower1, upper1, lower2, upper2):
   
    '''
    The function "add" implements the interval arithmetic operation of adding 
    two intervals with outwardly rounding.

    Parameters
    ----------
    lower1: float
    The lower bound of the first interval.
    upper1: float
    The upper bound of the first interval.

    lower2: float
    The lower bound of the second interval.
    upper2: float
    The upper bound of the second interval.

    Returns
    -------
    result: float
    The addition of two intervals using interval arithmetic with outwardly 
    rounding.
    '''
    
    result = outRound(lower1 + lower2, upper1 + upper2, n)
    return result

@cuda.jit
def multiply(lower1, upper1, lower2, upper2):

    '''
    The function "multiply" implements the interval arithmetic operation of 
    multiplying two intervals with outwardly rounding.

    Parameters
    ----------
    lower1: float
    The lower bound of the first interval.
    upper1: float
    The upper bound of the first interval.

    lower2: float
    The lower bound of the second interval.
    upper2: float
    The upper bound of the second interval.

    Returns
    -------
    result: float
    The multiplication of two intervals using interval arithmetic with outwardly 
    rounding.
    '''
        
    min1 = min(lower1*lower2, lower1*upper2, upper1*lower2, upper1*upper2)
    max1 = max(lower1*lower2, lower1*upper2, upper1*lower2, upper1*upper2)
    result = outRound(min1,max1,n)
    return result
        
@cuda.jit
def power(lower, upper, pow):

    '''
    The function "power" implements the interval arithmetic operation of raising
    the power of an interval with outwardly rounding.
    Note: This function is NOT generally applicable.  It cannot be applied to 
    the case where pow < 0, upper > 0, and lower < 0 (i.e., the interval 
    includes zero while the power is smaller than zero).

    Parameters
    ----------
    lower: float
    The lower bound of the interval.
    upper: float
    The upper bound of the interval.
    pow: int
    The power that needs to be raised to the interval.

    Returns
    -------
    result: float
    The interval raised to the specified power using interval arithmetic with 
    outwardly rounding.
    '''
    
    if lower < 0 and upper > 0 and pow % 1 == 0 and pow % 2 == 0:
        result = outRound(0, max(lower ** pow, upper ** pow), n)
        return result
    elif lower < 0 and pow % 1 != 0:
        print ('''ERROR. The “power” function cannot derive the non-integer 
            power of a negative number.''')
        return None
    elif pow < 0 and upper > 0 and lower < 0:
        print ('''ERROR. The “power” function cannot derive the interval hull 
            involving infinity.''')
        return None
    else:
        result = outRound(min(lower ** pow, upper ** pow), 
            max(lower ** pow, upper ** pow), n)
        return result

@cuda.jit
def times(lower, upper, val):
        
    '''
    The function "times" implements the interval arithmetic operation of 
    multiplying an interval and a real number with outwardly rounding.

    Parameters
    ----------
    lower: float
    The lower bound of the interval.
    upper: float
    The upper bound of the interval.
    val: float or int
    The real number to be multiplied with the interval.

    Returns
    -------
    result: float
    The interval multiplied with the real number using interval arithmetic with 
    outwardly rounding.
    '''

    result = outRound(min(lower * val, upper * val), \
        max(lower * val, upper * val), n)
    return result

@cuda.jit
def plus(lower, upper, val):

    '''
    The function "plus" implements the interval arithmetic operation of 
    adding an interval and a real number with outwardly rounding.

    Parameters
    ----------
    lower: float
    The lower bound of the interval.
    upper: float
    The upper bound of the interval.
    val: float or int
    The real number to be added with the interval.

    Returns
    -------
    result: float
    The interval added with a real number using interval arithmetic with 
    outwardly rounding.
    '''

    result = outRound(lower + val, upper + val, n)
    return result

@cuda.jit 
def minus(lower1, upper1, lower2, upper2):
    
    '''
    The function "minus" implements the interval arithmetic operation of 
    subtracting two intervals with outwardly rounding.

    Parameters
    ----------
    lower1: float
    The lower bound of the first interval.
    upper1: float
    The upper bound of the first interval.

    lower2: float
    The lower bound of the second interval.
    upper2: float
    The upper bound of the second interval.

    Returns
    -------
    result: float
    The subtraction of two intervals using interval arithmetic with outwardly 
    rounding.
    '''
    
    result = outRound (lower1-upper2, upper1-lower2, n)
    return result
    
@cuda.jit
def sine(lower, upper):
        
    '''
    Function "sine" implements the interval arithmetic operation of applying the
    sine function to the interval with outwardly rounding.

    Parameters
    ----------
    lower: float
    The lower bound of the interval.
    upper: float
    The upper bound of the interval.

    Returns
    -------
    result: float
    The interval after applying sine function using interval arithmetic with 
    outwardly rounding.
    '''

    lowerOut = min(math.sin(lower), math.sin(upper))
    upperOut = max(math.sin(lower), math.sin(upper))
    temp1 = lower // (math.pi / 2)
    temp2 = upper // (math.pi / 2)
    if (temp1 + 1) // 4 != (temp2 + 1) // 4:
        lowerOut = -1
    if (temp1 - 1) // 4 != (temp2 - 1) // 4:
        upperOut = 1
    result = outRound(lowerOut, upperOut, n)
    return result

@cuda.jit
def cosine(lower, upper):
    '''
    Function "cosine" implements the interval arithmetic operation of applying 
    the cosine function to the interval with outwardly rounding.

    Parameters
    ----------
    lower: float
    The lower bound of the interval.
    upper: float
    The upper bound of the interval.

    Returns
    -------
    result: float
    The interval after applying cosine function using interval arithmetic with 
    outwardly rounding.
    '''
    
    lowerOut = min(math.cos(lower), math.cos(upper))
    upperOut = max(math.cos(lower), math.cos(upper))
    temp1 = lower // (math.pi / 2)
    temp2 = upper // (math.pi / 2)
    if (temp1 + 2) // 4 != (temp2 + 2) // 4:
        lowerOut = -1
    if (temp1) // 4 != (temp2) // 4:
        upperOut = 1
    return outRound(lowerOut, upperOut, n)

@cuda.jit
def divide(lower1, upper1, lower2, upper2):
    
    '''
    Function "divide" implements the interval arithmetic operation of dividing 
    two intervals with outwardly rounding.

    Parameters
    ----------
    lower1: float
    The lower bound of the first interval.
    upper1: float
    The upper bound of the first interval.

    lower2: float
    The lower bound of the second interval.
    upper2: float
    The upper bound of the second interval.

    Returns
    -------
    result: float
    The division of two intervals using interval arithmetic with outwardly 
    rounding.
    '''
    
    if lower2 < 0 and upper2 > 0:
        result = -math.inf, math.inf
        return result
    elif lower2 == 0:
        if lower1 >= 0:
            result = lower1 / upper2, math.inf
            return result
        elif upper1 <= 0:
            result = -math.inf, upper1 / upper2
            return result
        else:
            result = -math.inf, math.inf
            return result
    elif upper2 == 0:
        if lower1 >= 0:
            result = -math.inf, lower1/lower2
            return result
        elif upper1 <= 0:
            result = upper1/lower2, math.inf
            return result
        else:
            result = -math.inf, math.inf
            return result
    else:
        temp1, temp2 = multiply(lower1, upper1, 1 / upper2, 1 / lower2)
        return outRound(temp1, temp2, n)