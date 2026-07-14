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
Python script "IA.py” implements ten basic interval arithmetic operations
on GPU-based devices for interval evaluation.

As stated in the NVIDIA white paper [1], floating point data in CUDA-enabled 
GPUs can be stored in either 32-bit or 64-bit formats. This script implements 
the interval arithmetic operations in 64-bit format (double precision). Such 
implementation enables the input and output of 64-bit floating point data, 
which has around 16 significant digits. The outwardly rounded interval contains 
all possible results for an interval arithmetic operation based on the CUDA
C++ Programming Guide [2] and private communications with the NVIDIA CUDA Math 
API Team.

References:

[1] NVIDIA Corporation. 2025. Floating Point and IEEE 754 Compliance for NVIDIA 
GPUs. 
Available at: https://docs.nvidia.com/cuda/floating-point/index.html
[2] NVIDIA Corporation. 2025. CUDA C++ Programming Guide.
Avaliable at: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
[3] Private communications with the NVIDIA CUDA Math API Team.

'''

import numpy as np
from numba import cuda

@cuda.jit
def add(lower1, upper1, lower2, upper2):
   
    '''
    Function "add" implements the interval arithmetic operation of adding 
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
    
    lower = cuda.libdevice.dadd_rd(lower1, lower2)
    upper = cuda.libdevice.dadd_ru(upper1, upper2)
    return lower, upper

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
    
    lower = cuda.libdevice.dadd_rd(lower1, -upper2)
    upper = cuda.libdevice.dadd_ru(upper1, -lower2)
    return lower, upper

@cuda.jit
def multiply(lower1, upper1, lower2, upper2):

    '''
    Function "multiply" implements the interval arithmetic operation of 
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
 
    lower = min(cuda.libdevice.dmul_rd(lower1, lower2), 
        cuda.libdevice.dmul_rd(lower1, upper2), 
        cuda.libdevice.dmul_rd(upper1, lower2), 
        cuda.libdevice.dmul_rd(upper1, upper2))
    upper = max(cuda.libdevice.dmul_ru(lower1, lower2), 
        cuda.libdevice.dmul_ru(lower1, upper2), 
        cuda.libdevice.dmul_ru(upper1, lower2), 
        cuda.libdevice.dmul_ru(upper1, upper2))

    return lower, upper
       
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
        result = -np.inf, np.inf
        return result
    elif lower2 == 0:
        if lower1 >= 0:
            result = cuda.libdevice.ddiv_rd(lower1, upper2), np.inf
            return result
        elif upper1 <= 0:
            result = -np.inf, cuda.libdevice.ddiv_ru(lower1, upper2)
            return result
        else:
            result = -np.inf, np.inf
            return result
    elif upper2 == 0:
        if lower1 >= 0:
            result = -np.inf, cuda.libdevice.ddiv_ru(lower1, upper2)
            return result
        elif upper1 <= 0:
            result = cuda.libdevice.ddiv_rd(lower1, upper2), np.inf
            return result
        else:
            result = -np.inf, np.inf
            return result
    else:
        temp1, temp2 = multiply(lower1, upper1, 
            cuda.libdevice.drcp_rd(upper2), 
            cuda.libdevice.drcp_ru(lower2))
        return temp1, temp2
     
@cuda.jit
def power(lower, upper, pow):

    '''
    The function "power" implements the interval arithmetic operation of raising
    the power of an interval with outwardly rounding.
    Note: 
    [1] This function is NOT generally applicable. It cannot be applied to 
    the case where pow < 0, upper > 0, and lower < 0 (i.e., the interval 
    includes zero while the power is smaller than zero). 
    [2] When calculated lower or upper bounds for the operation equals to the
    smallest representable positive number (0.0) or largest representable 
    negative number (-0.0) in the type numpy.float64, functions "min" and "max" 
    do not work properly. Thus, this edge case is handled seperately in the 
    implementation.

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
        # Maximum ulp error = 2 for this operation in cuda.libdevice
        upperT = max(cuda.libdevice.pow(lower, pow), 
            cuda.libdevice.pow(upper, pow))
        upperT = upperT.view(np.int64)
        upperT = upperT + 5
        upperT = upperT.view(np.float64)
        return 0, upperT
    elif lower < 0 and pow % 1 != 0:
        print ('''ERROR. The “power” function cannot derive the non-integer 
            power of a negative number.''')
        return None
    elif pow < 0 and upper > 0 and lower < 0:
        print ('''ERROR. The “power” function cannot derive the interval hull 
            involving infinity.''')
        return None
    else:
        lowerT = min(cuda.libdevice.pow(lower, pow), 
            cuda.libdevice.pow(upper, pow))
        upperT = max(cuda.libdevice.pow(lower, pow), 
            cuda.libdevice.pow(upper, pow))
        
        if (lowerT < 0):
            lowerT = lowerT.view(np.int64)
            lowerT = lowerT + 5
            lowerT = lowerT.view(np.float64)
        elif (lowerT >= 0):
            lowerT = lowerT.view(np.int64)
            # Handle the case when lowerT = -0.0 in numpy.float64 type
            if (lowerT == -9223372036854775808):
                lowerT = -2.5e-323
            else:
                lowerT = lowerT - 5
                # If the value of lowerT in numpy.int64 type is smaller than 0,
                # the lowerT is set to -2.5e-323 which is equivalent to rounding 
                # down the value 0 five times in numpy.float64 type using 
                # "cuda.libdevice.nextafter".
                if (lowerT < 0):
                    lowerT = -2.5e-323
                else:
                    lowerT = lowerT.view(np.float64)    

        if (upperT >= 0):
            upperT = upperT.view(np.int64)
            # Handle the case when upperT = -0.0 in numpy.float64 type
            if (upperT == -9223372036854775808):
                upperT = 2.5e-323
            else:
                upperT = upperT + 5
                upperT = upperT.view(np.float64)
        elif (upperT < 0):
            upperT = upperT.view(np.int64)
            # If the value of upperT in numpy.int64 type is smaller than 
            # -9223372036854775803 (5 ulps from the smallest representable 
            # number with type numpy.int64), the upperT is set to 2.5e-323
            # which is equivalent to rounding up the value 0 five times in 
            # numpy.float64 type using "cuda.libdevice.nextafter".
            if (upperT < -9223372036854775803):
                upperT = 2.5e-323
            else:
                upperT = upperT - 5
                upperT = upperT.view(np.float64)     
        
        return lowerT, upperT
    
@cuda.jit
def times(lower, upper, val):
        
    '''
    Function "times" implements the interval arithmetic operation of 
    multiplying an interval and a real number with outwardly rounding.
    The function ONLY takes in positive values for the variable "val".

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

    lowerT = cuda.libdevice.dmul_rd(lower, val)
    upperT = cuda.libdevice.dmul_ru(upper, val)

    return lowerT, upperT

@cuda.jit
def plus(lower, upper, val):

    '''
    Function "plus" implements the interval arithmetic operation of 
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
    
    lower = cuda.libdevice.dadd_rd(lower, val)
    upper = cuda.libdevice.dadd_ru(upper, val)
    
    return lower, upper

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

    # Maximum ulp error = 2 for this operation in cuda.libdevice
    lowerOut = min(cuda.libdevice.sin(lower), cuda.libdevice.sin(upper))
    upperOut = max(cuda.libdevice.sin(lower), cuda.libdevice.sin(upper))

    if (lowerOut < 0):
        lowerOut = lowerOut.view(np.int64)
        lowerOut = lowerOut + 5
        lowerOut = lowerOut.view(np.float64)
        if (lowerOut < -1):
            lowerOut = -1
    elif (lowerOut >= 0):
        lowerOut = lowerOut.view(np.int64)
        lowerOut = lowerOut - 5
        # If the value of lowerOut in numpy.int64 type is smaller than 0,
        # the lowerOut is set to -2.5e-323 which is equivalent to rounding down
        # the value 0 five times in numpy.float64 type using 
        # "cuda.libdevice.nextafter".
        if (lowerOut < 0):
            lowerOut = -2.5e-323
        else:
            lowerOut = lowerOut.view(np.float64)   
    if (upperOut >= 0):
        upperOut = upperOut.view(np.int64)
        upperOut = upperOut + 5
        upperOut = upperOut.view(np.float64)
        if (upperOut > 1):
            upperOut = 1
    elif (upperOut < 0):
        upperOut = upperOut.view(np.int64)
        # If the value of upperOut in numpy.int64 type is smaller than 
        # -9223372036854775803 (5 ulps from the smallest representable number
        # with type numpy.int64), the upperOut is set to 2.5e-323 which is 
        # equivalent to rounding up the value 0 five times in numpy.float64 
        # type using "cuda.libdevice.nextafter".
        if (upperOut < -9223372036854775803):
            upperOut = 2.5e-323
        else:
            upperOut = upperOut - 5
            upperOut = upperOut.view(np.float64)     
    
    # The value \pi / 2 here is either in rounded down or up form to further 
    # ensure the interval arithmetic operation is rigorous due to the limited 
    # precision of \pi for calculating the corresponding segement on the sine 
    # function for the interval.
    
    if (lower >= 0):
        temp1 = lower // 1.570796326794897
    else:
        temp1 = lower // 1.5707963267948961
    if (upper >= 0):
        temp2 = upper // 1.5707963267948961
    else:
        temp2 = upper // 1.570796326794897

    # The if statement below determines if the evaluated value of the sine 
    # function for the input interval is NOT monotonically increasing or 
    # decreasing. 
    if (temp1 + 1) // 4 != (temp2 + 1) // 4:
        lowerOut = -1
    if (temp1 - 1) // 4 != (temp2 - 1) // 4:
        upperOut = 1
         
    return lowerOut, upperOut

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
    
    # Maximum ulp error = 2 for this operation in cuda.libdevice
    lowerOut = min(cuda.libdevice.cos(lower), cuda.libdevice.cos(upper))
    upperOut = max(cuda.libdevice.cos(lower), cuda.libdevice.cos(upper))  

    if (lowerOut < 0):
        lowerOut = lowerOut.view(np.int64)
        lowerOut = lowerOut + 5
        lowerOut = lowerOut.view(np.float64)
        if (lowerOut < -1):
            lowerOut = -1
    elif (lowerOut >= 0):
        lowerOut = lowerOut.view(np.int64)
        lowerOut = lowerOut - 5 
        # If the value of lowerOut in numpy.int64 type is smaller than 0,
        # the lowerOut is set to -2.5e-323 which is equivalent to rounding down
        # the value 0 five times in numpy.float64 type using 
        # "cuda.libdevice.nextafter".
        if (lowerOut < 0):
            lowerOut = -2.5e-323
        else:
            lowerOut = lowerOut.view(np.float64)   
    if (upperOut >= 0):
        upperOut = upperOut.view(np.int64)
        upperOut = upperOut + 5
        upperOut = upperOut.view(np.float64)
        if (upperOut > 1):
            upperOut = 1
    elif (upperOut < 0):
        upperOut = upperOut.view(np.int64)
        # If the value of upperOut in numpy.int64 type is smaller than 
        # -9223372036854775803 (5 ulps from the smallest representable number
        # with type numpy.int64), the upperOut is set to 2.5e-323 which is 
        # equivalent to rounding up the value 0 five times in numpy.float64 
        # type using "cuda.libdevice.nextafter".
        if (upperOut < -9223372036854775803):
            upperOut = 2.5e-323
        else:
            upperOut = upperOut - 5
            upperOut = upperOut.view(np.float64)     

    # The value \pi / 2 here is either in rounded down or up form to further 
    # ensure the interval arithmetic operation is rigorous due to the limited 
    # precision of \pi for calculating the corresponding segement on the cosine 
    # function for the interval.
    
    if (lower >= 0):
        temp1 = lower // 1.570796326794897
    else:
        temp1 = lower // 1.5707963267948961
    if (upper >= 0):
        temp2 = upper // 1.5707963267948961
    else:
        temp2 = upper // 1.570796326794897    
        
    # The if statement below determines if the evaluated value of the cosine 
    # function for the input interval is NOT monotonically increasing or 
    # decreasing. 
    if (temp1 + 2) // 4 != (temp2 + 2) // 4:
        lowerOut = -1
    if (temp1) // 4 != (temp2) // 4:
        upperOut = 1
        
    return lowerOut, upperOut

@cuda.jit
def exp(lower, upper):
    '''
    Function "exp" implements the interval arithmetic operation of applying 
    the exponential function to the interval with outwardly rounding.

    Parameters
    ----------
    lower: float
    The lower bound of the interval.
    upper: float
    The upper bound of the interval.

    Returns
    -------
    result: float
    The interval after applying exponential function using interval arithmetic 
    with outwardly rounding.
    '''
    
    # Maximum ulp error = 1 for this operation in cuda.libdevice
    lowerT = cuda.libdevice.exp(lower)
    upperT = cuda.libdevice.exp(upper)

    lowerT = lowerT.view(np.int64)
    lowerT = lowerT - 2
    # If the value of lowerT in numpy.int64 type is smaller than 0,
    # the lowerT is set to 5e-324 which is equivalent to rounding up
    # the value 0 once in numpy.float64 type using "cuda.libdevice.nextafter".
    if (lowerT < 0):
        lowerT = 5e-324
    else:
        lowerT = lowerT.view(np.float64)     
  
    upperT = upperT.view(np.int64)
    upperT = upperT + 2
    upperT = upperT.view(np.float64)
    
    return lowerT, upperT
