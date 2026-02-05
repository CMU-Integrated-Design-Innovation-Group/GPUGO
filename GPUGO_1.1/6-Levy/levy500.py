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
Python script "Levy500.py” finds the global minimum of the Levy function using a 
GPU-based global optimization method. The Levy function F(X) is introduced as 
the Equation (15) in the paper written by Levy and Montalvo [1]. The Levy 
function has 10 variables (n = 500) in this script, and the value of each 
variable belongs to the range [-10, 10]. 

The GPU-based global optimization method is realized through an iterative 
process that rules out the suboptimal subregions and regions in the search 
domain where the global minimum cannot exist and leaves a finite set of regions 
where the global minimum could exist. Specifically, a list is created to include 
all regions in the search domain where the global minimum of the Levy function 
could exist, and the list is initialized with a single region that covers the 
whole search domain; during the iterative process, each iteration includes the 
four steps below.

Step 1 - Sampling the selected region: The selected region from the list is 
partitioned into many subregions. One or multiple sample points are chosen in 
each subregion, and interval evaluation of the Levy function is performed at 
each sample point. The smallest upper bound for the value of the Levy function 
among all sample points is used to update the upper bound of the global minimum 
of the Levy function. 

Step 2 - Ruling out suboptimal subregions in the selected region on the GPU: 
The suboptimal subregion(s) in the selected region are ruled out using the 
smallest upper bound of the global minimum derived from Step 1, and the 
generalized KKT conditions for all variables. The remaining subregion(s) are 
inserted into the list.

Step 3 - Ruling out suboptimal regions in the list on the CPU or GPU: The 
suboptimal region(s) in the list are ruled out using the smallest upper bound 
of the global minimum derived from Step 1. Step 2 and Step 3 could be performed 
at the same time if Step 3 is performed on the CPU. 

Step 4 - Checking stopping criteria: The region with the smallest lower bound 
for the value of the Levy function is selected and removed from the list. If 
the widths of the selected region do not satisfy the user-specified width 
tolerance, the selected region is employed for the next iteration. Otherwise, 
another region is selected and removed from the list until the list becomes 
empty. 

When the iterative process is halted, the 10-dimensional region(s) that satisfy 
the user-specified width tolerance, where the global minimum of the Levy 
function could exist, and the lower and upper bounds for the value of the 
global minimum of the Levy function are outputted as final results. 

This script utilizes 64 bits, double-precision float data, which has around 16 
significant digits according to IEEE 754. However, due to using outward 
rounding in the module IALevy, the result of this script has 15 or fewer 
significant digits. 

Reference:
[1] Levy, A. V., and Montalvo, A. 1985. The Tunneling Algorithm for the Global 
Minimization of Functions. SIAM Journal on Scientific and Statistical 
Computing, 6(1), 15-29. DOI: https://doi.org/10.1137/0906002.

'''

from numba import cuda
import numpy as np
import IALevy as ia
import math
import time

# Device (GPU) code
'''
================================================================================
Device functions: LevyValue_IA, LevySample_IA, LevyDerivative_IA, and 
selectVarInSubRegion.
================================================================================
''' 

@cuda.jit(device=True)
def LevyValue_IA (xValIA, pos, minUpper, flag, cycleIdx):
    
    '''
    The device function "LevyValue_IA" is called by the kernel function 
    “ruleOutGPU”. The device function computes the lower and upper bounds for 
    the value of the Levy function in a subregion and determines whether the 
    subregion should be ruled out based on the upper bound of the global 
    minimum of the Levy function. Specifically, the bounds for the value of 
    each variable within the subregion (the position of the subregion) are 
    first calculated using the absolute position of the thread in the grid that
    corresponds to the subregion, the bounds for the value of each variable 
    within the selected region (the position of the selected region), and the 
    cycling index of the selected region. The lower and upper bounds for the 
    value of the Levy function in the subregion are then derived using interval 
    arithmetic. If the lower bound is larger than the upper bound of the global
    minimum of the Levy function, the subregion is ruled out. Otherwise, the 
    subregion is not ruled out. 

    Parameters
    ----------
    xValIA: A 1-D NumPy array (numpy.float64) 
    The array represents the selected region in the current iteration. The 
    lower bounds and upper bounds for the values of variables are stored in the 
    order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i in the 
    function F(X), where i belongs to {1, 2, …, n}. In this script, n = 500.

    pos: int32
    The absolute position of a thread in the grid in GPU computing that
    corresponds to one subregion.

    minUpper: numpy.float64
    The upper bound of the global minimum of the Levy function F(X).

    flag: int32
    The flag equals 0 if the subregion is not ruled out before the device 
    function "LevyValue_IA" is called. The flag equals 1 if the subregion is 
    ruled out before the device function "LevyValue_IA" is called.

    cycleIdx: numpy.int32 
    The cycling index of the selected region.

    Returns
    -------
    fMin: numpy.float64
    The lower bound for the value of the Levy function F(X) in the subregion.

    flag: int32
    The flag equals 0 if the subregion is not ruled out. The flag equals 1 if 
    the subregion is ruled out.

    '''
    
    n = 500 # The number of variables in the Levy function

    # Interval evaluation of the Levy function in the subregion

    # Initialize the lower and upper bounds for the variables X_1 and X_n in 
    # the subregion
    # The selected region is partitioned into many subregions, and the device 
    # function “selectVarInSubRegion” returns the lower and upper bounds for 
    # the value of a specific variable (defined by 1 or n here) in a specific 
    # subregion (defined by “pos” here).

    lowerCurr, upperCurr = selectVarInSubRegion (xValIA, 1, pos, cycleIdx)
    # Initialize as X_n using lowerPrev and upperPrev to save the use of 
    # registers in GPU computing
    lowerPrev, upperPrev = selectVarInSubRegion (xValIA, n, pos, cycleIdx)

    # F_1 = (pi / 4) * x_1 operation in the code list
    fMin, fMax = ia.times(lowerCurr, upperCurr, math.pi / 4)
    # F_2 = F_1 + (3pi / 4) operation in the code list
    fMin, fMax = ia.plus(fMin, fMax, (3 * math.pi) / 4)
    # F_3 = sin(F_2) operation in the code list
    fMin, fMax = ia.sine(fMin, fMax)
    # F_4 = F_3 ^ 2 operation in the code list
    fMin, fMax = ia.power(fMin, fMax, 2)
    # F_5 = 10 * F_4 operation in the code list
    fMin, fMax = ia.times(fMin, fMax, 10)
    # F_6 = (1 / 4) * x_n operation in the code list
    f6Min, f6Max = ia.times(lowerPrev, upperPrev, 1 / 4)
    # F_7 = F_6 - 1 / 4 operation in the code list
    f6Min, f6Max = ia.plus(f6Min, f6Max, -1 / 4)
    # F_8 = F_7 ^ 2 operation in the code list
    f6Min, f6Max = ia.power(f6Min, f6Max, 2)
    # F_9 = F_5 + F_8 operation in the code list
    fMin, fMax = ia.add(f6Min, f6Max, fMin, fMax)
    
    # Initialize the sum of terms within the summation sign
    fsMin, fsMax = 0, 0 
    # Compute the sum of terms within the summation sign. 
    for i in range (1, n):
        # Update the lower and upper bounds for the values of variables X_i and 
        # X_(i+1) in each iteration
        lowerPrev, upperPrev = lowerCurr, upperCurr
        lowerCurr, upperCurr = selectVarInSubRegion (xValIA,i + 1,pos,cycleIdx)

        # F_s1 = (1 / 4) * x_i operation in the code list
        fsMinCur, fsMaxCur = ia.times(lowerPrev, upperPrev, 1 / 4)
        # F_s2 = F_s1 - 1 / 4 operation in the code list
        fsMinCur, fsMaxCur = ia.plus(fsMinCur, fsMaxCur, -1 / 4)
        # F_s3 = F_s2 ^ 2 operation in the code list
        fsMinCur, fsMaxCur = ia.power(fsMinCur, fsMaxCur, 2)
        # F_s4 = (pi / 4) * x_(i + 1) operation in the code list
        fs4Min, fs4Max = ia.times(lowerCurr, upperCurr, math.pi / 4)
        # F_s5 = F_s4 + (3pi / 4) operation in the code list
        fs4Min, fs4Max = ia.plus(fs4Min, fs4Max, (3 * math.pi) / 4)
        # F_s6 = sin(F_s5) operation in the code list
        fs4Min, fs4Max = ia.sine(fs4Min, fs4Max)
        # F_s7 = F_s6 ^ 2 operation in the code list
        fs4Min, fs4Max = ia.power(fs4Min, fs4Max, 2)
        # F_s8 = 10 * F_s7 operation in the code list
        fs4Min, fs4Max = ia.times(fs4Min, fs4Max, 10)
        # F_s9 = 1 + F_s8 operation in the code list
        fs4Min, fs4Max = ia.plus(fs4Min, fs4Max, 1)
        # F_s10 = F_s3 * F_s9 operation in the code list
        fsMinCur, fsMaxCur = ia.multiply(fsMinCur, fsMaxCur, fs4Min, fs4Max)      
        # F_10 = sum (F_s10) operation in the code list
        fsMin, fsMax = ia.add(fsMin, fsMax, fsMinCur, fsMaxCur)
    # F_11 = F_9 + F_10 operation in the code list
    fMin, fMax = ia.add(fsMin, fsMax, fMin, fMax)
    # f = (pi / n) * F_11 operation in the code list
    fMin, fMax = ia.times(fMin, fMax, math.pi / n)

    # Check whether the subregion should be ruled out.
    # The flag equals 0 if the subregion is not ruled out. The flag equals 1 if 
    # the subregion is ruled out.  
  
    if (fMin > minUpper):
        flag = 1
    return fMin, flag

@cuda.jit(device=True)
def LevySample_IA (xValIA, pos, num, cycleIdx):
    
    '''
    The device function "LevySample_IA" is called by the kernel function 
    “sampling”. The device function "LevySample_IA" samples over a subregion and 
    derives the smallest upper bound for the value of the Levy function among 
    all sample points in the subregion. Specifically, the bounds for the value 
    of each variable within the subregion (the position of the subregion) are 
    first calculated using the absolute position of the thread in the grid that 
    corresponds to the subregion, the bounds for the value of each variable 
    within the selected region (the position of the selected region), and the 
    cycling index of the selected region. User-specified number of sample 
    points are uniformly chosen along the diagonal of the subregion, where the 
    diagonal connects the two vertices of the subregion with the coordinates of 
    lower bounds and upper bounds for all variables, respectively. The lower 
    and upper bounds for the value of the Levy function at each sample point 
    are then computed using interval arithmetic. The smallest upper bound for 
    the value of the Levy function among all sample points is returned as the 
    result. 

    Parameters
    ----------
    xValIA: A 1-D NumPy array (numpy.float64) 
    The array represents the selected region in the current iteration. The 
    lower bounds and upper bounds for the values of variables are stored in the 
    order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i in the 
    function F(X), where i belongs to {1, 2, …, n}. In this script, n = 500.

    pos: int32
    The absolute position of a thread in the grid in GPU computing that 
    corresponds to one subregion.

    num: int32
    The number of sample points chosen in the subregion.

    cycleIdx: numpy.int32 
    The cycling index of the selected region.

    Returns
    -------
    fMax: numpy.float64
    The smallest upper bound for the value of the Levy function among all sample 
    points in the subregion.

    '''
    
    n = 500 # The number of variables in the Levy function

    # Initialize the value of the lowest upper bound among all sample points in 
    # the subregion
    fMaxOut = 0
    
    # A for loop iterate over the number of sample points
    for j in range (0, num):
        fMin, fMax = 0, 0
        # Initialize the lower and upper bounds for the variables X_1 and X_n
        lowerCurr, upperCurr = selectVarInSubRegion (xValIA, 1, pos, cycleIdx)
        # Initialize X_n using lowerPrev and upperPrev to save the use of 
        # registers in GPU computing
        lowerPrev, upperPrev = selectVarInSubRegion (xValIA, n, pos, cycleIdx)

        # Calculate the increment between two adjacent sample points
        # Compute the coordinates of the sample point based on the index j in 
        # the current iteration
 
        incrementCurr = (upperCurr - lowerCurr) / (num + 1)
        incrementPrev = (upperPrev - lowerPrev) / (num + 1)

        stepCurr = lowerCurr + incrementCurr * (j + 1)
        stepPrev = lowerPrev + incrementPrev * (j + 1)

        # F_1 = (pi / 4) * x_1 operation in the code list
        fMin, fMax = ia.times(stepCurr, stepCurr, math.pi / 4)
        # F_2 = F_1 + (3pi / 4) operation in the code list
        fMin, fMax = ia.plus(fMin, fMax, (3 * math.pi) / 4)
        # F_3 = sin(F_2) operation in the code list
        fMin, fMax = ia.sine(fMin, fMax)
        # F_4 = F_3 ^ 2 operation in the code list
        fMin, fMax = ia.power(fMin, fMax, 2)
        # F_5 = 10 * F_4 operation in the code list
        fMin, fMax = ia.times(fMin, fMax, 10)
        # F_6 = (1 / 4) * x_n operation in the code list
        f6Min, f6Max = ia.times(stepPrev, stepPrev, 1 / 4)
        # F_7 = F_6 - 1 / 4 operation in the code list
        f6Min, f6Max = ia.plus(f6Min, f6Max, -1 / 4)
        # F_8 = F_7 ^ 2 operation in the code list
        f6Min, f6Max = ia.power(f6Min, f6Max, 2)
        # F_9 = F_5 + F_8 operation in the code list
        fMin, fMax = ia.add(f6Min, f6Max, fMin, fMax)
        
        # Initialize the sum of terms within the summation sign
        fsMin, fsMax = 0, 0 
        # Compute the sum of terms within the summation sign. 
        for i in range (1, n):
            # Update the lower and upper bounds for the values of variables X_i 
            # and X_(i+1) in each iteration
            # Calculate the increment between two adjacent sample points
            # Compute the coordinates of the sample point based on the index j 
            # in the current iteration
            stepPrev = stepCurr
            lowerCurr,upperCurr = selectVarInSubRegion (xValIA,i+1,pos,cycleIdx)
            incrementCurr = (upperCurr - lowerCurr) / (num + 1)
            stepCurr = lowerCurr + incrementCurr * (j + 1)

            # F_s1 = (1 / 4) * x_i operation in the code list
            fsMinCur, fsMaxCur = ia.times(stepPrev, stepPrev, 1 / 4)
            # F_s2 = F_s1 - 1 / 4 operation in the code list
            fsMinCur, fsMaxCur = ia.plus(fsMinCur, fsMaxCur, -1 / 4)
            # F_s3 = F_s2 ^ 2 operation in the code list
            fsMinCur, fsMaxCur = ia.power(fsMinCur, fsMaxCur, 2)
            # F_s4 = (pi / 4) * x_(i + 1) operation in the code list
            fs4Min, fs4Max = ia.times(stepCurr, stepCurr, math.pi / 4)
            # F_s5 = F_s4 + (3pi / 4) operation in the code list
            fs4Min, fs4Max = ia.plus(fs4Min, fs4Max, (3 * math.pi) / 4)
            # F_s6 = sin(F_s5) operation in the code list
            fs4Min, fs4Max = ia.sine(fs4Min, fs4Max)
            # F_s7 = F_s6 ^ 2 operation in the code list
            fs4Min, fs4Max = ia.power(fs4Min, fs4Max, 2)
            # F_s8 = 10 * F_s7 operation in the code list
            fs4Min, fs4Max = ia.times(fs4Min, fs4Max, 10)
            # F_s9 = 1 + F_s8 operation in the code list
            fs4Min, fs4Max = ia.plus(fs4Min, fs4Max, 1)
            # F_s10 = F_s3 * F_s9 operation in the code list
            fsMinCur, fsMaxCur = ia.multiply(fsMinCur, fsMaxCur, fs4Min, fs4Max)      
            # F_10 = sum (F_s10) operation in the code list
            fsMin, fsMax = ia.add(fsMin, fsMax, fsMinCur, fsMaxCur)
        # F_11 = F_9 + F_10 operation in the code list
        fMin, fMax = ia.add(fsMin, fsMax, fMin, fMax)
        # f = (pi / n) * F_11 operation in the code list
        fMin, fMax = ia.times(fMin, fMax, math.pi / n)

        # The if statement is used to select the smallest upper bound for the 
        # value of the Levy function among all sample points within the 
        # subregion
        if (j == 0):
            # Initialize the value of fMaxOut in the first iteration
            fMaxOut = fMax
        else:
            # Update the value of fMaxOut to derive the smallest upper bound for 
            # the value of the Levy function among all sample points in the
            # subregion
            if (fMaxOut > fMax):
                fMaxOut = fMax

    return fMaxOut

@cuda.jit(device=True)
def LevyDerivative_IA (xdIA, pos, flagVal, cycleIdx):
    
    '''
    Device function "LevyDerivative_IA" is called by the kernel function 
    “ruleOutGPU”. The device function checks the generalized KKT conditions for 
    all the variables in a subregion and determines whether the subregion 
    should be ruled out. Specifically, since the first order derivative of the 
    objective function to any of the variables does not equal zero in the 
    search domain of the optimization problem, the device function checks 
    whether any of the related bound constraints could be active for each of 
    these variables. If the bound constraint of one variable cannot be active, 
    the subregion is ruled out. Otherwise, the subregion is not ruled out.

    Parameters
    ----------
    xdIA: A 1-D NumPy array (numpy.float64) 
    The array represents the selected region in the current iteration. The 
    lower bounds and upper bounds for the values of variables are stored in the 
    order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i in the 
    function F(X), where i belongs to {1, 2, …, n}. In this script, n = 500.

    pos: int32
    The absolute position of a thread in the grid in GPU computing that 
    corresponds to one subregion.

    flagVal: int32
    The flagVal equals 0 if the subregion is not ruled out before the device 
    function "LevyDerivative_IA" is called. The flagVal equals 1 if the 
    subregion is ruled out before the device function "LevyDerivative_IA" is 
    called.

    cycleIdx: numpy.int32 
    The cycling index of the selected region.

    Returns
    -------
    flag: int32
    The flag equals 0 if the subregion is not ruled out. The flag equals 1 if
    the subregion is ruled out.

    '''
    
    n = 500 # The number of variables in the Levy function

    flag = 0

    # Check the flag returned from sample-bound comparison
    if (flagVal == 1):
        flag = 1
        return flag
    
    # Initialize the lower and upper bounds for the value of the variables X_i, 
    # X_(i+1), and X_(i-1)
    lowerNext, upperNext = selectVarInSubRegion (xdIA, 2, pos, cycleIdx)
    lowerCurr, upperCurr = selectVarInSubRegion (xdIA, 1, pos, cycleIdx)
    lowerPrev, upperPrev = 0, 0

    # Code list for the first order derivative when i = 1
    
    # F_1 = (pi / 2) * x_1 operation in the code list
    dfMin, dfMax = ia.times(lowerCurr, upperCurr, math.pi / 2)
    # F_2 = F_1 + (3pi / 2) operation in the code list
    dfMin, dfMax = ia.plus(dfMin, dfMax, (3 * math.pi) / 2)
    # F_3 = sin(F_2) operation in the code list
    dfMin, dfMax = ia.sine(dfMin, dfMax)
    # F_4 = (5pi / 2) * F_3 operation in the code list
    dfMin, dfMax = ia.times(dfMin, dfMax, (5 * math.pi) / 2) 
    # F_5 = (pi / 4) * x_2 operation in the code list
    df5Min, df5Max = ia.times(lowerNext, upperNext, math.pi / 4)
    # F_6 = F_5 + 3pi / 4 operation in the code list
    df5Min, df5Max = ia.plus(df5Min, df5Max, (3 * math.pi) / 4)
    # F_7 = sin(F_6) operation in the code list
    df5Min, df5Max = ia.sine(df5Min, df5Max)
    # F_8 = F_7 ^ 2 operation in the code list
    df5Min, df5Max = ia.power(df5Min, df5Max, 2)
    # F_9 = 10 * F_8 operation in the code list
    df5Min, df5Max = ia.times(df5Min, df5Max, 10)
    # F_10 = 1 + F_9 operation in the code list
    df5Min, df5Max = ia.plus(df5Min, df5Max, 1)
    # F_11 = x_1 - 1 operation in the code list
    df11Min, df11Max = ia.plus(lowerCurr, upperCurr, -1)
    # F_12 = F_11 * F_10 operation in the code list
    df5Min, df5Max = ia.multiply(df11Min, df11Max, df5Min, df5Max)
    # F_13 = (1 / 8) * F_12 operation in the code list
    df5Min, df5Max = ia.times(df5Min, df5Max, 1 / 8)
    # F_14 = F4 + F_13 operation in the code list
    dfMin, dfMax = ia.add(dfMin, dfMax, df5Min, df5Max)
    # df_1 = (pi / n) * F_14 operation in the code list
    dfMin, dfMax = ia.times(dfMin, dfMax, math.pi / n)

    # Check the derivative to determine if the subregion could be ruled out
    if (dfMin > 0 and lowerCurr != -10):
        flag = 1
        return flag
    elif (dfMax < 0 and upperCurr != 10):
        flag = 1
        return flag

    # Code list for the first order derivatives when 1 < i < n
    for i in range (2, n):
        lowerPrev, upperPrev = lowerCurr, upperCurr
        lowerCurr, upperCurr = lowerNext, upperNext
        lowerNext, upperNext = selectVarInSubRegion (xdIA, i + 1, pos, cycleIdx)

        # F_1 = (pi / 2) * x_i operation in the code list
        dfMin, dfMax = ia.times(lowerCurr, upperCurr, math.pi / 2)
        # F_2 = F_1 + (3pi / 2) operation in the code list
        dfMin, dfMax = ia.plus(dfMin, dfMax, (3 * math.pi) / 2)
        # F_3 = sin(F_2) operation in the code list
        dfMin, dfMax = ia.sine(dfMin, dfMax)
        # F_4 = x_(i-1) - 1 operation in the code list
        df4Min, df4Max = ia.plus(lowerPrev, upperPrev, -1)
        # F_5 = F_4 ^ 2 operation in the code list
        df4Min, df4Max = ia.power(df4Min, df4Max, 2)
        # F_6 = F_3 * F_5 operation in the code list
        dfMin, dfMax = ia.multiply(dfMin, dfMax, df4Min, df4Max)
        # F_7 = (5pi / 32) * F_6 operation in the code list
        dfMin, dfMax = ia.times(dfMin, dfMax, (5 * math.pi) / 32)      
        # F_8 = (pi / 4) * x_(i+1) operation in the code list
        df7Min, df7Max = ia.times(lowerNext, upperNext, math.pi / 4)
        # F_9 = F_8 + 3pi / 4 operation in the code list
        df7Min, df7Max = ia.plus(df7Min, df7Max, (3 * math.pi) / 4)
        # F_10 = sin(F_9) operation in the code list
        df7Min, df7Max = ia.sine(df7Min, df7Max)
        # F_11 = F_10 ^ 2 operation in the code list
        df7Min, df7Max = ia.power(df7Min, df7Max, 2)
        # F_12 = 10 * F_11 operation in the code list
        df7Min, df7Max = ia.times(df7Min, df7Max, 10)
        # F_13 = 1 + F_12 operation in the code list
        df7Min, df7Max = ia.plus(df7Min, df7Max, 1)
        # F_14 = x_i - 1 operation in the code list
        df14Min, df14Max = ia.plus(lowerCurr, upperCurr, -1)
        # F_15 = F_14 * F_13 operation in the code list
        df7Min, df7Max = ia.multiply(df14Min, df14Max, df7Min, df7Max)
        # F_16 = (1 / 8) * F_15 operation in the code list
        df7Min, df7Max = ia.times(df7Min, df7Max, 1 / 8)
        # F_17 = F_7 + F_16 operation in the code list
        dfMin, dfMax = ia.add(dfMin, dfMax, df7Min, df7Max)
        # df_i = (pi / n) * F_17 operation in the code list
        dfMin, dfMax = ia.times(dfMin, dfMax, math.pi / n)

        # Check the derivative to determine if the subregion could be ruled out
        if (dfMin > 0 and lowerCurr != -10):
            flag = 1
            return flag
        elif (dfMax < 0 and upperCurr != 10):
            flag = 1
            return flag
        
    # Code list for the first order derivative when i = n
    # F_1 = (pi / 2) * x_n operation in the code list
    dfMin, dfMax = ia.times(lowerNext, upperNext, math.pi / 2)
    # F_2 = F_1 + (3pi / 2) operation in the code list
    dfMin, dfMax = ia.plus(dfMin, dfMax, (3 * math.pi) / 2)
    # F_3 = sin(F_2) operation in the code list
    dfMin, dfMax = ia.sine(dfMin, dfMax)
    # F_4 = x_(n-1) - 1 operation in the code list
    df4Min, df4Max = ia.plus(lowerCurr, upperCurr, -1)
    # F_5 = F_4 ^ 2 operation in the code list
    df4Min, df4Max = ia.power(df4Min, df4Max, 2)
    # F_6 = F_3 * F_5 operation in the code list
    dfMin, dfMax = ia.multiply(dfMin, dfMax, df4Min, df4Max)
    # F_7 = (5pi / 32) * F_6 operation in the code list
    dfMin, dfMax = ia.times(dfMin, dfMax, (5 * math.pi) / 32)  
    # F_8 = x_(n) - 1 operation in the code list
    df4Min, df4Max = ia.plus(lowerNext, upperNext, -1)
    # F_9 = (1 / 8) * F_8 operation in the code list
    df4Min, df4Max = ia.times(df4Min, df4Max, 1 / 8)
    # F_10 = F_9 + F_7 operation in the code list
    dfMin, dfMax = ia.add(df4Min, df4Max, dfMin, dfMax)
    # df_n = (pi / n) * F_10 operation in the code list
    dfMin, dfMax = ia.times(dfMin, dfMax, math.pi / n)

    # Check the derivative to determine if the subregion should be ruled out
    if (dfMin > 0 and lowerCurr != -10):
        flag = 1
        return flag
    elif (dfMax < 0 and upperCurr != 10):
        flag = 1
        return flag
    
    return flag

@cuda.jit(device=True)
def selectVarInSubRegion (xVal, num, pos, cycleIdx):

    '''
    The device function "selectVarInSubRegion" is called by device functions 
    "LevyValue_IA", "LevySample_IA", and "LevyDerivative_IA". The device 
    function "selectVarInSubRegion" computes the lower and upper bounds for the 
    value of the variable X_num in a specific subregion (defined by “pos”) 
    based on the cycling index of the selected region. Specifically, if the 
    variable is one of the ten variables whose ranges are partitioned, a while 
    loop is used to compute the subinterval index of the specific variable 
    X_num. Otherwise, the device function outputs the corresponding lower and 
    upper bounds for the value of the specific variable X_num from the array 
    that represents the selected region (i.e. the range of the specific 
    variable is not partitioned). 

    Parameters
    ----------
    xVal: A 1-D NumPy array (numpy.float64) 
    The array represents the selected region in the current iteration. The 
    lower bounds and upper bounds for the values of variables are stored in the 
    order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i in the 
    function F(X), where i belongs to {1, 2, …, n}. In this script, n = 500.

    num: int32
    The index of the variable X, where num is chosen from {1, 2, …, n}.

    pos: int32
    The absolute position of a thread in the grid in GPU computing that 
    corresponds to one subregion.

    cycleIdx: numpy.int32
    The cycling index of the selected region.

    Returns
    -------
    lowerX, upperX: numpy.float64
    The lower bound and upper bound for the value of the variable with the 
    selected index in the subregion. 

    '''
    
    length = 500 # The number of variables in the Levy function
    # The number of subintervals into which the range of each variable in the 
    # subregion is uniformly partitioned
    wid = 4

    # Initialize the number of variables for variable cycling
    cycleSize = 10
    
    # Checker1: Check if the cycling index is out of bound
    if (cycleIdx > length):
        print ("Cycling index is out of bound")
        return None
    
    # Checker2: Check if the index of the variable X_num (num) is out of bound
    if (num > length):
        print ("Index of variable X_i is out of bound")
        return None
    
    # The case when the range of the specific variable is partitioned based on 
    # the cycling index
    if (cycleIdx <= num and num < (cycleIdx + 10)):
        numloop = num - cycleIdx + 1

        # When numloop == num == 1, numloop is set to 2 since it is not 
        # necessary to compute the subinterval index for the first variable
        # X_cycleIdx through another iteration in the while loop. Function 
        # divmod() computes the index for X_cycleIdx at the same time when it 
        # is computing the subinterval index for X_(cycleIdx+1). The quotient is 
        # the subinterval index for X_(cycleIdx+1) and the remainder is the
        # subinterval index for X_cycleIdx.
        if (numloop == 1):
            numloop = 2
            
        # Initialize the counter for the number of iterations in the loop 
        i = cycleSize
            
        # Initialize the absolute position of the thread in the grid to be the 
        # remainder in the divmod() function for the first iteration
        rem = pos
        # Compute the index of the subregion for a given variable X_num with 
        # the absolute position of the thread through an iterative process
        while (i >= numloop):
            # The remainder from the previous iteration is fed back into the 
            # "divmod" function in the current iteration.
            (curr, rem) = divmod (rem, wid ** (i - 1))
            # When i == 2, divmod() computes the subregion index of X_cycleIdx 
            # and X_(cycleIdx+1) at the same time. The quotient is the 
            # subinterval index for X_(cycleIdx+1) and the remainder is the 
            # subinterval index for X_cycleIdx.
            # In the case, if num == cycleIdx, "curr" is replaced by "rem" for 
            # X_cycleIdx. 
            if (i == 2 and num == cycleIdx):
                curr = rem
            i -= 1 # Subtract the counter by 1 after each iteration
        
        '''
        Obtain the lower and upper bounds of the value of the variable X_num 
        using the subinterval index derived by the while loop
        ''' 
        lowIdx = (num - 1) * 2
        upIdx = (num - 1) * 2 + 1
        lowerX = ((xVal[upIdx] - xVal[lowIdx]) / wid) * curr + xVal[lowIdx]
        upperX = ((xVal[upIdx] - xVal[lowIdx]) / wid) * (curr + 1)+xVal[lowIdx]
    # The case when the range of the specific variable is not partitioned based 
    # on the cycling index
    else:
        lowIdx = (num - 1) * 2
        upIdx = (num - 1) * 2 + 1
        lowerX = xVal[lowIdx]
        upperX = xVal[upIdx]
    return lowerX, upperX

'''
================================================================================
Kernel functions: sampling, ruleOutGPU, and constructFlagArr.
================================================================================
'''

@cuda.jit
def sampling (region, output_lowestUpper, numSamples,cycIdx):

    '''
    The kernel function "sampling" partitions the selected region into many 
    subregions, chooses one or multiple sample points in each subregion, and 
    computes the smallest upper bound for the value of the Levy function among 
    all sample points in the selected region to update the upper bound of the 
    global minimum of the Levy function. Specifically, each thread in the same 
    block computes the smallest upper bound for the value of the Levy function 
    among all sample points in its corresponding subregion and stores the 
    smallest upper bound on an array on GPU shared memory. The smallest element 
    on the array is derived through a reduction tree. The smallest element on 
    the array is used to update the upper bound of the global minimum of the 
    Levy function (stored on GPU global memory) using atomic operations. 

    Parameters
    ----------
    region: A 1-D NumPy array (numpy.float64) 
    The array represents the selected region in the current iteration. The 
    lower bounds and upper bounds for the values of variables are stored in the 
    order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i in the 
    function F(X), where i belongs to {1, 2, …, n}. In this script, n = 500.

    output_lowestUpper: 1-D NumPy array (numpy.float64)
    The upper bound of the global minimum of the Levy function. The value is 
    initialized by the upper bound of the global minimum of the Levy function 
    derived from the last iteration.  

    numSamples: 1-D NumPy array (numpy.int32) 
    User-specified number of sample point(s) to be chosen from each subregion.

    cycIdx: 1-D NumPy array (numpy.int32) 
    The cycling index of the selected region. 

    Returns
    ------------
    N/A

    '''

    # Absolute position of a thread in the grid 
    pos = cuda.grid(1)
    # Thread index in the current block
    thr = cuda.threadIdx.x 
    # Threads per block of the grid
    blockDim = cuda.blockDim.x

    '''
    Create an array on GPU shared memory that stores the upper bounds for the 
    value of the Levy function derived by threads in the same block. The size 
    of the array on GPU shared memory needs to be consistent with the number of 
    threads per block in GPU computing.
    '''
    shr = cuda.shared.array(512, dtype = np.float64)

    # Compute the smallest upper bound for the value of the Levy function among 
    # all sample points in a subregion 
    shr[thr] = LevySample_IA (region, pos, numSamples[0], cycIdx[0])

    # Initialize stride for the reduction tree
    stride = int(blockDim / 2)
    while stride >= 1:
        '''
        Synchronize all threads in the same block
        '''
        cuda.syncthreads()
    
        # In each iteration, a thread remains active only when its index is 
        # smaller than the value of stride
        if thr < stride:
            if shr[thr + stride] < shr[thr]:
                shr[thr] = shr[thr + stride]
        # Values that need to be processed is halved in each iteration of the 
        # reduction tree
        stride = int(stride / 2)

    '''
    Thread 0 in each block is used to update the upper bound of the global 
    minimum of the Levy function using atomic operations.
    '''
    if thr == 0:
        cuda.atomic.min(output_lowestUpper, 0, shr[0]) 

@cuda.jit
def ruleOutGPU (region, result, atomic_index, lowestUpper, lowerBounds, cycIdx):

    '''
    The kernel function "ruleOutGPU" uses the upper bound of the global minimum 
    of the Levy function and the first order derivatives of the Levy function 
    to rule out the suboptimal subregions in the selected region where the 
    global minimum of the Levy function cannot exist. The kernel function 
    outputs the indices of the remaining subregions where the global minimum 
    could exist and the lower bounds for the value of the Levy function in 
    these remaining subregions. The index of each subregion is represented by 
    the absolute position of its corresponding thread in the grid in GPU 
    computing. For the rule-out process, the kernel function first calls the 
    device function "LevyValue_IA" to rule out suboptimal subregions based on 
    the upper bound of the global minimum of the Levy function. The device 
    function "LevyDerivative_IA" is then called to rule out suboptimal 
    subregions based on the first order derivatives of the Levy function. 

    Parameters
    ----------
    region: A 1-D NumPy array (numpy.float64) 
    The array represents the selected region in the current iteration. The 
    lower bounds and upper bounds for the values of variables are stored in the 
    order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i in 
    the function F(X), where i belongs to {1, 2, …, n}. In this script, n = 500.

    result: 1-D NumPy array (numpy.int32)
    The array stores the absolute positions of threads in the grid in GPU
    computing, where the corresponding subregions of these threads are not 
    ruled out. This array is initialized before the execution of the kernel 
    function. 

    atomic_index: 1-D NumPy array (numpy.int32)
    The counter counts the number of subregions to be stored in the result 
    array and the lowerBounds array. This one-element array is initialized 
    before the execution of the kernel function.

    lowestUpper: 1-D NumPy array (numpy.float64)
    The upper bound of the global minimum of the Levy function. This one-element 
    array is initialized before the execution of the kernel function.

    lowerBounds: 1-D NumPy array (numpy.float64)
    The array stores the lower bounds for the value of the Levy function in the 
    subregions that are not ruled out.

    cycIdx: 1-D NumPy array (numpy.int32) 
    The cycling index of the selected region.

    Returns
    ------------
    N/A

    '''

    # Absolute position of a thread in the grid 
    pos = cuda.grid(1)
    
    # Initialize the flag as 0 (the subregion is not ruled out when flag = 0)
    flag = 0

    # Compute the flag value to decide whether the subregion could be ruled out
    fMin, flag = LevyValue_IA (region, pos, lowestUpper[0], flag,cycIdx[0])
    # NOTE: Comment OUT the line if do NOT use derivative rule-out
    flag = LevyDerivative_IA (region, pos, flag, cycIdx[0])
    if flag == 0:
        # Atomic add is used to count the number of subregions in the 
        # selected region that are not ruled out. Notably, cuda.atomic.add 
        # returns the old value of atomic_index[0] before the addition. The 
        # value j is used to manage the indices of the result array and the 
        # lowerBounds array. 

        j = cuda.atomic.add(atomic_index, 0, 1) 

        result[j] = pos
        lowerBounds[j] = fMin

        # DEBUGGING STATEMENT: This print statement should only print once for 
        # all threads after the kernel function is launched. There is ONLY ONE 
        # subregion in which the lower bound for the value of the Levy function 
        # is 0.
        # if (fMin == 0):
            # print("The absolute position of the thread that has the lower")
            # print("bound of zero is:", pos)

@cuda.jit
def constructFlagArr (lowerBounds_list, flag, lowestUpper):

    '''
    The kernel function "constructFlagArr" computes a flag array, each element 
    in the flag array (0 or 1) indicates whether a region in the region list 
    should be ruled out, where the region list includes all remaining regions 
    from the previous iteration. If the lower bound for the value of the Levy 
    function in a region is smaller or equal to the upper bound of the global 
    minimum of the Levy function obtained from the current iteration, the 
    region is not ruled out. Otherwise, the region is ruled out.

    Parameters
    ----------
    lowerBounds_list: A 1-D NumPy array (numpy.float64)
    An array that stores the lower bounds for the value of the Levy function in 
    all regions in the region list derived from the previous iteration.

    flag: A 1-D NumPy array (numpy.int32)
    An array that stores the flag value (0 or 1) that indicates whether a 
    region is ruled out.

    lowestUpper: 1-D NumPy array (numpy.float64)
    The upper bound of the global minimum of the Levy function. This one-element 
    array is initialized before the execution of the kernel function.

    Returns
    ------------
    N/A

    '''
        
    # Absolute position of a thread in the grid 
    pos = cuda.grid(1)
    if (pos < lowerBounds_list.size): # Prevent out-of-bound access
        # If the region is not ruled out, the flag value is assigned as 0. 
        # Otherwise, the flag value is assigned as 1.
        if (lowerBounds_list[pos] <= lowestUpper[0]):
            flag[pos] = 0
        else:
            flag[pos] = 1

# Host code

'''
================================================================================
Host function: completeRegion.
================================================================================
'''

def completeRegion (xVal, pos, cycleIdx):

    '''
    The host function "completeRegion" partitions the selected region into many 
    subregions and outputs the lower and upper bounds for the value of each 
    variables X_i in a specified subregion as an array, where the subregion is 
    specified by the absolute position of its corresponding thread in the grid 
    in GPU computing. The host function uses the cycling index of the selected 
    region to determine whether the range of a variable should be partitioned. 
    Specifically, for the ten variables whose ranges are partitioned, a while 
    loop is used to compute the subinterval indices for these ten variables; 
    for the other variables whose ranges are not partitioned, the corresponding 
    elements in the output array are given the same values as that in the input 
    array represented the selected region.

    Parameters
    ----------
    xVal: A 1-D NumPy array (numpy.float64) 
    The array represents the selected region in the current iteration. The 
    lower bounds and upper bounds for the values of variables are stored in the 
    order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i in 
    the function F(X), where i belongs to {1, 2, …, n}. In this script, n = 500.

    pos: int
    The absolute position of a thread in the grid in GPU computing that 
    corresponds to one subregion.

    cycleIdx: int32
    The cycling index of the selected region.

    Returns
    -------
    xNewVal: A 1-D Numpy array (numpy.float64) 
    The array represents the specified subregion. The lower bounds and upper 
    bounds for the values of variables X_i in the function F(X)are computed 
    based on the absolute position of a thread in the grid and stored in the 
    array with the order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n].

    '''
    
    length = 500 # The number of variables in the Levy function
    # The number of subregions into which the range of each variable is 
    # uniformly partitioned 
    wid = 4 

    # Initialize the number of variables for variable cycling
    cycleSize = 10
    
    # Checker: Check if the cycling index is out of bound
    if (cycleIdx > length):
        print ("Cycling index is out of bound")
        return None
    
    # Initialize the counter for the number of iterations in the while loop 
    i = length
    
    # Initialize the absolute position of the thread in the grid to be the 
    # remainder in the divmod() function for the first iteration in the while 
    # loop
    rem = pos
    
    # Initialize the 1-D NumPy array that stores the lower and upper bounds for 
    # the value of all variables X_i in the subregion in the Levy function
    xNewVal = np.zeros(length * 2)
    
    # Initialize the counter for the variables correspond to the cycling index
    cycleCnt = cycleSize

    # Compute the lower and upper bounds for each variable X_i in the subregion 
    # based on the absolute position of its corresponding thread in GPU 
    # computing
    while (i >= 2):
        # The case when the range of a variable is partitioned based on the 
        # cycling index
        if (cycleIdx <= i and i < (cycleIdx + 10)):
            # The remainder from the previous iteration is fed back into the 
            # "divmod" function in the current iteration.
            (curr, rem) = divmod (rem, wid ** (cycleCnt - 1))
            # When i == 2, divmod() computes the subinterval index of 
            # X_cycleIdx and X_(cycleIdx+1) at the same time. The quotient is 
            # the subinterval index of X_(cycleIdx+1) and the remainder is the 
            # subinterval index of X_cycleIdx. 
            if (cycleCnt == 2):
                lowIdx = (i - 1) * 2
                upIdx = (i - 1) * 2 + 1
                lowerX = ((xVal[upIdx] - xVal[lowIdx]) / wid)*curr+xVal[lowIdx]
                upperX = ((xVal[upIdx]-xVal[lowIdx])/wid)*(curr+1)+xVal[lowIdx]
                xNewVal[upIdx] = upperX
                xNewVal[lowIdx] = lowerX
                lowIdx = (i - 2) * 2
                upIdx = (i - 2) * 2 + 1
                lowerX = ((xVal[upIdx] - xVal[lowIdx]) / wid) * rem+xVal[lowIdx]
                upperX = ((xVal[upIdx] - xVal[lowIdx])/wid)*(rem+1)+xVal[lowIdx]
                xNewVal[upIdx] = upperX
                xNewVal[lowIdx] = lowerX
                i -= 2 # Subtract the counter by 2 after the iteration
            else:
                '''
                Obtain and store the lower and upper bounds for the value of the
                variables X_i except X_(cycleIdx+1) and X_cycleIdx based on the 
                subinterval index computed for each variable
                ''' 
                lowIdx = (i - 1) * 2
                upIdx = (i - 1) * 2 + 1
                lowerX = ((xVal[upIdx] - xVal[lowIdx]) / wid)*curr+xVal[lowIdx]
                upperX = ((xVal[upIdx]-xVal[lowIdx])/wid)*(curr+1)+xVal[lowIdx]
                xNewVal[upIdx] = upperX
                xNewVal[lowIdx] = lowerX
                i -= 1 # Subtract the counter by 1 after each iteration
            cycleCnt -= 1

        # The case when the range of a variable is not partitioned based on the 
        # cycling index
        else:
            lowIdx = (i - 1) * 2
            upIdx = (i - 1) * 2 + 1
            xNewVal[lowIdx] = xVal[lowIdx]
            xNewVal[upIdx] = xVal[upIdx]

            i -= 1 # Subtract the counter by 1 after one iteration

            # Handle the case for the variable in the first dimension (X_1)
            if i == 1:
                lowIdx = (i - 1) * 2
                upIdx = (i - 1) * 2 + 1
                xNewVal[lowIdx] = xVal[lowIdx]
                xNewVal[upIdx] = xVal[upIdx]
        
    return xNewVal

# START OF MAIN CODE

print ("START OF THE ITERATIVE GLOBAL OPTIMIZATION PROCESS...")

# Define a NumPy array that represents the 500-dimensional search domain. 
# The NumPy array is initialized by the lower and upper bounds for the value of 
# the 500 variables in the Levy function. 

initialRegion = np.tile([-10, 10], 500)

start = time.time()

# Initialize the index of iteration 
iter = 0

# Initialize the maximum cycling index for the Levy function with 500 variables
maximumCyc = 491

# Initialize the selected_region_list array (stores the selected region to be
# analyzed during the iterative process)
selected_region_list = np.array([initialRegion], dtype = np.float64)

# Initialize FOUR arrays that are used to store the information of the 
# remaining regions that may include the global minimum of the Levy function

# Initialize the index_list array (stores the absolute positions of threads in 
# the grid of GPU computing that derive the remaining regions)
index_list = np.array([], dtype = np.int32)

# Initialize the lowerBounds_list array (stores the lower bounds for the values 
# of the Levy function in the remaining regions)
lowerBounds_list = np.array([], dtype = np.float64)

# Initialize the iteration_list array (stores the indices of the iteration 
# steps that derive the remaining regions)
iteration_list = np.array([], dtype = np.int32)

# Initialize the cycling_list array (stores the cycling index corresponds
# to each region that may include the global minimum of the Levy function)
cycling_list = np.array([], dtype = np.int32)

# Initialize the initial upper bound of the global minimum of the Levy function 
# as 1000
lowestUpperVal = 1000

# Initialize the output array (stores the region(s) within the search domain 
# that could contain the global minimum)
output = []

# Initialize the array that stores the lower bound(s) for the value of 
# the objective function. 
outputValLower = []

# Initialize the array that stores the upper bound(s) for the value of 
# the objective function. 
outputValUpper = []

# Define the width tolerance for the region(s) that are output as the final 
# result
widthTolerance = 1e-4

# User-specified number of sample point(s) to be chosen in each subregion
numSamples = 10

# Initialize the switch for the iterative process to obtain the region(s) that 
# could contain the global minimum of the Levy function
terminate = False

# Print the number of variables for the Levy function, width tolerance and 
# the number of samples for the sampling process
print("\n ---------------------------------------------------------------")
print("Number of variables for the Levy function:", 500)
print("Width tolerance:", widthTolerance)
print("Number of samples for sampling on GPU:", numSamples)
print(" ---------------------------------------------------------------")

# Select the GPU device to use on the server
# cuda.select_device(0)
    
# Start the iterative process
while (terminate == False):
    # Initialize the selected region for the first iteration 
    # iter = 0 in the first iteration 
    if (iter == 0):
        # Select the region in the first iteration: first element in the 
        # NumPy array selected_region_list
        selected_region = selected_region_list[0]

        # Initialize the cycling index in the first iteration as 1
        cycleIdxVal = 1

    # Store the lowestUpperVal from the previous iteration
    # LowestUpper is initialized as 1000 for the first iteration.
    lowestUpperPrev = lowestUpperVal

    # Initialize blocks per grid and threads per block for the 
    # FIRST kernel function, “sampling”, to sample over the selected region 
    # and update the upper bound of the global minimum of the Levy function 
    # using the smallest upper bound for the value of the Levy function 
    # among all sample points in the selected region
    blockspergridSample = 2048
    threadsperblockSample = 512

    # Transfer the selected region onto GPU global memory
    region_global = cuda.to_device(selected_region)

    # Transfer the upper bound of the global minimum of the Levy function 
    # onto GPU global memory
    output_lowestUpper = np.array([lowestUpperVal], dtype = np.float64)
    output_lowestUpper_global = cuda.to_device(output_lowestUpper) 

    # Transfer the number of sample point(s) to be chosen in each subregion 
    # onto GPU global memory
    numSamplesArr = np.array([numSamples], dtype = np.int32)
    numSamplesArr_global = cuda.to_device(numSamplesArr)

    # Transfer the cycling index onto GPU global memory

    cycleIdxArr = np.array([cycleIdxVal], dtype = np.int32)
    cycleIdxArr_global = cuda.to_device(cycleIdxArr)

    # Launch the FIRST kernel function, “sampling”, to sample over the 
    # selected region and update the upper bound of the global minimum of 
    # the Levy function 
    
    sampling [blockspergridSample, threadsperblockSample](region_global, 
        output_lowestUpper_global,numSamplesArr_global,
        cycleIdxArr_global)
            
    cuda.synchronize()

    # Transfer the array "output_lowestUpper" from device (GPU) to 
    # host (CPU) 
    lowestUpper_host = output_lowestUpper_global.copy_to_host()

    # Update the upper bound of the global minimum of the Levy function 
    lowestUpperVal = lowestUpper_host[0]  

    # DEBUGGING STATEMENT:
    np.set_printoptions(precision=20) 
    print("\n ---------------------------------------------------------------")
    print(f"At iteration {iter}:")
    print("The upper bound of the global minimum of the Levy function is", 
        lowestUpperVal)

    # Initialize blocks per grid and threads per block for the 
    # SECOND kernel function, “ruleOutGPU”, that rules out suboptimal 
    # subregions in the selected region
    blockspergridRule = 4096
    threadsperblockRule = 256

    # Calculate the total number of threads in the grid
    totalThread = blockspergridRule * threadsperblockRule

    # Initialize the atomic_index array (stores the number of subregions that 
    # are not ruled out) and the result array (stores the absolute positions of 
    # threads that correspond to subregions that are not ruled out)

    # atomic_index must be initialized as 0 at each iteration. Using 
    # cuda.device_array to initialize atomic_index leads to error during the
    # iterative process

    atomic_index = np.array([0], dtype = np.int32)

    atomic_index_global = cuda.to_device(atomic_index)
        
    # Initialize the result array on GPU global memory

    result_global = cuda.device_array(totalThread, dtype = np.int32)

    # Initialize the lowerBounds array which stores the lower bounds for the 
    # value of the Levy function in the subregions that could contain the 
    # global minimum 

    lowerBounds_global = cuda.device_array(totalThread, dtype = np.float64)

    # Initialize the atomic_lowestUpper array which stores the updated upper 
    # bound of the global minimum of the Levy function F(X) computed from the 
    # FIRST kernel function.

    lowestUpper_global = cuda.to_device([lowestUpperVal])

    # Transfer the selected region onto GPU global memory
    region_global = cuda.to_device(selected_region)

    # Transfer the cycling index onto GPU global memory
    cycleIdxArr_global = cuda.to_device(cycleIdxArr)

    # Launch the SECOND kernel function, “ruleOutGPU”, to rule out suboptimal 
    # subregions in the selected region

    ruleOutGPU [blockspergridRule, threadsperblockRule](region_global, 
        result_global, atomic_index_global, lowestUpper_global, 
        lowerBounds_global, cycleIdxArr_global)

    # Rule out the suboptimal regions in the region list derived from the 
    # previous iteration on CPU. If the number of remaining regions in the 
    # region list is less than or equal to 350000, the rule-out process takes 
    # place on the CPU. This CPU rule-out process runs simultaneously with the
    # SECOND kernel function, “ruleOutGPU”. Notably, if the updated upper bound 
    # of the global minimum of the Levy function computed in the current 
    # iteration is the same as that in the previous iteration, it is not 
    # necessary to perform this rule-out process.
    lengthLowerList = len(lowerBounds_list)

    if (iter > 0 and lowestUpperPrev != lowestUpperVal and 
        lengthLowerList <= 350000 and lengthLowerList > 0):
        flag_list = np.zeros(lengthLowerList)

        for i in range (0, len(lowerBounds_list)):
            if (lowerBounds_list[i] > lowestUpperVal):
                    flag_list[i] = 1
        lowerBounds_list = lowerBounds_list[flag_list == 0]
        index_list = index_list[flag_list == 0]
        iteration_list = iteration_list[flag_list == 0]
        cycling_list = cycling_list[flag_list == 0]
    
    # Synchronize all threads in the grid in GPU computing and the CPU 
    # computing to ensure that all threads in the GPU and the CPU complete 
    # their computation tasks 

    cuda.synchronize()

    # Transfer a part of the lowerBounds array that stores the lower bounds for 
    # the values of the Levy function in the subregions that are not ruled out 
    # from GPU to CPU

    lowerBounds_partial_global = lowerBounds_global[:atomic_index_global[0]]
    lowerBounds_host = lowerBounds_partial_global.copy_to_host()

    # Transfer a part of the result array that stores the absolute positions of 
    # the threads in the grid in GPU computing (these threads correspond to the 
    # subregions that are not ruled out) from GPU to CPU

    result_partial_global = result_global[:atomic_index_global[0]]
    result_host = result_partial_global.copy_to_host()

    # Rule out the suboptimal regions in the region list derived from the 
    # previous iteration on GPU. If the number of remaining regions in the 
    # region list is larger than 350000, the rule-out process takes place on 
    # the GPU using the THIRD kernel function, “constructFlagArr”. Notably, if 
    # the updated upper bound of the global minimum of the Levy function 
    # computed in the current iteration is the same as that in the previous 
    # iteration, it is not necessary to perform this rule-out process.

    if (iter > 0 and lowestUpperPrev != lowestUpperVal and
        lengthLowerList > 350000):

        # Initialize blocks per grid and threads per block for the 
        # THIRD kernel function, “constructFlagArr”, to rule out suboptimal 
        # regions in the region list
        threadsperblockFlag = 512
        blockspergridFlag = math.ceil(lengthLowerList / 512)

        # Initialize the flag array on GPU global memory
        flag_global = cuda.device_array(lengthLowerList, dtype = np.int32)
        lowerBounds_list_global = cuda.to_device(lowerBounds_list)
        lowestUpperVal_rule = np.array([lowestUpperVal], dtype = np.float64)
        lowestUpperVal_rule_global = cuda.to_device(lowestUpperVal_rule)

        # Launch the THIRD kernel function to construct the flag array, which 
        # is used to rule out suboptimal regions in the region list 
        constructFlagArr [blockspergridFlag, threadsperblockFlag](
            lowerBounds_list_global, flag_global,lowestUpperVal_rule_global)
        
        # Transfer the flag array from GPU to CPU
        flag_host = flag_global.copy_to_host()

        # Rule out suboptimal regions from the FOUR arrays, these FOUR arrays 
        # must be NumPy arrays
        lowerBounds_list = lowerBounds_list[flag_host == 0]
        index_list = index_list[flag_host == 0]
        iteration_list = iteration_list[flag_host == 0]
        cycling_list = cycling_list[flag_host == 0]

    # Derive the number of subregions that are not ruled out through the SECOND 
    # kernel function, “ruleOutGPU”
    length = len(result_host)

    # DEBUGGING STATEMENT: Print the number of subregions that are not ruled out
    # through the SECOND kernel function “ruleOutGPU” 
    # print("After the launch of the SECOND kernel function “ruleOutGPU”,") 
    # print("the number of subregions that are not ruled out is: ", length)

    # Update the FOUR arrays initialized to store remaining regions that could 
    # contain the global minimum of the Levy function by appending the 
    # corresponding information of the subregions that are not ruled out (lower 
    # bounds for the value of the Levy function in the current iteration, the 
    # index of the current iteration, the cycling index of the selected region, 
    # and the absolute positions of threads in the grid) after the CPU or GPU 
    # rule-out process.

    tempIter = np.full(length, iter)
    iteration_list = np.append (iteration_list, tempIter)

    tempCycling = np.full(length, cycleIdxVal)
    cycling_list = np.append (cycling_list, tempCycling)

    index_list = np.append (index_list, result_host)

    lowerBounds_list = np.append (lowerBounds_list, lowerBounds_host)
    
    # Increment the counter for the index of iteration
    iter += 1

    # Initialize the switch for the following iterative process to check the 
    # stopping criteria
    found = False

    lengthList = len(lowerBounds_list)

    # DEBUGGING STATEMENT: Print the number of regions remaining in the list
    print("Number of remaining regions in the current iteration:", lengthList)

    # The region with the smallest lower bound for the value of the Levy 
    # function in the region among all remaining regions in the region list is 
    # selected and then removed from the region list. If the width in each 
    # dimension of the selected region is smaller than the user-specified width 
    # tolerance, the region is added to the output array as the final result. 
    # If the width is larger than or equal to the width tolerance, the region 
    # is selected as the region to be analyzed in the next iteration. The 
    # iterative process to check the stopping criteria terminates when one of 
    # the two cases occurs: (1) the region selected for the next iteration is 
    # found; (2) the region is added to the output array, and the region list is 
    # empty.

    # Handle the case when the length of the remaining regions list is not 0
    if (lengthList != 0):
        while (found == False):
            # Find the index in the FOUR arrays that corresponds to the region 
            # with the smallest lower bound for the value of the Levy function 
            # in the region
            index_of_smallest = np.argmin(lowerBounds_list)

            # Obtain the absolute position of the thread, and the 
            # index of the iteration to reconstruct the selected region
            index = index_list[index_of_smallest]
            iteration = iteration_list[index_of_smallest]

            # The cycling index of each region keeps the same cycling index with 
            # the selected region in the current iteration here because the 
            # cycling index will be used to reconstruct the region if the region 
            # is selected for the next iteration. After the selected region is 
            # reconstructed, the cycling index of the selected region will be 
            # added by 10 before the next iteration. 
            cycleIdxVal = cycling_list[index_of_smallest]

            # Obtain the lower bound for the value of the Levy function in the 
            # selected region        
            lowerB = lowerBounds_list[index_of_smallest]

            # Remove the selected region from the FOUR arrays
            lowerBounds_list = np.delete(lowerBounds_list, index_of_smallest)
            index_list = np.delete(index_list, index_of_smallest)
            iteration_list = np.delete(iteration_list, index_of_smallest)
            cycling_list = np.delete(cycling_list, index_of_smallest)

            # Reconstruct the selected region
            # The lower and upper bounds for the value of each variable in the 
            # selected region are derived using the host function 
            # “completeRegion”.
            selected_region=completeRegion(selected_region_list[iteration],
                index, cycleIdxVal)

            # Calculate the width for the first dimension (X_1) of the selected 
            # region 
            width = selected_region[1] - selected_region[0]

            # DEBUGGING STATEMENT:
            # print ("The width of the current selected region is:", width)

            # If the upper bound of the global minimum used to rule out 
            # subregions in the current iteration is updated, the value is 
            # added to the outputValUpper list
            if (iter == 0 and lowestUpperVal != 1000):
                outputValUpper += [lowestUpperVal]
            elif (lowestUpperPrev != lowestUpperVal):
                outputValUpper += [lowestUpperVal]
                
            # Checker to determine whether the selected region should be added 
            # to the output array based on its width.
            # Note: For the Levy function with MORE THAN 10 variables, the width 
            # for each variable should be the same ONLY when the cycling index 
            # of the selected region at the current iteration equals the maximum 
            # cycling index.
            if (width < widthTolerance and cycleIdxVal == maximumCyc):
                # Add the region to the output array
                output += [selected_region]
                outputValLower += [lowerB]
                # Checker to determine if the computation should be terminated
                if (len(lowerBounds_list) == 0):
                    # Print the output array and the lower and upper bounds of
                    # the global minimum of the Levy function
                    np.set_printoptions(threshold=np.inf,
                        linewidth=np.inf, precision=20)
                    print ("\n ----GLOBAL MINIMUM OBTAINED----")
                    print('The regions that may include the global minimum are:'
                        , output)
                    print ("The length of the output array is:", len(output))

                    # Flip the switches of the checker and the iterative process 
                    # to end the whole computation
                    found = True
                    terminate = True
            else:
                # Flip the switch of the checker to True to indicate that the 
                # region for the next iteration has been found
                found = True

                # DEBUGGING STATEMENT:
                np.set_printoptions(precision=20) 
                print ("The selected region for the next iteration is: ", 
                    selected_region)

                # Append the selected region to the selected_region_list
                selected_regionCp = np.copy(selected_region)
                selected_regionCp = selected_regionCp[np.newaxis, :]
                selected_region_list = np.append (selected_region_list, 
                    selected_regionCp, axis = 0)

                # Update the cycling index for the next iteration
                # The cycling index circles back to 1 in the next iteration if 
                # it reaches the maximum value at the current iteration
                if cycleIdxVal == maximumCyc:
                    cycleIdxVal = 1
                    print ("The cycling index is:", cycleIdxVal)
                # If the cycling index at the current iteration does not equal 
                # the maximum value, the cycling index is added by 10 for the 
                # next iteration
                else:
                    cycleIdxVal += 10
                    print ("The cycling index is:", cycleIdxVal)
    # Handle the case when the length of the remaining regions list is 0 which
    # indicates the end of the iterative process
    else:
        # Flip the switch of the iterative process to end the whole computation
        terminate = True
        np.set_printoptions(threshold=np.inf,
            linewidth=np.inf, precision=20)
        print ("\n ----GLOBAL MINIMUM OBTAINED----")
        print ('The regions that may include the global minimum are:', output)  
        print ("The length of the output array is:", len(output))
  
end = time.time()

# Print the result of optimization
np.set_printoptions(precision = 20) 
print ("The global minimum of the objective function is:")
print(f"[{np.min(outputValLower)}, {np.min(outputValUpper)}]")

# Print the computation time
print("The time elapsed is:", end - start, "s")

# Print the total number of iterations
print("The total number of iterations is:", iter)

print ("\n ----END OF THE ITERATIVE GLOBAL OPTIMIZATION PROCESS----")

