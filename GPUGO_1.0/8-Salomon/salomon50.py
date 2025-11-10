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
Python script "salomon50.py” finds the global minimum of the Salomon 
function using a GPU-based global optimization method. The Salomon function 
F(X) is first introduced by Ralf Salomon in 1996 [1]. The dimension of the 
Salomon function is set as 50 in this script. In other words, the Salomon 
function has 50 variables, and the value of each variable belongs to the range 
[-100, 110]. 

The GPU-based global optimization method is realized through an iterative 
process that rules out the suboptimal subregions and regions in the search 
domain where the global minimum cannot exist and leaves a finite set of regions 
where the global minimum could exist. Specifically, a list is created to include 
all regions in the search domain where the global minimum of the objective 
function could exist, and the list is initialized with a single region that 
covers the whole search domain defined by the bound constraints. During the 
iterative process, each iteration includes the four steps below.

Step 1 - Sampling the selected region: The selected region from the list is 
partitioned into many subregions. One or multiple sample points are chosen in 
each subregion, and interval evaluation of the Salomon function is performed 
at each sample point. The smallest upper bound for the value of the Salomon 
function among all sample points is used to update the upper bound of the 
global minimum of the Salomon function. 

Step 2 - Ruling out suboptimal subregions in the selected region on the GPU: 
The suboptimal subregion(s) in the selected region are ruled out using the 
upper bound of the global minimum derived from Step 1, and the generalized 
KKT conditions for all variables. The remaining subregion(s) are inserted into 
the list.

Step 3 - Ruling out suboptimal regions in the list on the CPU or GPU: The 
suboptimal region(s) in the list are ruled out using the updated upper bound 
of the global minimum derived from Step 1. Step 2 and Step 3 could be executed 
in parallel if Step 3 is executed on the CPU. 

Step 4 - Checking stopping criteria: The region with the smallest lower bound 
for the value of the objective function is selected and removed from the list. 
If the width of the specified variable in the selected region does not satisfy 
the user-specified width tolerance, the selected region is employed for the next 
iteration. Otherwise, the region is inserted into the result array, and another 
region is selected and removed from the list until the list becomes empty. 

Notably, for the Salomon function with more than 10 variables, a variable 
cycling technique is employed. Specifically, when the selected region is 
partitioned into many subregions in Step 2 and Step 3, only 10 dimensions of 
the selected region is partitioned in turn at each iteration, instead of 
partitioning all its dimensions at the same time. 

When the iterative process is halted, the 50-dimensional region(s) that 
satisfy the user-specified width tolerance (region(s) in the result array), 
where the global minimum of the objective function could exist, and the lower 
and upper bounds for the value of the global minimum of the objective function 
are outputted. 

This script utilizes 64 bits, double-precision float data, which has around 16 
significant digits according to IEEE 754. However, due to using outward 
rounding in the module IASalomon, the result of this script has 15 or fewer 
significant digits. 

Reference:
[1] Salomon, R., (1996) "Re-evaluating Genetic Algorithm Performance Under 
Coordinate Rotation of Benchmark Functions: A Survey of Some Theoretical and 
Practical Aspects of Genetic Algorithms". BioSystems 39(3), 263-278.
https://doi.org/10.1016/0303-2647(96)01621-8

'''

from numba import cuda
import numpy as np
import IASalomon as ia
import math
import time

# Device (GPU) code
'''
================================================================================
Device functions: salomonValue_IA, salomonSample_IA, salomonDerivative_IA, 
and selectVarInSubRegion.
================================================================================
''' 

@cuda.jit(device=True)
def salomonValue_IA (xValIA, pos, minUpper, flag, cycleIdx):
    
    '''
    The device function "salomonValue_IA" is called by the kernel function 
    “ruleOutGPU”. The device function computes the lower and upper bounds for 
    the value of the Salomon function which determines if a subregion should 
    be ruled out using a code list. A code list defines the sequence of interval 
    arithmetic operations to evaluate the function. The bounds for variables 
    within the subregion in the function F(X) are calculated based on the 
    absolute position of the thread in the grid in GPU computing and the cycling
    index of the selected region. After the evaluation process using the code 
    list, the computed lower bound for the value of the function that belongs 
    to the subregion is compared with the smallest upper bound for the global 
    minimum to decide if the subregion should be ruled out from the space. 
    Specifically, if the lower bound is smaller than or equal to the smallest 
    upper bound, the subregion should be kept with the flag value of 0. If the 
    lower bound is larger than the smallest upper bound, the subregion should be 
    ruled out with the flag value of 1. 

    Parameters
    ----------
    xValIA: A 1-D NumPy array (numpy.float64) represents the selected region in 
    the current iteration step. The lower bounds and upper bounds are stored in 
    the order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i 
    in the function F(X), where i = 1, 2, … n. In this script, n = d = 50.

    pos: int32
    The absolute position of a thread in the grid in GPU computing.

    minUpper: numpy.float64
    The upper bound of the global minimum of the Salomon function F(X).

    flag: int32
    The flag equals 0 if the subregion is not ruled out. The flag equals 1 if 
    the subregion is ruled out.

    cycleIdx: numpy.int32 
    The cycling index of the selected region.
    
    Returns
    -------
    fMin: numpy.float64
    The lower bound for the value of the Salomon function F(X) in the 
    subregion.
    flag: int32
    The flag equals 0 if the subregion is not ruled out. The flag equals 1 if 
    the subregion is ruled out.

    '''
    
    d = 50 # The number of variables in the Salomon function

    # Interval evaluation of the Salomon function in the subregion
    
    # Initialize the sum of terms within the summation sign of (x_i) ^ 2
    fsMin, fsMax = 0, 0 

    # Compute the sum of terms within the summation sign of (x_i) ^ 2
    for i in range (1, d + 1):
        # Update the lower and upper bounds for the values of variables X_i in 
        # each iteration
        lowerX, upperX = selectVarInSubRegion (xValIA, i, pos, cycleIdx)
        # F_s1 = (x_i) ^ 2 operation in the code list
        fsMinCur, fsMaxCur = ia.power(lowerX, upperX, 2) 
        # F_1 = sum (F_s5) operation in the code list
        fsMin, fsMax = ia.add(fsMin, fsMax, fsMinCur, fsMaxCur)  
    # F_2 = sqrt(F_1) operation in the code list
    fMin, fMax = ia.power(fsMin, fsMax, 0.5)       
    # F_3 = 2 \pi F_2 operation in the code list
    f3Min, f3Max = ia.times(fMin, fMax, 2 * math.pi)  
    # F_4 = cos (F_3) operation in the code list
    f3Min, f3Max = ia.cosine(f3Min, f3Max)
    # F_5 = 0.1 F_2 operation in the code list
    fMin, fMax = ia.times(fMin, fMax, 0.1)   
    # F_6 = F_5 - F_4 operation in the code list
    fMin, fMax = ia.minus(fMin, fMax, f3Min, f3Max)  
    # F_7 = 1 + F_6 operation in the code list
    fMin, fMax = ia.plus(fMin, fMax, 1) 

    # Check whether the subregion should be ruled out.
    # The flag equals 0 if the subregion is not ruled out. The flag equals 1 if 
    # the subregion is ruled out.  
  
    if (fMin > minUpper):
        flag = 1
    return fMin, flag

@cuda.jit(device=True)
def salomonSample_IA (xValIA, pos, num, cycleIdx):
    
    '''
    The device function "salomonSample_IA" is called by the kernel function 
    “sampling”. The device function "salomonSample_IA" samples over a 
    subregion and derives the smallest upper bound for the value of the 
    Salomon function among all sample points in the subregion. Specifically, 
    the bounds for the value of each variable within the subregion (the 
    position of the subregion) are first calculated using the absolute 
    position of the thread in the grid that corresponds to the subregion, the 
    bounds for the value of each variable within the selected region (the 
    position of the selected region), and the cycling index of the selected 
    region. User-specified number of sample points are uniformly chosen along 
    the diagonal of the subregion, where the diagonal connects the two vertices 
    of the subregion with the coordinates of lower bounds and upper bounds for 
    all variables, respectively. The lower and upper bounds for the value of 
    the Salomon function at each sample point are then computed using 
    interval arithmetic. The smallest upper bound for the value of the 
    Salomon function among all sample points is returned as the result. 

    Parameters
    ----------
    xValIA: A 1-D NumPy array (numpy.float64) 
    The array represents the selected region in the current iteration. The 
    lower bounds and upper bounds for the values of variables are stored in the 
    order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i in the 
    function F(X), where i belongs to {1, 2, …, n}. In this script, n = d = 50.

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
    The smallest upper bound for the value of the Salomon function among all 
    sample points in the subregion.

    '''
    
    d = 50 # The number of variables in the Salomon function

    # Initialize the value of the lowest upper bound among all sample points in 
    # the subregion
    fMaxOut = 0
    
    # A for loop iterate over the number of sample points
    for j in range (0, num):
        # Initialize the sum of terms within the summation sign of (x_i) ^ 2
        fsMin, fsMax = 0, 0 

        # Compute the sum of terms within the summation sign of (x_i) ^ 2
        for i in range (1, d + 1):
            lowerCurr, upperCurr=selectVarInSubRegion (xValIA, i, pos, cycleIdx)

            # Calculate the increment between two adjacent sample points
            # Compute the coordinates of the sample point based on the index j 
            # in the current iteration
    
            incrementCurr = (upperCurr - lowerCurr) / (num + 1)

            stepCurr = lowerCurr + incrementCurr * (j + 1)
            
            # F_s1 = (x_i) ^ 2 operation in the code list
            fsMinCur, fsMaxCur = ia.power(stepCurr, stepCurr, 2) 
            # F_1 = sum (F_s5) operation in the code list
            fsMin, fsMax = ia.add(fsMin, fsMax, fsMinCur, fsMaxCur)  
        # F_2 = sqrt(F_1) operation in the code list
        fMin, fMax = ia.power(fsMin, fsMax, 0.5)       
        # F_3 = 2 \pi F_2 operation in the code list
        f3Min, f3Max = ia.times(fMin, fMax, 2 * math.pi)  
        # F_4 = cos (F_3) operation in the code list
        f3Min, f3Max = ia.cosine(f3Min, f3Max)
        # F_5 = 0.1 F_2 operation in the code list
        fMin, fMax = ia.times(fMin, fMax, 0.1)   
        # F_6 = F_5 - F_4 operation in the code list
        fMin, fMax = ia.minus(fMin, fMax, f3Min, f3Max)  
        # F_7 = 1 + F_6 operation in the code list
        fMin, fMax = ia.plus(fMin, fMax, 1)   

        # The if statement is used to select the smallest upper bound for the 
        # value of the Salomon function among all sample points within the 
        # subregion
        if (j == 0):
            # Initialize the value of fMaxOut in the first iteration
            fMaxOut = fMax
        else:
            # Update the value of fMaxOut to derive the smallest upper bound for 
            # the value of the Salomon function among all sample points in the
            # subregion
            if (fMaxOut > fMax):
                fMaxOut = fMax

    return fMaxOut

@cuda.jit(device=True)
def salomonDerivative_IA (xdIA, pos, flagVal, cycleIdx):
    
    '''
    Device function "salomonDerivative_IA" is called by the kernel function 
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
    xdIA: A 1-D NumPy array (numpy.float64) represents the selected region in 
    the current iteration step. The lower bounds and upper bounds are stored in 
    the order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i 
    in the function F(X), where i = 1, 2, … n. In this script, n = d = 50.

    pos: int32
    The absolute position of a thread in the grid in GPU computing.

    flagVal: int32
    The flag equals 0 if the subregion is not ruled out. The flag equals 1 if 
    the subregion is ruled out.

    cycleIdx: numpy.int32 
    The cycling index of the selected region.

    Returns
    -------
    flag: int32
    The flag equals 0 if the subregion is not ruled out. The flag equals 1 if
    the subregion is ruled out.

    '''
    
    d = 50 # The number of variables in the Salomon function

    flag = 0
    
    # Check the flag returned from sample-bound comparison
    if (flagVal == 1):
        flag = 1
        return flag

    identifier = True
    for k in range (1, d + 1):
        lowerX, upperX = selectVarInSubRegion (xdIA, k, pos, cycleIdx)
        if (lowerX > 0 or upperX < 0):
            identifier = False
            break
    if (identifier == True):
        return flag

    # Calculate the term sqrt(sum((x_i) ^ 2))
    
    # Initialize the sum of terms within the summation sign of (x_i) ^ 2
    fsMin, fsMax = 0, 0 

    # Compute the sum of terms within the summation sign of (x_i) ^ 2
    for i in range (1, d + 1):
        # Update the lower and upper bounds for the values of variables X_i in 
        # each iteration
        lowerX, upperX = selectVarInSubRegion (xdIA, i, pos, cycleIdx)
        # F_s1 = (x_i) ^ 2 operation in the code list
        fsMinCur, fsMaxCur = ia.power(lowerX, upperX, 2) 
        # F_1 = sum (F_s5) operation in the code list
        fsMin, fsMax = ia.add(fsMin, fsMax, fsMinCur, fsMaxCur)  
    # F_2 = sqrt(F_1) operation in the code list
    dtempMin, dtempMax = ia.power(fsMin, fsMax, 0.5)  
    
    # Calculate the first order derivatives of the Salomon Function            
    for j in range (1, d + 1):
        lowerX, upperX = selectVarInSubRegion (xdIA, j, pos, cycleIdx)
        
        # F_3 = 2 \pi F_2 operation in the code list
        dfMin, dfMax = ia.times(dtempMin, dtempMax, 2 * math.pi)     
        # F_4 = sin (F_3) operation in the code list
        dfMin, dfMax = ia.sine(dfMin, dfMax)    
        # F_5 = F_4 * x_j operation in the code list
        dfMin, dfMax = ia.multiply(dfMin, dfMax, lowerX, upperX)  
        # F_6 = 2 \pi F_5 operation in the code list
        dfMin, dfMax = ia.times(dfMin, dfMax, 2 * math.pi)  
        # F_7 = 0.1 * x_j operation in the code list
        df7Min, df7Max = ia.times(lowerX, upperX, 0.1)
        # F_8 = F_6 + F_7 operation in the code list
        dfMin, dfMax = ia.add(dfMin, dfMax, df7Min, df7Max)      
        # F_9 = F_8 / F_2 operation in the code list
        dfMin, dfMax = ia.divide(dfMin, dfMax, dtempMin, dtempMax)   

        # Check the derivative to determine if the subregion could be ruled out
        if (dfMin > 0 and lowerX != -100):
            flag = 1
            return flag
        elif (dfMax < 0 and upperX != 110):
            flag = 1
            return flag
        
    return flag


@cuda.jit(device=True)
def selectVarInSubRegion (xVal, num, pos, cycleIdx):

    '''
    The device function "selectVarInSubRegion" is called by device functions 
    "salomonValue_IA", "salomonSample_IA", and "salomonDerivative_IA". 
    The device function "selectVarInSubRegion" computes the lower and upper 
    bounds for the value of the variable X_num in a specific subregion (defined 
    by “pos”) based on the cycling index of the selected region. Specifically, 
    if the variable is one of the ten variables whose ranges are partitioned, 
    a while loop is used to compute the subinterval index of the specific 
    variable X_num. Otherwise, the device function outputs the corresponding 
    lower and upper bounds for the value of the specific variable X_num from 
    the array that represents the selected region (i.e. the range of the 
    specific variable is not partitioned). 

    Parameters
    ----------
    xVal: A 1-D NumPy array (numpy.float64) 
    The array represents the selected region in the current iteration. The 
    lower bounds and upper bounds for the values of variables are stored in the 
    order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i in the 
    function F(X), where i belongs to {1, 2, …, n}. In this script, n = d = 50.

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
    
    length = 50 # The number of variables in the Salomon function
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
def sampling (region, output_lowestUpper, numSamples, cycIdx):

    '''
    The kernel function "sampling" partitions the selected region into many 
    subregions, chooses one or multiple sample points in each subregion, and 
    computes the smallest upper bound for the value of the Salomon function 
    among all sample points in the selected region to update the upper bound of 
    the global minimum of the Salomon function. Specifically, each thread in 
    the same block computes the smallest upper bound for the value of the 
    Salomon function among all sample points in its corresponding subregion 
    and stores the smallest upper bound on an array on GPU shared memory. The 
    smallest element on the array is derived through a reduction tree. The 
    smallest element on the array is used to update the upper bound of the 
    global minimum of the Salomon function (stored on GPU global memory) 
    using atomic operations. 

    Parameters
    ----------
    region: A 1-D NumPy array (numpy.float64) 
    The array represents the selected region in the current iteration. The 
    lower bounds and upper bounds for the values of variables are stored in the 
    order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i in the 
    function F(X), where i belongs to {1, 2, …, n}. In this script, n = d = 50.

    output_lowestUpper: 1-D NumPy array (numpy.float64)
    The upper bound of the global minimum of the Salomon function. The value 
    is initialized by the upper bound of the global minimum of the Salomon 
    function derived from the last iteration.  

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
    value of the Salomon function derived by threads in the same block. The 
    size of the array on GPU shared memory needs to be consistent with the 
    number of threads per block in GPU computing.
    '''
    shr = cuda.shared.array(512, dtype = np.float64)

    # Compute the smallest upper bound for the value of the Salomon function 
    # among all sample points in a subregion 
    shr[thr] = salomonSample_IA (region, pos, numSamples[0], cycIdx[0])

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
    minimum of the Salomon function using atomic operations.
    '''
    if thr == 0:
        cuda.atomic.min(output_lowestUpper, 0, shr[0]) 
        
@cuda.jit
def ruleOutGPU (region, result, atomic_index, lowestUpper, lowerBounds, cycIdx):

    '''
    The kernel function "ruleOutGPU" rules out the suboptimal subregions in the 
    selected region where the global minimum cannot exist. The kernel function 
    outputs the indices of the remaining subregions where the global minimum 
    could exist and the lower bounds for the value of the Salomon function in 
    these remaining subregions. The index of each subregion is represented by 
    the absolute position of the thread in the grid in GPU computing. For the 
    rule-out process, the kernel function first calls the device function 
    "salomonValue_IA" which rules out suboptimal subregions through computing 
    the lower bound for the value of the function. Then, the device function 
    "salomonDerivative_IA" is called to further rule out suboptimal subregions 
    through computing the first order derivatives of the function. Specifically, 
    the flag equals 0 if the subregion is not ruled out. The flag equals 1 
    if the subregion is ruled out. 

    Parameters
    ----------
    region: A 1-D NumPy array (numpy.float64) represents the selected region 
    in the current iteration step. The lower bounds and upper bounds are stored 
    in the order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i 
    in the function F(X), where i = 1, 2, …, n. In this script, n = d = 50.

    result: 1-D NumPy array (numpy.int32)
    The array stores the absolute positions of threads in the grid in GPU
    computing, where the corresponding subregions of these threads may contain 
    the global minimum of the Salomon function F(X). This array is initialized 
    before the execution of the kernel function. 

    atomic_index: 1-D NumPy array (numpy.int32)
    The counter counts the number of subregions to be stored in the result 
    array and the lowerBounds array. This one-element array is initialized 
    before execution of the kernel function.

    lowestUpper: 1-D NumPy array (numpy.float64)
    The upper bound of the global minimum of the function F(X). This one-element 
    array is initialized before execution of the kernel function.

    lowerBounds: 1-D NumPy array (numpy.float64)
    The array stores the lower bounds for the value of the Salomon function in 
    the subregions that are not ruled out.

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

    # Compute the flag value to decide whether the subregion should be ruled out
    fMin, flag = salomonValue_IA (region,pos,lowestUpper[0],flag,cycIdx[0])
    # NOTE: Comment OUT the line if do NOT use derivative rule-out
    flag = salomonDerivative_IA (region, pos, flag, cycIdx[0])
    if flag == 0:
        # Atomic add is used to count the number of subregions in the 
        # selected region that may include the global minimum of F(X). Notably, 
        # cuda.atomic.add returns the old value of atomic_index[0] before the 
        # addition. The value j is used to manage the indeices of the result 
        # array and the lowerBounds array. 

        j = cuda.atomic.add(atomic_index, 0, 1) 

        result[j] = pos
        lowerBounds[j] = fMin

        # DEBUGGING STATEMENT: this print statement should only print once in
        # each iteration step. There is ONLY ONE subregion that will give the 
        # lower bound of the value of the function as 0.
        # if (fMin == 0):
            # print("The absolulte position of the thread that has the lower")
            # print("bound of zero is:", pos)

@cuda.jit
def constructFlagArr (lowerBounds_list, flag, lowestUpper):

    '''
    The kernel function "constructFlagArr" outputs a flag array, each element 
    in the flag array (0 or 1) indicates whether a region in the region list 
    should be ruled out, where the region list includes all remaining regions 
    from the previous iteration step. If the lower bound of the region is 
    smaller or equal to the lowest upper bound obtained from the current 
    iteration, the region is assigned the value of 0 at the corresponding 
    indices in the array to indicate that the region is not ruled out. 
    If the lower bound of the region is larger than the lowest upper bound, the 
    value of 1 is assigned to indicate that the region is ruled out.

    Parameters
    ----------
    lowerBounds_list: A 1-D NumPy array (numpy.float64)
    An array that stores the lower bounds of all regions in the region list 
    accumulated from previous iteration steps.

    flag: A 1-D NumPy array (numpy.int32)
    An array that stores the information (0 or 1) that determines whether a 
    region is ruled out.

    lowestUpper: 1-D NumPy array (numpy.float64)
    The upper bound of the global minimum of the function F(X). This one-element 
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
    order [LB_1, UB_1, LB_2, UB_2, ..., LB_n, UB_n] for the variables X_i in the 
    function F(X), where i belongs to {1, 2, …, n}. In this script, n = d = 50.

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
    
    length = 50 # The number of variables in the Salomon function
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
    # the value of all variables X_i in the subregion in the Salomon function
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

# Define a NumPy array that represents the 50-dimensional search domain. 
# The NumPy array is initialized by the lower and upper bounds for the value of 
# the 50 variables in the Salomon function. 

initialRegion = np.tile([-100, 110], 50)

start = time.time()

# Initialize number of iteration steps
iter = 0

# Initialize the maximum cycling index for the Salomon function with 50 
# variables
maximumCyc = 41

# Initialize the region_list array (stores the selected region for each
# iteration step of the iterative rule-out process)
region_list = np.array([initialRegion], dtype = np.float64)

# Initialize FOUR arrays that are used to store the information of the 
# remaining regions in the iterative rule-out process

# Initialize the index_list array (stores the absolute positions of threads
# that may contain the global minimum of the function F(X))
index_list = np.array([], dtype = np.int32)

# Initialize the lowerBounds_list array (stores the lower bounds for the 
# values of the F(X) for regions possibly containing the global minimum of the 
# function F(X))
lowerBounds_list = np.array([], dtype = np.float64)

# Initialize the iteration_list array (stores the indices of the iteration 
# step for regions that may contain the global minimum of function F(X))
iteration_list = np.array([], dtype = np.int32)

# Initialize the cycling_list array (stores the cycling index corresponds
# to each region that may include the global minimum of the Salomon function)
cycling_list = np.array([], dtype = np.int32)

# Initialize the initial smallest upper bound for the global minimum as 1000000
lowestUpperVal = 1000000

# Initialize the output array (stores the region(s) within the search domain 
# that could contain the global minimum)
output = []

# Initialize the array that stores the lower bound(s) for the value of 
# the objective function. 
outputValLower = []

# Initialize the array that stores the upper bound(s) for the value of 
# the objective function. 
outputValUpper = []

# Define the width tolerance for the resulting region(s) that could contain 
# the global minimum of the Salomon function F(X)
widthTolerance = 1e-4

# User-specified number of samples to take for the sampling process for finding
# the smallest upper bound for the global minimum in each iteration step within
# each subregion
numSamples = 10

# Initialize the switch for the iterative rule-out process to obtain the 
# region(s) that could contain the global minimum
terminate = False

# Print the number of variables for the Salomon function, width tolerance and 
# the number of samples for the sampling process
print("\n ---------------------------------------------------------------")
print("Number of variables for the Salomon function:", 50)
print("Width tolerance:", widthTolerance)
print("Number of samples for sampling on GPU:", numSamples)
print(" ---------------------------------------------------------------")

# Select the GPU device to use on the server
# cuda.select_device(0)
  
# Start the iterative process
while (terminate == False):
    # Initialize the selected region for the first iteration step
    # iter = 0 in the first iteration step
    if (iter == 0):
        # Select the region in the first iteration step: first element in the 
        # NumPy array region_list
        selected_region = region_list[0]

        # Initialize the cycling index in the first iteration as 1
        cycleIdxVal = 1

    # Store the lowestUpperVal from the previous iteration
    # LowestUpper is initialized as 1000000 for the first iteration.
    lowestUpperPrev = lowestUpperVal
   
    # Initialize blocks per grid and threads per block for the 
    # FIRST kernel function, “sampling”, to sample over the selected region 
    # and update the upper bound of the global minimum of the Salomon 
    # function using the smallest upper bound for the value of the Salomon 
    # function among all sample points in the selected region
    blockspergridSample = 2048
    threadsperblockSample = 512

    # Transfer the selected region onto GPU global memory
    region_global = cuda.to_device(selected_region)

    # Transfer the upper bound of the global minimum of the Salomon function 
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
    # the Salomon function 
    
    sampling [blockspergridSample, threadsperblockSample](region_global, 
        output_lowestUpper_global,numSamplesArr_global, cycleIdxArr_global)
            
    cuda.synchronize()

    # Transfer the array "output_lowestUpper" from device (GPU) to 
    # host (CPU) 
    lowestUpper_host = output_lowestUpper_global.copy_to_host()

    # Update the upper bound of the global minimum of the Salomon function 
    lowestUpperVal = lowestUpper_host[0] 

    # DEBUGGING STATEMENT:
    np.set_printoptions(precision=20) 
    print("\n ---------------------------------------------------------------")
    print(f"At iteration {iter}:")
    print("The smallest upper bound selected is", lowestUpperVal)

    # Initialize blocks per grid and threads per block for the 
    # SECOND kernel function that rules out suboptimal subregions in the 
    # selected region
    blockspergridRule = 4096
    threadsperblockRule = 256

    # Calculate the total number of threads in the grid
    totalThread = blockspergridRule * threadsperblockRule

    # Initialize the atomic_index array (stores the number of subregions that 
    # are not ruled out in the current iteration step) and the result array 
    # (stores the absolute positions of threads that correspond to subregions 
    # possibly containing the global minimum of the Salomon function F(X))

    # atomic_index must be initialized as 0 at each iteration step. Using 
    # cuda.device_array to initialize atomic_index leads to error during the
    # iteration process

    atomic_index = np.array([0], dtype = np.int32)

    atomic_index_global = cuda.to_device(atomic_index)
        
    # Initialize the result array on GPU global memory

    result_global = cuda.device_array(totalThread, dtype = np.int32)

    # Initialize the lowerBounds array which stores the lower bounds for the 
    # value of the Salomon function in the subregions that could contain the 
    # global minimum 

    lowerBounds_global = cuda.device_array(totalThread, dtype = np.float64)

    # Initialize the atomic_lowestUpper array which stores the updated upper 
    # bound of the global minimum of the Salomon function F(X) computed from 
    # the FIRST kernel function.

    lowestUpper_global = cuda.to_device([lowestUpperVal])

    # Transfer the selected region onto GPU global memory
    region_global = cuda.to_device(selected_region)

    # Transfer the cycling index onto GPU global memory
    cycleIdxArr_global = cuda.to_device(cycleIdxArr)

    # Launch the SECOND kernel function to rule out suboptimal subregions in 
    # the selected region

    ruleOutGPU [blockspergridRule, threadsperblockRule](region_global, 
        result_global, atomic_index_global, lowestUpper_global, 
        lowerBounds_global, cycleIdxArr_global)

    # Rule out the suboptimal regions in the region list derived from the 
    # previous iteration step on CPU. If the number of regions in the region 
    # list is less than or equal to 350000, the rule-out process will take 
    # place on the CPU. This method runs simultaneously with the SECOND kernel 
    # function. Notably, if the lowest upper bound computed in the current 
    # iteration is the same as the previous iteration, such rule-out process is 
    # not necessary.
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
    # the values of the Salomon function from GPU to CPU

    lowerBounds_partial_global = lowerBounds_global[:atomic_index_global[0]]
    lowerBounds_host = lowerBounds_partial_global.copy_to_host()

    # Transfer a part of the result array that stores the absolute positions of 
    # the threads in the grid in GPU computing from GPU to CPU

    result_partial_global = result_global[:atomic_index_global[0]]
    result_host = result_partial_global.copy_to_host()

    # Rule out the suboptimal regions in the region list derived from the 
    # previous iteration step on GPU. If the number of regions in the region 
    # list is larger than 350000, the rule-out process will be performed on 
    # the GPU using the THIRD kernel function. Notably, if the lowest upper 
    # bound computed in the current iteration is the same as the previous 
    # iteration, the such rule-out process is not necessary.

    if (iter > 0 and lowestUpperPrev != lowestUpperVal and
        lengthLowerList > 350000):

        # Initialize blocks per grid and threads per block for the 
        # THIRD kernel function
        threadsperblockFlag = 512
        blockspergridFlag = math.ceil(lengthLowerList / 512)

        # Initialize the flag array on GPU global memory
        flag_global = cuda.device_array(lengthLowerList, dtype = np.int32)
        lowerBounds_list_global = cuda.to_device(lowerBounds_list)
        lowestUpperVal_rule = np.array([lowestUpperVal], dtype = np.float64)
        lowestUpperVal_rule_global = cuda.to_device(lowestUpperVal_rule)

        # Launch the THIRD kernel function to construct the flag array for 
        # ruling out regions on the list
        constructFlagArr [blockspergridFlag, threadsperblockFlag](
            lowerBounds_list_global, flag_global,lowestUpperVal_rule_global)
        
        # Transfer the flag array from GPU to CPU
        flag_host = flag_global.copy_to_host()

        # Rule out regions on all FOUR arrays, arrays have to be NumPy 
        # arrays
        lowerBounds_list = lowerBounds_list[flag_host == 0]
        index_list = index_list[flag_host == 0]
        iteration_list = iteration_list[flag_host == 0]
        cycling_list = cycling_list[flag_host == 0]

    # Derive the number of subregions that are not ruled out through the SECOND 
    # kernel function, “ruleOutGPU”
    length = len(result_host)

    # Update the FOUR arrays initialized to store remaining regions that could 
    # contain the global minimum of the Salomon function by appending the 
    # corresponding information of the subregions that are not ruled out (lower 
    # bounds for the value of the Salomon function in the current iteration, 
    # the index of the current iteration, the cycling index of the selected 
    # region, and the absolute positions of threads in the grid) after the CPU 
    # or GPU rule-out process.

    tempIter = np.full(length, iter)
    iteration_list = np.append (iteration_list, tempIter)

    tempCycling = np.full(length, cycleIdxVal)
    cycling_list = np.append (cycling_list, tempCycling)

    index_list = np.append (index_list, result_host)

    lowerBounds_list = np.append (lowerBounds_list, lowerBounds_host)
    
    # Increment the counter for counting the number of iterations
    iter += 1

    # Initialize the switch for the following iterative process to check the 
    # stopping criteria
    found = False

    lengthList = len(lowerBounds_list)

    # DEBUGGING STATEMENT: Print the number of regions remaining in the list
    print("Number of remaining regions in the current iteration:", lengthList)

    # The region with the smallest lower bound for the value of the Salomon 
    # function among all regions in the region list is selected and then 
    # removed from the region list. If the width in each dimension of the 
    # selected region is smaller than the user-defined width tolerance, the 
    # region is added to the output array as a region that may contain the 
    # global minimum. If the width is not smaller than the width tolerance, the 
    # region is selected as the region to be partitioned and analyzed in the 
    # next iteration step. The iterative process terminates when one of the two 
    # cases occurs: (1) the region selected for the next iteration step is 
    # found; (2) the region is added to the output array, and there is no 
    # region remained in the region list to be processed.
    
    # Handle the case when the length of the remaining regions list is not 0
    if (lengthList != 0):
        while (found == False):
            # Find the index in the FOUR arrays that stores the smallest lower
            # bound for the value of the function F(X)
            index_of_smallest = np.argmin(lowerBounds_list)

            # Obtain the absolute position of the thread, and the 
            # index of the iteration step to reconstruct the selected region
            index = index_list[index_of_smallest]
            iteration = iteration_list[index_of_smallest]

            # The cycling index of each region keeps the same cycling index with 
            # the selected region in the current iteration here because the 
            # cycling index will be used to reconstruct the region if the region
            # is selected for the next iteration. After the selected region is 
            # reconstructed, the cycling index of the selected region will be 
            # added by 10 before the next iteration. 
            cycleIdxVal = cycling_list[index_of_smallest]

            # Obtain the lower bound for the value of the Salomon function in 
            # the selected region
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
            selected_region = completeRegion (region_list[iteration], index,
                cycleIdxVal)

            # Calculate the width in the first dimension of the selected region 
            width = selected_region[1] - selected_region[0]

            # DEBUGGING STATEMENT:
            # print ("The width of the current selected region is:", width)

            # If the upper bound of the global minimum used to rule out 
            # subregions in the current iteration is updated, the value is 
            # added to the outputValUpper list
            if (iter == 0 and lowestUpperVal != 1000000):
                outputValUpper += [lowestUpperVal]
            elif (lowestUpperPrev != lowestUpperVal):
                outputValUpper += [lowestUpperVal]
                
            # Checker to determine if the region should be added to the output 
            # array based on the width calculated.
            # Note: For the Salomon function with MORE THAN 10 variables, the 
            # width for each variable should be the same ONLY when the cycling 
            # index of the selected region at the current iteration equals the 
            # maximum cycling index.
            if (width < widthTolerance and cycleIdxVal == maximumCyc):
                # Add the region to the output array
                output += [selected_region]
                outputValLower += [lowerB]
                # Checker to determine if the computation should be terminated
                if (len(lowerBounds_list) == 0):
                    # Print the output array and the lower and upper bounds of
                    # the global minimum of the function
                    np.set_printoptions(threshold=np.inf,
                        linewidth=np.inf, precision=20)
                    print ("\n ----GLOBAL MINIMUM OBTAINED----")
                    print('The regions that may include the global minimum are:'
                        , output)
                    print ("The length of the output array is:", len(output))

                    # Flip the switches of the checker and the iterative 
                    # rule-out process to end the whole computation
                    found = True
                    terminate = True
            else:
                # Flip the switch of the checker to True to indicate that the 
                # region for the next iteration has been found
                found = True

                # DEBUGGING STATEMENT:
                np.set_printoptions(precision=20) 
                print ("The selected region for the next iteration step is: ", 
                    selected_region)

                # Append the selected region to the region_list
                selected_regionCp = np.copy(selected_region)
                selected_regionCp = selected_regionCp[np.newaxis, :]
                region_list = np.append (region_list,selected_regionCp,axis = 0)

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

# Print the total number of iteration steps
print("The total number of iterations is:", iter)

print ("\n ----END OF THE ITERATIVE GLOBAL OPTIMIZATION PROCESS----")

