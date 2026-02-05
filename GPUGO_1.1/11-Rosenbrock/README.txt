# Project Name

GPU-Based Global Optimization (GPUGO)

## File Descriptions

The folder “11-Rosenbrock” contains five Python scripts below: 

- IARosenbrock.py: The Python script “IARosenbrock.py” contains the 
interval arithmetic operations for evaluating the objective function 
(Rosenbrock Function). The functions in this script are called by the main 
runnable script.

- rosenbrock50.py: The Python script “rosenbrock50.py” is the main script that 
finds the global minimum of the Rosenbrock Function using the GPU-based global 
optimization method. The objective function has 50 variables in this script.

- rosenbrock100.py: The Python script “rosenbrock100.py” is the main script that 
finds the global minimum of the Rosenbrock Function using the GPU-based global 
optimization method. The objective function has 100 variables in this script.

- rosenbrock500.py: The Python script “rosenbrock500.py” is the main script that 
finds the global minimum of the Rosenbrock Function using the GPU-based global 
optimization method. The objective function has 500 variables in this script.

- rosenbrock1000.py: The Python script “rosenbrock1000.py” is the main script that 
finds the global minimum of the Rosenbrock Function using the GPU-based global 
optimization method. The objective function has 1000 variables in this script.

## Execution Instructions

The execution of the Python scripts in the folder “11-Rosenbrock” could follow the 
three steps below.

1. Set up the environment based on the Environment section below.
2. Ensure the “IARosenbrock.py” script is in the same directory as the main script.
3. Run the main script “rosenbrockXXXX.py” in Python.

## Environment

The Python scripts included in the folder “11-Rosenbrock” have been compiled and 
executed successfully in the following environment.

- Python Version: 3.10.11
- Numba Version: 0.60.0
- CUDA Toolkit Version: 12.4.131

## License
Copyright 2025 Carnegie Mellon University. All rights reserved.  
   
GPUGO is freely available for academic or non-profit organizations’ 
noncommercial research only. Please check the license file for further details. 
If you are interested in a commercial license, please contact CMU Center for 
Technology Transfer and Enterprise Creation at innovation@cmu.edu.



