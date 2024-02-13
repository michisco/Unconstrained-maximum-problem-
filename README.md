# Unconstrained maximum problem with SGD and BFGS algorithms

## Description of the problem
The problem of estimating the matrix norm $∥A∥_2$ for a (possibly rectangular) matrix $A∈R^{m×n}$, using its definition as an (unconstrained) maximum problem. Using:
- A standard gradient descent (steepest descent) approach.
- A quasi-Newton method such as BFGS.

## Structure
All scripts are located in the `src` folder. There is a MATLAB file to create random matrices with different dimensions and densities. <br />
In the `Data` folder, there are the matrices used for the experiments. <br />
Finally, the notebook file `SGD_BFGS.ipynb` shows the results achieved from the experiments. 
