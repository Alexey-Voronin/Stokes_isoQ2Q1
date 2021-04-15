# Low-order preconditioning of the Stokes equations

## Abstract 
A well-known strategy for building effective preconditioners for higher-order discretizations of some PDEs, such as Poisson's equation, is to leverage effective preconditioners for their low-order analogs. In this work, we show that high-quality preconditioners can also be derived for the Taylor-Hood discretization of the Stokes equations in much the same manner. In particular, we investigate the use of geometric multigrid based on the Q1isoQ2/Q1 discretization of the Stokes operator as a preconditioner for the Q2/Q1 discretization of the Stokes system. We utilize local Fourier analysis to optimize the damping parameters for Vanka and Braess-Sarazin relaxation schemes and to achieve robust convergence. These results are then verified and compared against the measured multigrid performance. While geometric multigrid can be applied directly to the Q2/Q1 system, our ultimate motivation is to apply algebraic multigrid within solvers for Q2/Q1 systems via the Q1isoQ2/Q1 discretization, which will be considered in a companion paper.

### Authors 
  - Alexey Voronin (voronin2@illinois.edu)
  - Yunhui He
  - Scott MacLachlan
  - Luke N. Olson
  - Raymond Tuminaro

The published paper can be found at
   - [arxiv](https://arxiv.org/abs/2103.11967)
   - TBD

This git reposetory contains code described in above mention publications. 

# Running the code

To be able to use this code, you will need 
   - [PyAMG](https://github.com/pyamg/pyamg)
   - [Firedrake](https://www.firedrakeproject.org/)

The code can be found in sysmg directory and all the relevant data collection scripts are in the data_collection folders. 
