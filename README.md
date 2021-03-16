# InvUQ
### A simple code for Bayesian uncertainty quantification for inverse problems
For use with Monte Carlo datasets


The code consists of two parts. The first part is Prior Falsification with PCA. This is used to verify that your prior Monte Carlo data matches the main trends of your observed data.

The second part is Posterior Sampling. To use the EmpiricalPosteriorSampling function, you must create an inverse mapping from data to model parameters. The inverse mapping is then applied to a set of prior data realizations (not used in model training) to quantify uncertainty in the inverse mapping. The set of true model parameters from the prior, inverse-mapped model parameters from the prior, and inverse mapped model parameters from the data can then be used to generate fast posterior distributions. 

See the notebook titled "Seismic Example with Random Forest Inverse Model" for an example of both portions of the code.

