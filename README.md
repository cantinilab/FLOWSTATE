# FLOWSTATE : Inferring Cell Fate Trajectories in Time-Resolved Metabolic RNA Labeling data
![Figure 1](doc/image/fig1.png)
FLOWSTATE is a trajectory inference method that leverages the information provided by metabollically labelled RNA. FLOWSTATE modeling cellular differentiation as the minimization of a potential function. Understanding this potential can elucidate the genes and transcription factors driving cellular evolution, reveal diverse potential trajectories, and forecast a cell’s future evolution based on its initial state

FlOWSTATE is built on the Scverse ecosystem, allowing seamless integration with pre-existing single-cell analysis tools such as Scanpy. It uses the JAX ecosystem for deep learning and efficient optimal transport computations. 
This folder contains the code used to train, evaluate, and analyze dynamical models of cell-state progression. The project combines velocity and potential-based modeling, optimal transport, and downstream biological interpretation.

## Install the package

- FLOWSTATE relies on JAX for fast GPU computations and OTT for Optimal Transport computations.
- **System requirements**: Python >= 3.12. 

### via PyPI
The easiest way to install FLOWSTATE is via PyPI. The installation should take around a minute.  
```bash
pip install flowstate
```

By default, JAX is installed for CPU. To get the GPU version, use 

```bash
pip install flowstate jax[cuda12]==0.4.30
```

refer to [JAX's docs](https://jax.readthedocs.io/en/latest/installation.html)). 

### via GitHub (development version)

```bash
git clone git@github.com:cantinilab/flowstate.git
pip install ./flowstate/
```


## Getting started
FlOWSTATE takes as an input an AnnData object, where omics information and labeled RNA information are stored in `obsm`, and `obs` contains time information.
here you can find a [tutorial] (https://github.com/cantinilab/FLOWSTATE/tutorial.ipynb)

The code to reproduce the figure of the paper can be found here https://github.com/cantinilab/FLOWSTATE_reproducibility