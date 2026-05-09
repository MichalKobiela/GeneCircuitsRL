Code for Sequential Design of Genetic Circuits Under Uncertainty with Reinforcement Learning (https://arxiv.org/pdf/2605.06552)
Michal Kobiela, Diego A. Oyarzun, Michael U. Gutmann
CC-BY license

ENVIRONMENT SETUP

1. Install Conda (Miniconda recommended):
https://docs.conda.io/en/latest/miniconda.html

2. Clone the repository

3. Create the environment:
conda env create -f environment.yml

4. Activate the environment:
conda activate circuitsrl

5. Instantiate the Julia environment:
Run, for example, the first cell of rl_repressilator.ipynb:
jl.Pkg.instantiate()

To run experiments, use the provided .ipynb Jupyter notebook files.

REPOSITORY STRUCTURE

- additional results: Results for the oscillator without epistemic uncertainty and a simplified case study with epistemic uncertainty
- datasets: Dataset for the regressor used in the growth cutoff study
- logs: Training information and saved RL policies
- models: Julia implementation of the repressilator and host-aware mechanistic models
- regressor_model.pth: Saved regressor for the host-aware case study
- regressor.ipynb: Training code for the host-aware regressor
- rl_host_cutoff.ipynb: Policy training for the host-aware growth cutoff study
- rl_host.ipynb: Policy training for the host-aware study without cutoff
- rl_repressilator.ipynb: Policy training for the repressilator model
- Manifest.toml, Project.toml: Julia environment files
- environment.yml: Conda environment definition
