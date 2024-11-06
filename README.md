# real_validation_saferl_treec

Repository with the results and code to plot the figures and reproduce the simulation results of the paper "Real-world validation of safe reinforcement learning, model predictive control and decision tree-based home energy management systems".

The url of the paper is: <https://arxiv.org/pdf/2408.07435>

The energy contract used to calculate the electricity cost in the paper is provided in the file `contract_rtp_tariff_april_2024.pdf`.

## Installation

Large files are stored using Git LFS and installing Git LFS is required to clone the repository correctly. To install Git LFS, follow the instructions in the following link: <https://git-lfs.com/>.

To install the required packages, anaconda is recommended. The conda-forge channel is used to install the packages. 
To download anaconda, go to the following link: <https://www.anaconda.com/download/success>

First, create a conda environment with the required packages:

```bash
conda create --name real_saferl_treec --file requirements_conda.txt
```
Then install the local submodules using pip:

```bash
pip install -r requirements_pip.txt
```

## Structure of the repository

The repository is structured as follows:

- `data/`: Contains the data, ems models and results of the experiments.
- `src/`: Contains the code used to reproduce the simulations and figures of the paper.
- `submodules/`: Contains the energy grid simulator and TreeC code used in the experiments.

## Usage


To plot the figures from the paper:

```bash
python reproduce_paper.py --plot_paper_figures
```

To reproduce the results from the paper (Change values of `forecasting_model_dir`, `treec_model_dir`, and `pretrained_rl_dir` to use different models than the ones used in the paper):

```bash
python reproduce_paper.py --reproduce_paper_results
```

To plot the figures of the reproduced results:

```bash
python reproduce_paper.py --plot_new_reproduced_figures
```

To train new TreeC models with a very small population and few generations (Remove the `pop_size` and `gen` arguments to train with the values used in the paper):

```bash
python reproduce_paper.py --train_treec --pop_size 10 --gen 10
```

To train new forecasting models:

```bash
python reproduce_paper.py --train_forecasting
```

