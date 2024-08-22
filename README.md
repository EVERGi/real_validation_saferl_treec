# real_validation_saferl_treec

Repository with the results and code to plot the figures of the paper "Real-world validation of safe reinforcement learning, model predictive control and decision tree-based home energy management systems".

The url of the paper is: <https://arxiv.org/pdf/2408.07435>

The energy contract used to calculate the electricity cost is the file `contract_rtp_tariff_april_2024.pdf`.

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To reproduce the figures of the paper, run the following command:

```bash
python plot_results/plot_all_figures.py
```