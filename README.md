# GDML: Graph Machine Learning based Doubly Robust Estimator for Network Causal Effects

This repository contains code for estimating causal effects in networked settings, where treatments and outcomes may be influenced by network connections.

## Overview

The project implements methods for estimating:
- Average Direct Effects (ADE) - the direct effect of a treatment on an individual's outcome
- Average Peer Effects (APE) - the effect of neighbors' treatments on an individual's outcome

The implementation uses Graph Neural Networks (GNNs) to model relationships in network data and estimates causal effects using a double machine learning approach.

## Project Structure

```
network-causal-inference/
├── README.md                 # This file
├── requirements.txt          # Package dependencies
├── src/                      # Source code
│   ├── __init__.py           # Package initialization
│   ├── data_generators.py    # Data generating processes (DGPs)
│   ├── data_processors.py    # Dataset loading/processing utilities
│   ├── dataset.py            # Custom PyTorch Geometric dataset
│   ├── models.py             # GNN model implementations
│   ├── experiment.py         # Experiment runner
│   ├── analysis.py           # Result analysis functions
│   └── main.py               # Main script
├── data/                     # Dataset directory (create this directory and place your data here)
├── results/                  # Experimental results (CSV)
└── plots/                    # Generated visualizations
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/username/network-causal-inference.git
cd network-causal-inference
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running an Experiment

To run an experiment with default settings:

```bash
python -m src.main
```

This will:
1. Load or generate the specified dataset
2. Run the specified number of trials
3. Train GNN models and estimate causal effects
4. Save results and generate plots

### Command-line Arguments

The following arguments can be used to customize the experiment:

```
--trials INT         Number of trials (default: 100)
--dataset STRING     Dataset name: 'cora', 'pubmed', 'citeseer', 'BC', 'SBM' (default: 'cora')
--model STRING       GNN model: 'GCN', 'GraphSage', 'GIN' (default: 'GIN')
--gamma FLOAT        Network effect strength (default: 0.25)
--theta INT             True direct treatment effect (default: 10)
--alpha INT           True peer effect (default: 5)
--K INT              Number of folds for cross-validation (default: 3)
--device INT         GPU device ID (default: 0)
--seed INT           Random seed (default: 0)
--save_plots         Save plots to disk
```

Example:
```bash
python -m src.main --dataset cora --model GIN --trials 50 --gamma 0.5 --save_plots
```

## Data Generating Processes

The code provides two data generating processes for synthetically simulating individual features within network: one with and one without non-linearity. You can use either of these dgps or add your dgp in `data_generative_process.py` file.



## Models

The implementation supports three Graph Neural Network architectures:

1. **GCN**: Graph Convolutional Network
2. **GraphSAGE**: Graph Sample and Aggregate
3. **GIN**: Graph Isomorphism Network

These models can be used for both treatment and outcome predictions.

## Datasets

The framework supports several standard network datasets:

- **Cora**: Citation network of computer science papers
- **Pubmed**: Citation network of medical papers
- **Flickr**: Network of images shared on Flickr
- **SBM**: Synthetic network from a Stochastic Block Model
- **Indian Village**: Survey data from villages in Karnataka, India used for investigating the impact of Self-Help Group participation on financial risk tolerance through outstanding loan

## License

[MIT License](LICENSE)
