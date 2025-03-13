import os
import argparse
import random
import numpy as np
import torch

from data_preprocess import cora_process, pubmed_process, sbm_process, find_focal_set
from experiment import run_experiment
from analysis import load_results, calculate_statistics, plot_effect_distributions, plot_variance_matrix


def main():
    """Main function to run experiments"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Network Causal Inference Experiments')
    
    # Dataset and model parameters
    parser.add_argument('--trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'pubmed', 'flickr', 'SBM', 'indianvillage'], 
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='GIN', choices=['GCN', 'GraphSage', 'GIN'],
                        help='GNN model to use')
    
    # Experiment parameters
    parser.add_argument('--gamma', type=float, default=0.25, help='Network confounding effect strength')
    parser.add_argument('--theta', type=int, default=10, help='True direct treatment effect')
    parser.add_argument('--alpha', type=int, default=5, help='True peer effect')
    parser.add_argument('--K', type=int, default=3, help='Number of folds for cross-validation')
    
    # System parameters
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to disk')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f'Starting experiment with dataset: {args.dataset}, model: {args.model}')
    print(f'Running {args.trials} trials with gamma={args.gamma}, theta={args.theta}, alpha={args.alpha}')
    
    # Load or generate dataset
    print('Processing dataset...')
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    if args.dataset == 'cora':
        features, matrix = cora_process(args)
    elif args.dataset == 'pubmed':
        features, matrix = pubmed_process(args)
    elif args.dataset == 'citeseer':
        features, matrix = citeseer_process(args)
    elif args.dataset == 'BC':
        matrix = BC_process(args)
    elif args.dataset == 'SBM':
        matrix = sbm_process(args)
    
    # Find focal set (or load from previously computed)
    focal_file = os.path.join(data_dir, f'focal_{args.dataset}.txt')
    if os.path.exists(focal_file):
        print(f'Loading focal set from {focal_file}')
        with open(focal_file, 'r') as f:
            s = f.read()[1:-1].split(', ')
            maximal_set = set([int(node) for node in s])
    else:
        print('Computing focal set...')
        maximal_set = find_focal_set(matrix)
        with open(focal_file, 'w') as f:
            f.write(str(maximal_set))
        print(f'Focal set saved to {focal_file}')
    
    print(f'Focal set size: {len(maximal_set)}')
    print(f'Number of edges: {matrix.sum() // 2}')
    
    # Run experiments
    results = run_experiment(args, matrix, maximal_set)
    
    # Load and analyze results
    results_df = load_results()
    stats = calculate_statistics(results_df, true_ade=args.theta, true_ape=args.alpha)
    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Save directory for plots
    if args.save_plots:
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
    else:
        plots_dir = None
    
    # Plot results
    # plot_effect_distributions(results_df, true_ade=args.theta, true_ape=args.alpha, save_dir=plots_dir)
    # plot_variance_matrix(results['variances'], save_dir=plots_dir)
    
    print("\nExperiment completed successfully!")


if __name__ == '__main__':
    main()
