import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir=None, filename=None):
    """
    Load experiment results from CSV file(s).
    
    Parameters:
        results_dir: Directory containing result files
        filename: Specific file to load (if None, loads the most recent)
        
    Returns:
        DataFrame containing results
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    if filename is not None:
        # Load specific file
        file_path = os.path.join(results_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Results file {file_path} not found")
        return pd.read_csv(file_path)
    else:
        # Find all result files and load the most recent
        result_files = glob.glob(os.path.join(results_dir, '*.csv'))
        if not result_files:
            raise FileNotFoundError(f"No result files found in {results_dir}")
        
        # Sort by modification time (most recent last)
        result_files.sort(key=os.path.getmtime)
        return pd.read_csv(result_files[-1])


def calculate_statistics(results_df, true_ade=10, true_ape=5):
    """
    Calculate summary statistics for experiment results.
    
    Parameters:
        results_df: DataFrame containing experiment results
        true_ade: True average direct effect
        true_ape: True average peer effect
        
    Returns:
        Dictionary containing summary statistics
    """
    # Calculate bias
    ate = results_df['ADE'] + results_df['APE']

    ade_bias = results_df['ADE'].mean() - true_ade
    ape_bias = results_df['APE'].mean() - true_ape
    ate_bias = ate.mean() - true_ade - true_ape

    
    # Calculate MSE
    ade_mse = ((results_df['ADE'] - true_ade) ** 2).mean()
    ape_mse = ((results_df['APE'] - true_ape) ** 2).mean()
    ate_mse = ((ate - true_ade - true_ape) ** 2).mean()

    
    # Calculate standard deviations
    ade_std = results_df['ADE'].std()
    ape_std = results_df['APE'].std()
    ate_std = ate.std()

    
    # Calculate variance
    ade_var = results_df['ADE'].var()
    ape_var = results_df['APE'].var()
    ate_var = ate.var()

    
    # Calculate median runtime
    median_runtime = results_df['runtime'].median()
    
    return {
        'ade_mean': results_df['ADE'].mean(),
        'ape_mean': results_df['APE'].mean(),
        'ate_mean': ate.mean(),
        'ade_bias': ade_bias,
        'ape_bias': ape_bias,
        'ate_bias': ate_bias,
        'ade_mse': ade_mse,
        'ape_mse': ape_mse,
        'ate_mse': ate_mse,
        'ade_std': ade_std,
        'ape_std': ape_std,
        'ate_std': ate_std,
        'ade_var': ade_var,
        'ape_var': ape_var,
        'ate_var': ate_var,
        'median_runtime': median_runtime
    }


def plot_effect_distributions(results_df, true_ade=10, true_ape=5, save_dir=None):
    """
    Plot distributions of estimated effects.
    
    Parameters:
        results_df: DataFrame containing experiment results
        true_ade: True average direct effect
        true_ape: True average peer effect
        save_dir: Directory to save plots (if None, just displays)
        
    Returns:
        None
    """
    # Set up the figure
    results_df = results_df.replace([np.inf, -np.inf], np.nan)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot distribution of direct effects
    sns.histplot(results_df['ADE'], kde=True, ax=ax1)
    ax1.axvline(x=true_ade, color='r', linestyle='--', label=f'True effect ({true_ade})')
    ax1.axvline(x=results_df['ADE'].mean(), color='g', linestyle='-', label=f'Mean estimate ({results_df["ADE"].mean():.2f})')
    ax1.set_title('Distribution of Direct Effect Estimates')
    ax1.set_xlabel('Average Direct Effect (ADE)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Plot distribution of peer effects
    sns.histplot(results_df['APE'], kde=True, ax=ax2)
    ax2.axvline(x=true_ape, color='r', linestyle='--', label=f'True effect ({true_ape})')
    ax2.axvline(x=results_df['APE'].mean(), color='g', linestyle='-', label=f'Mean estimate ({results_df["APE"].mean():.2f})')
    ax2.set_title('Distribution of Peer Effect Estimates')
    ax2.set_xlabel('Average Peer Effect (APE)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'effect_distributions.png'), dpi=300)
    
    plt.show()


def plot_variance_matrix(variances, save_dir=None):
    """
    Visualize average variance-covariance matrix.
    
    Parameters:
        variances: List of variance matrices
        save_dir: Directory to save plots (if None, just displays)
        
    Returns:
        None
    """
    # Calculate average variance matrix
    avg_var = np.mean(variances, axis=0)
    
    # Create a DataFrame for better visualization
    var_df = pd.DataFrame(avg_var, 
                         index=['Direct Effect', 'Peer Effect'],
                         columns=['Direct Effect', 'Peer Effect'])
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(var_df, annot=True, cmap='coolwarm', fmt='.6f')
    plt.title('Average Variance-Covariance Matrix')
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'variance_matrix.png'), dpi=300)
    
    plt.show()


def compare_methods(results_files, method_names, true_ade=10, true_ape=5, save_dir=None):
    """
    Compare different methods/models based on experiment results.
    
    Parameters:
        results_files: List of file paths to results
        method_names: List of names for each method
        true_ade: True average direct effect
        true_ape: True average peer effect
        save_dir: Directory to save plots (if None, just displays)
        
    Returns:
        DataFrame with comparison metrics
    """
    if len(results_files) != len(method_names):
        raise ValueError("Number of result files must match number of method names")
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    methods_data = []
    
    # Load and calculate statistics for each method
    for file, name in zip(results_files, method_names):
        file_path = os.path.join(results_dir, file)
        df = pd.read_csv(file_path)
        stats = calculate_statistics(df, true_ade, true_ape)
        stats['method'] = name
        methods_data.append(stats)
    
    # Create DataFrame with all methods
    comparison_df = pd.DataFrame(methods_data)
    
    # Plotting comparison
    metrics = ['ade_bias', 'ape_bias', 'ade_mse', 'ape_mse', 'median_runtime']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5*len(metrics)))
    
    for i, metric in enumerate(metrics):
        sns.barplot(x='method', y=metric, data=comparison_df, ax=axes[i])
        axes[i].set_title(f'Comparison of {metric}')
        axes[i].set_xlabel('Method')
        
        # Better labels
        if 'bias' in metric:
            axes[i].set_ylabel('Bias (lower is better)')
        elif 'mse' in metric:
            axes[i].set_ylabel('MSE (lower is better)')
        elif 'runtime' in metric:
            axes[i].set_ylabel('Runtime in seconds (lower is better)')
    
    plt.tight_layout()
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'method_comparison.png'), dpi=300)
    
    plt.show()
    
    return comparison_df
