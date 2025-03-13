import os
import time
import random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv

from models import GCN, GraphSAGE, GIN
from data_preprocess import find_train_split, CustomDataset
from data_generative_process import dgp



def run_experiment(args, matrix, maximal_set):
    """
    Run causal inference experiment on network data.
    
    Parameters:
        args: Command-line arguments
        matrix: Adjacency matrix
        maximal_set: Set of focal nodes
        
    Returns:
        results: Dictionary containing experiment results
    """
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    n_covariates = 1
    
    ADEs = []  # Average Direct Effects
    APEs = []  # Average Peer Effects
    runtime = []  # Runtime for each trial
    variances = []  # Variance estimates
    n_coverage_theta, n_coverage_alpha = 0, 0  # Coverage counters
    avg_std = 0
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run trials
    for tr in range(args.trials):
        start = time.time()
        print(f'Running trial {tr+1}/{args.trials}')
        
        data, df_true, X = dgp(matrix, args.theta, args.alpha, args.gamma)

        
        J0 = np.zeros((2, 2))  # Information matrix
        var = np.zeros((2, 2))  # Variance matrix
        inds = list(maximal_set)
        
        print(f'Length of each fold: {len(inds) // args.K}')
        random.shuffle(inds)
        splits = [inds[len(inds) * i // args.K:len(inds) * (i + 1) // args.K] for i in range(args.K)]
        
        theta = 0  # Direct effect
        alpha = 0  # Peer effect
        
        # Cross-validation loop
        for i in range(args.K):
            psi = np.zeros((2, 1))
            
            x1 = np.array(find_train_split(splits, i))
            x2 = np.array(splits[i])
            
            x1.sort()
            x2.sort()
            
            # Create dataset for treatment model
            treatment_dataset = CustomDataset(
                matrix, 
                data[data.columns[0:n_covariates]].to_numpy(), 
                data['T'].to_numpy(), 
                'continuous',
                'discrete', 
                x1, 
                x2
            )[0].to(device)
            
            print('Datasets created!')
            
            if args.model == 'GraphSage':
                treatment_model = GraphSAGE(
                    'discrete', 
                    treatment_dataset.x.shape[1],
                    len(torch.unique(treatment_dataset.y))
                ).to(device)
            elif args.model == 'GCN':
                treatment_model = GCN(
                    'discrete', 
                    treatment_dataset.x.shape[1],
                    len(torch.unique(treatment_dataset.y))
                ).to(device)
            elif args.model == 'GIN':
                treatment_model = GIN(
                    'discrete', 
                    treatment_dataset.x.shape[1],
                    len(torch.unique(treatment_dataset.y))
                ).to(device)
            
            treatment_optimizer = torch.optim.Adam(
                treatment_model.parameters(), 
                lr=0.01, 
                weight_decay=5e-4
            )
            treatment_model.fit(treatment_dataset, treatment_optimizer, 'discrete')
            
            t_hat = torch.exp(treatment_model(treatment_dataset.x, treatment_dataset.edge_index)[0])[:, 1].detach().numpy()
            res_T = data['T'] - t_hat
            
            test_mask_np = np.array(treatment_dataset.test_mask.cpu())
            J0[0][0] += (-np.dot(res_T[test_mask_np], res_T[test_mask_np]))
            J0[0][1] += (-np.sum(np.multiply(np.matmul(np.transpose(res_T), np.transpose(matrix)), res_T)[test_mask_np]))
            J0[1][0] += (-np.sum(np.multiply(np.matmul(np.transpose(res_T), matrix), res_T)[test_mask_np]))
            J0[1][1] += (-np.sum(np.multiply(np.matmul(np.transpose(res_T), np.matmul(np.transpose(matrix), matrix)), res_T)[test_mask_np]))
            
            J0 /= len(splits[i])
            
            outcome_dataset = CustomDataset(
                matrix, 
                data[data.columns[0:n_covariates]].to_numpy(), 
                data['Y'].to_numpy(), 
                'continuous',
                'continuous', 
                x1, 
                x2
            )[0].to(device)
            
            if args.model == 'GraphSage':
                outcome_model = GraphSAGE('continuous', outcome_dataset.x.shape[1]).to(device)
            elif args.model == 'GCN':
                outcome_model = GCN('continuous', outcome_dataset.x.shape[1]).to(device)
            elif args.model == 'GIN':
                outcome_model = GIN('continuous', outcome_dataset.x.shape[1]).to(device)
            
            outcome_optimizer = torch.optim.Adam(
                outcome_model.parameters(), 
                lr=0.01, 
                weight_decay=5e-4
            )
            outcome_model.fit(outcome_dataset, outcome_optimizer, 'continuous')
            
            y_hat = outcome_model.test(outcome_dataset, 'continuous').detach().numpy()
            res_Y = data['Y'] - y_hat
            
            res_PE = np.matmul(matrix, res_T)
            
            reg_features = np.concatenate((
                res_T.to_numpy().reshape(-1, 1)[test_mask_np],
                res_PE.reshape(-1, 1)[test_mask_np],
                treatment_dataset.x[test_mask_np]
            ), axis=1)
            
            reg = LinearRegression().fit(
                reg_features,
                res_Y[test_mask_np].to_numpy()
            )
            
            theta += reg.coef_[0]  # Direct effect
            alpha += reg.coef_[1]  # Peer effect
            
            psi0 = np.multiply(res_Y - np.matmul((reg.coef_[0] + reg.coef_[1] * matrix), res_T), res_T)
            psi1 = np.multiply(np.matmul(np.transpose(res_Y - np.matmul((reg.coef_[0] + reg.coef_[1] * matrix), res_T)), matrix), res_T)
            
            for j in x2:
                psi[0][0] = psi0.iloc[j]
                psi[1][0] = psi1.iloc[j]
                var += np.matmul(psi, np.transpose(psi))
                
            var /= len(splits[i])
        
        print(f'Estimated effects: theta={theta/args.K}, alpha={alpha/args.K}')
        theta /= args.K
        alpha /= args.K
        
        J0 /= args.K
        var /= args.K
        var = np.matmul(np.matmul(inv(J0), var), np.transpose(inv(J0)))
        
        CI_theta_start = theta - (var[0][0] ** 0.5) * 1.96 / (len(inds))
        CI_theta_end = theta + (var[0][0] ** 0.5) * 1.96 / (len(inds))
        
        CI_alpha_start = alpha - (var[1][1] ** 0.5) * 1.96 / (len(inds))
        CI_alpha_end = alpha + (var[1][1] ** 0.5) * 1.96 / (len(inds))
        
        print(f'Direct effect 95% CI: [{CI_theta_start}, {CI_theta_end}]')
        print(f'Peer effect 95% CI: [{CI_alpha_start}, {CI_alpha_end}]')
        
        avg_std += (var[0][0] ** 0.5) / (len(inds)) ** 0.5
        
        if CI_theta_start <= args.theta and CI_theta_end >= args.theta:
            n_coverage_theta += 1
        
        if CI_alpha_start <= args.alpha and CI_alpha_end >= args.alpha:
            n_coverage_alpha += 1
        
        variances.append(var)
        ADEs.append(theta)
        APEs.append(alpha)
        runtime.append(time.time() - start)
    
    coverage_theta = n_coverage_theta / args.trials
    coverage_alpha = n_coverage_alpha / args.trials
    print(f'Coverage rate (theta): {coverage_theta:.4f}')
    print(f'Coverage rate (alpha): {coverage_alpha:.4f}')
    
    avg_ade = np.mean(ADEs)
    avg_ape = np.mean(APEs)
    avg_runtime = np.mean(runtime)
    
    print(f'Average direct effect: {avg_ade:.4f}')
    print(f'Average peer effect: {avg_ape:.4f}')
    print(f'Average runtime: {avg_runtime:.2f} seconds')
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    results_df = pd.DataFrame(list(zip(ADEs, APEs, variances, runtime)), 
                             columns=['ADE', 'APE', 'VAR', 'runtime'])
    
    results_file = os.path.join(results_dir, f'{args.dataset}_{timestr}.csv')
    results_df.to_csv(results_file, index=False)
    print(f'Results saved to {results_file}')
    
    return {
        'ADEs': ADEs,
        'APEs': APEs,
        'variances': variances,
        'runtime': runtime,
        'coverage_theta': coverage_theta,
        'coverage_alpha': coverage_alpha,
        'avg_ade': avg_ade,
        'avg_ape': avg_ape,
        'avg_runtime': avg_runtime
    }
