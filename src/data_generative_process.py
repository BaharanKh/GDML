import numpy as np
import pandas as pd
import scipy.special as sp



def dgp(A, theta=10, alpha=5, gamma=1.0):
    """
    DGP with direct effect (theta) and peer effect (alpha) on outcome.
    
    Parameters:
        A: Adjacency matrix
        theta: Direct treatment effect
        alpha: Peer effect from neighbors' treatments
        gamma: Neighbors' confounding effect strength
        
    Returns:
        df: DataFrame with features, treatment, and outcome
        network: Adjacency matrix as DataFrame
        df_true: True propensity scores
        X: Original feature matrix
    """
    X = np.random.normal(size=(A.shape[0], 1))
    
    # Treatment assignment with network interference
    piT = sp.expit((np.mean(X, axis=1) + gamma * np.mean(np.matmul(A, X), axis=1)) / 10)
    T = np.random.binomial(1, piT)
    
    # Calculate peer effects from neighbors' treatments
    peer_effect = np.matmul(A, T)
    
    # Outcome model with direct and peer effects
    Y = T * theta + peer_effect * alpha + np.sum(X, axis=1) + np.sum(np.matmul(A, X), axis=1)
    
    df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])],
                      index=[f'u{i}' for i in range(X.shape[0])])
    df['Y'] = Y
    df['T'] = T

    df_true = pd.DataFrame()
    df_true['piT'] = piT
    return df, df_true, X


def dgp_nonlinear(A, theta=10, gamma=1.0):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    """
    DGP with max aggregation of neighbors' features.
    
    Parameters:
        A: Adjacency matrix
        theta: Direct treatment effect
        gamma: Network effect strength
        
    Returns:
        df: DataFrame with features, treatment, and outcome
        network: Adjacency matrix as DataFrame
        df_true: True potential outcomes and propensity scores
        X: Original feature matrix
    """
    X = np.random.normal(loc=10, size=(A.shape[0], 1))

    piT = sp.expit((np.sum(X, axis=1) + gamma * np.max(np.multiply(A, X), axis=1)) / 100)
    T = np.random.binomial(1, piT)

    Y0 = (sigmoid(np.sum(X, axis=1) + 
          np.max(np.multiply(A, X), axis=1)) + 
          np.matmul(A, T.reshape(-1, 1))*alpha)
    Y1 = (sigmoid(np.sum(X, axis=1) + 
          np.max(np.multiply(A, X), axis=1)) + 
          theta + 
          np.matmul(A, T.reshape(-1, 1))*alpha)
    
    Y = Y1 * T + Y0 * (1 - T)
    df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])],
                      index=[f'u{i}' for i in range(X.shape[0])])
    df['Y'] = Y
    df['T'] = T

    df_true = pd.DataFrame()
    df_true['Y1'] = Y1
    df_true['Y0'] = Y0
    df_true['piT'] = piT
    return df, df_true, X