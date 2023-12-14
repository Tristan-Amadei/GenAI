import numpy as np

def anderson_darling_ticker(real_data, generated_sample, ticker):
    order_statistics = generated_sample[:, ticker].copy()
    order_statistics.sort()
    n = len(order_statistics)
    
    u_ticker = np.array([((real_data[:, ticker] < order_statistics[i]).sum() + 1) / (n + 2) for i in range(n)])
    logs = np.log(u_ticker) + np.log(1-np.flip(u_ticker))
    indices = np.arange(1, n + 1)
    W_n_ticker = -n - (1/n) * np.sum((2 * indices - 1) * logs)
    return W_n_ticker
    
def anderson_darling(real_data, generated_sample):
    Lm = 0
    for ticker in range(generated_sample.shape[1]):
        Lm += anderson_darling_ticker(real_data, generated_sample, ticker)
    return Lm / generated_sample.shape[1]


def compute_Zi(X, i):
    return np.sum(np.all(X[np.arange(X.shape[0]) != i] < X[i], axis=1)) / (X.shape[0]-1)

def kendall_error(real_data, generated_sample):
    Z = np.array([compute_Zi(real_data, i) for i in range(real_data.shape[0])])
    Z_tilde = np.array([compute_Zi(generated_sample, i) for i in range(generated_sample.shape[0])])
    
    Z.sort()
    Z_tilde.sort()
    
    n = min(len(Z), len(Z_tilde))
    Z = Z[:n]
    Z_tilde = Z_tilde[:n]
    return abs(Z - Z_tilde).sum() / n