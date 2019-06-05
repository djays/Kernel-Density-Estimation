import numpy as np
import time

def log_sum_exp(X):
    """ Compute log sum exp over X: axis 0. 
    Further prevent overflow/underflow using subtract max trick """
    max_x = np.max(X)
    X_ = X - max_x 
    return max_x  + np.log(np.sum(np.exp(X_)))

def log_likelihood(X_a, X_b, sigma):
    """Compute the mean log likelihood of X_b under X_a using a mixture of gaussians, one gaussian per feature
    :param X_a : np.ndarray : (samples, features)
    :param X_b : np.ndarray: (samples, features)
    :sigma : Standard deviation used for all the gaussians 
    
    """
    #start_time = time.time()

    # For effeciency Pre-compute fixed parts of the model
    # log_prob_z: -log k
    var = sigma ** 2
    log_prob_z = -np.log(X_a.shape[0])
    log_normal_b = -0.5 * np.log(2 * np.pi * var) * X_a.shape[1]
    log_prob_x = np.float64()
    
    for i in range(X_b.shape[0]):
        x_b = X_b[i]
        log_normal_a = -((x_b - X_a) ** 2)/(2 * var) 
        log_prob_x += log_sum_exp(log_normal_a.sum(axis=1)) + log_normal_b +  log_prob_z
    log_prob_x = log_prob_x/X_b.shape[0]
    
    #print("Total time: %.2f mins" % ((time.time() - start_time)/60)) 
    
    return log_prob_x
    
def log_sum_exp_batch(X):
    """ Compute log sum exp over X: axis 1. 
    Further prevent overflow/underflow using subtract max trick """
    max_x = np.max(X, axis=1)
    X_ = X - max_x[:, np.newaxis] 
    return max_x  + np.log(np.sum(np.exp(X_), axis=1))

def log_likelihood_batch(X_a, X_b, sigma):
    """Compute the mean log likelihood of X_b under X_a using a mixture of gaussians, one gaussian per feature
    :param X_a : np.ndarray : (samples, features)
    :param X_b : np.ndarray: (samples, features)
    :sigma : Standard deviation used for all the gaussians 
    
    """
    start_time = time.time()

    # For effeciency Pre-compute fixed parts of the model
    # log_prob_z: -log k
    var = sigma ** 2
    log_prob_z = -np.log(X_a.shape[0])
    log_normal_b = -0.5 * np.log(2 * np.pi * var)
    log_prob_x = 0.0
    
    # To better utilize memory, batch the ops by creating a new dimension
    batch_count = 10
    for i in range(X_b.shape[0]//batch_count):
        X_b_batch = X_b[i*batch_count:(i+1) * batch_count, np.newaxis]
        log_normal_a = -((X_b_batch - X_a) ** 2)/(2 * var)
        log_prob_xi = log_sum_exp_batch(log_normal_a.sum(axis=2)) + log_normal_b + log_prob_z
        log_prob_x += log_prob_xi.sum()
    log_prob_x = log_prob_x/X_b.shape[0]
    
    print("Total time: %.2f mins" % ((time.time() - start_time)/60)) 
    
    return log_prob_x