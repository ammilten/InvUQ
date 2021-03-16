import numpy as np

def sample_ecdf(p_pred_obs, p_pred_test, p_true_test, W):
    '''
    This function samples the posterior for a single parameter

    Inputs:
        p_pred_obs: a [Qx1] array of parameters predicted from observed data using the machine learning inverse model
        p_pred_test: a [Nx1] array of parameters predicted from the test set of the prior data using the machine learning inverse model
        p_pred_true: a [Nx1] array of parameters used to generate the prior data
        W: a number specifying the tolerance for constructing empirical CDFs

    Outputs:
        postsamples: a [Qx1] array of posterior parameter samples

    Definition of Variables:
        Q: number of posterior samples/observed data realizations
        N: number of samples in the test set
    '''
    postsamples = np.zeros(p_pred_obs.shape[0])
    for i in range(p_pred_obs.shape[0]):
        INDS = np.where(np.logical_and(p_pred_test>=p_pred_obs[i]-W, p_pred_test<=p_pred_obs[i]+W))
        csamples = p_true_test[INDS]
        
        if csamples.shape[0] is 0:
            print("Data realization "+str(i)+": No test set samples available")
            sample_ind = np.random.choice(p_true_test.shape[0], 1)
            sample = p_true_test[sample_ind]
        else:
            sample_ind = np.random.choice(csamples.shape[0], 1)
            sample = csamples[sample_ind]
            
        postsamples[i] = sample
        
    return postsamples

def compute_default_tol(p_pred_test):
    '''
    This function computes default tolerances for empirical CDF sampling

    Inputs:
        p_pred_test: a [NxP] matrix of parameters predicted from the test set of the prior data using the machine learning inverse model

    Ouputs:
        Ws: a [1xP] array of tolerances for constructing empirical CDFs for each parameter

    Definition of Variables:
        P: number of parameters to sample
        N: number of samples in the test set
    '''

    rangefraction = 1/20

    # Compute min and max values for each parameter
    pmin = np.min(p_pred_test,axis=0)
    pmax = np.max(p_pred_test,axis=0)

    # W is (pmax-pmin)*rangefraction
    return rangefraction * (pmax-pmin)



def EmpiricalPosteriorSampling(p_pred_obs, p_pred_test, p_true_test, Ws=None):
    '''
    This function performs Jeffrey sampling for a set of parameters

    Inputs:
        p_pred_obs: a [QxP] matrix of parameters predicted from observed data using the machine learning inverse model
        p_pred_test: a [NxP] matrix of parameters predicted from the test set of the prior data using the machine learning inverse model
        p_pred_true: a [NxP] matrix of parameters used to generate the prior data
        Ws (optional): a [1xP] array of tolerances for constructing empirical CDFs for each parameter

    Outputs:
        postsamples: a [QxP] matrix of posterior parameter samples

    Definition of Variables:
        Q: number of posterior samples/observed data realizations
        P: number of parameters to sample
        N: number of samples in the test set
    '''

    if Ws is None:
        Ws = compute_default_tol(p_pred_test)
    
    postsamples = np.zeros(p_pred_obs.shape)
    for i in range(len(Ws)):
        print('Sampling parameter '+str(i))
        postsamples[:,i] = sample_ecdf(p_pred_obs[:,i], p_pred_test[:,i], p_true_test[:,i], Ws[i])
    return postsamples


