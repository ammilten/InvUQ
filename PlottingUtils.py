import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(p_pred_prior, p_true_prior, ntrain=None, p_pred_obs=None, p_true_obs=None, varnames=None, units=None, figsize=(14,12)):
    '''
    This function plots a scatter plot of true vs predicted paramters
    
    Inputs:
        p_pred_prior: a [NxP] matrix of prior parameters predicted by the ML inverse model
        p_true_prior: a [NxP] matrix of corresponding true parameters for p_pred_prior
        ntrain (optional): integer specifying the size of the training set, to assign train and test set different symbols. Assumes first ntrain samples from p_pred_prior and p_true_prior are training set, and remaining samples are test set.
        p_pred_obs (optional): a [QxP] matrix of parameters predicted by the ML inverse model applied to the observerd data realizations, plotted as black vertical lines
        p_true_obs (optional): a [1xP] matrix specifying the true model parameters, plotted as red horizontal lines
        varnames (optional): list of variable name strings for x axis labels
        units (optional): list of unit strings for x axis labels
        figsize (optional): 2-tuple specifying y size and x size of the figure, respectively

    Defintion of dimensions:
        P: number of prior parameters
        Q: number of observed data samples
        N: number of prior realizations
    '''
    nparams = p_pred_prior.shape[1]
    nrows = np.floor(nparams/2+1).astype(np.int)
    plt.subplots(nrows=nrows, ncols=2, figsize=figsize)
    
    for i in range(nparams):
        pltnum = nrows*100+20+1+i
        plt.subplot(pltnum)

        pmin = np.min(p_pred_prior[:,i])
        pmax = np.max(p_pred_prior[:,i])

        pmin_true = np.min(p_true_prior[:,i])
        pmax_true = np.max(p_true_prior[:,i])
        aa = (pmax_true-pmin_true) * 0.1

        if ntrain is not None:
            plt.scatter(p_pred_prior[ntrain+1:,i], p_true_prior[ntrain+1:,i], color=(.25,.25,1), marker='s', edgecolor='k', label='Test')
            plt.scatter(p_pred_prior[:ntrain,i], p_true_prior[:ntrain,i], color=(1,.5,.5), marker='o', edgecolor='k', label='Train')
        else:
            plt.scatter(p_pred_prior[:,i], p_true_prior[:,i], color=(.25,.25,1), marker='s', edgecolor='k', label='Test')
        
        if p_pred_obs is not None:
            p50 = np.percentile(p_pred_obs[:,i],50)
            p2 = np.percentile(p_pred_obs[:,i],2.5)
            p97 = np.percentile(p_pred_obs[:,i],97.5)

            plt.plot([p50, p50], [pmin_true-aa, pmax_true+aa],'-k',linewidth=3, label='Median')
            plt.plot([p2, p2], [pmin_true-aa, pmax_true+aa],':k',linewidth=3,label='95% Confidence Interval')
            plt.plot([p97, p97], [pmin_true-aa, pmax_true+aa],':k',linewidth=3)

        plt.plot([pmin,pmax], [pmin,pmax],'-c',linewidth=2,label='1:1')
        xl = plt.gca().get_xlim()

        if p_true_obs is not None:
            plt.plot(xl,[p_true_obs[i],p_true_obs[i]], '-r', linewidth=3, label='True')

        if varnames is not None:
            if units is not None:
                plt.title(varnames[i] + ' ('+units[i]+')',fontsize=16)
            else:
                plt.title(varnames[i], fontsize=16)
        
        plt.ylim((pmin_true-aa,pmax_true+aa))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    plt.subplot(nrows*100+20+1)
    plt.legend()
    plt.subplot(nrows*100+20+nrows*2-1)
    plt.xlabel('Predicted',fontsize=16)
    plt.ylabel('True',fontsize=16)

    plt.tight_layout()
    return

def plot_posteriors(posterior, prior=None, truths=None, varnames=None, units=None, figsize=(14,12)):
    '''
    This function plots histograms of posterior distributions for each parameter, and also the prior distributions if specified.

    Inputs:
        posterior: a [QxP] matrix of posterior samples for each parameter
        prior (optional): a [NxP] matrix of prior samples for each parameter
        truths (optional): a [1xP] array of true parameter values, plotted as a red vertical line
        varnames (optional): list of variable name strings for x axis labels
        units (optional): list of unit strings for x axis labels
        figsize (optional): 2-tuple specifying y size and x size of the figure, respectively

    Defintion of dimensions:
        P: number of prior parameters
        Q: number of observed data samples
        N: number of prior realizations
    '''
    nparams = posterior.shape[1]
    nrows = np.floor(nparams/2+1).astype(np.int)
    plt.subplots(nrows=nrows, ncols=2, figsize=figsize)

    for i in range(nparams):
        pltnum = nrows*100+20+1+i
        plt.subplot(pltnum)
        
        if prior is not None:
            plt.hist(prior[:,i],density=True,facecolor=(.75,.75,.75),edgecolor='k',label='Prior')

        plt.hist(posterior[:,i],density=True,facecolor=(.25,.25,1,.8),edgecolor='k',label='Posterior')
            
        yl = plt.gca().get_ylim()
        
        if truths is not None:
            plt.plot([truths[i],truths[i]], yl, '-r', linewidth=3, )
        
        if varnames is not None:
            if units is not None:
                plt.xlabel(varnames[i] + ' ('+units[i]+')',fontsize=16)
            else:
                plt.xlabel(varnames[i], fontsize=16)  
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
    plt.subplot(nrows*100+20+1)
    plt.legend()
    plt.subplot(nrows*100+20+nparams)
    plt.ylabel('Probability Density',fontsize=16)
    
    plt.tight_layout()
        
    return


