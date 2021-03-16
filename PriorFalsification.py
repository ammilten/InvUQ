from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

class PriorFalsification:
    '''
    This class contains methods to perform and visualize prior falsification based on PCA
    '''

    def __init__(self, zprior, zobs, npc=None):
        '''
        This function constructs the prior falsification object and performs PCA on the data
        Inputs:
            zprior: an [NxD] matrix of prior data realizations
            zobs: a [QxD] array of observed data
            npc (optional): integer specifying number of principal components to compute. If not specified then N principal components are used.

        Output:
            PF: PosteriorFalsification object with the following attributes:
                - eigvecs: a [DxC] matrix of principal component eigenvectors
                - Sprior: a [NxC] matrix of prior data scores
                - Sobs: a [QxC] array of observed data scores
                - eigvals: a [1xC] array of eigenvalues
                - var: a [1xC] array of component-wise variance fraction
                - cumvar: a [1xC] array of cumulative variance fraction
                - npc: integer specifying the number of PCs calculated
                - pca: sklearn.decomposition.PCA object fitted to zprior

        Defintion of dimensions:
            D: data dimensionality
            Q: number of observed data samples
            N: number of prior realizations
            C: number of principal components to calculate
        '''   
        self.pca = PCA(n_components=npc)
        self.pca.fit(zprior)
        self.Sprior = self.pca.transform(zprior)
        self.Sobs = self.pca.transform(zobs)
        self.eigvecs = self.pca.components_
        self.eigvals = self.pca.explained_variance_
        self.var = self.pca.explained_variance_ratio_
        self.cumvar = np.cumsum(self.var)
        self.npc = self.Sprior.shape[1]

    def plot_scores(self, c1=0, c2=1):
        '''
        This function plots a scatter plot of prior and observed PC scores

        Inputs:
            c1 (optional): integer of first PC to plot (starting from 0)
            c2 (optional): integer of second PC to plot (starting from 0)

        '''

        plt.scatter(
            self.Sprior[:,c1], self.Sprior[:,c2],
            label='Prior',
            edgecolor='k',
            c=(0.5,0.5,0.5)
            )

        plt.scatter(
            self.Sobs[:,c1], self.Sobs[:,c2], 500,
            label='Obs', 
            marker='*',
            edgecolor='k',
            c=(225/255,223/255,0/255)
            )

        plt.xlabel('PC '+str(c1+1)+' Score ('+str(np.round(self.var[c1]*100,2))+'%)',fontsize=16)
        plt.ylabel('PC '+str(c2+1)+' Score ('+str(np.round(self.var[c2]*100,2))+'%)',fontsize=16)
        plt.legend(fontsize=14)

    def plot_scree(self, npcs=None):
        '''
        This function generates a scree plot

        Inputs:
            npcs (optional): integer specifying the number of PCs to plot
        '''

        if npcs is None:
            n = self.npc
        else:
            n = npcs

        pc = list(range(0,n+1))
        variance =  np.concatenate((np.array([0]),self.cumvar[:n]))

        plt.plot(pc,variance,'-k',linewidth=4)
        plt.ylim((0,1))
        plt.xlabel('Principal Component Number',fontsize=14)
        plt.ylabel('Cumulative Variance Fraction',fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    






