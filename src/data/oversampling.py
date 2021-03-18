import pandas as pd
import numpy as np

from scipy import stats

import gc

def calculate_prob_dist_for_data_augmentation(X_list,granularity):
    X = pd.concat(X_list)
    X_treatment = X[X['cp_type']==0]
    prob_dist = []
    dist = np.arange(0,1,1/granularity)
    for col in X.columns[4:]:
        x = X_treatment[col].values
        kernel = stats.gaussian_kde(x)
        prob = kernel.pdf(dist)
        prob_dist.append(prob)
    prob_dist = np.array(prob_dist)
    return prob_dist

def generate_data_augmentation(X,prob_dist,granularity,max_dev=0.1,normal_std_dev=0.1,additional=10):
    x = X.values[:,4:].copy().astype(np.float16)

    max_dev_steps = int(max_dev*granularity)

    #Calculate normal matrix
    normal_p = np.arange(-max_dev*granularity,max_dev*granularity+1,1)
    normal_p = normal_p/granularity
    normal_p = 1/(normal_std_dev)*np.exp(-(normal_p*normal_p)/normal_std_dev**2)
    normal_p = normal_p.astype(np.float16)
    normal_p = np.repeat(normal_p[np.newaxis,:], x.shape[1], axis=0)
    normal_p = np.repeat(normal_p[np.newaxis,:,:], x.shape[0], axis=0)

    #Transform x so that it rounds to the desired granularity
    x_rounded = (np.round(x*granularity)).astype(int)


    #For each and every value a in x, we want to calculate a vector of probability of size 2n+1 such as
    #The probability value at index 0 is the probability that we remove max_dev to a
    i_steps = np.arange(-max_dev_steps,max_dev_steps+1,1) #initialization vector for the steps
    i_initial = np.tile(np.array([[i_steps]]),(x.shape[0],x.shape[1],1))
    x_rounded_repeated = np.repeat(x_rounded[:, :, np.newaxis], i_steps.shape[0], axis=2)
    idx = i_initial + x_rounded_repeated
    idx = idx.copy()
    idx = np.clip(idx,0,granularity-1) #For each 
    
    del i_initial, x_rounded_repeated
    gc.collect()

    #prob_candidates = prob_dist[0,0,idx].copy()
    prob_candidates = np.zeros(idx.shape)
    for j in range(idx.shape[1]):
        prob_candidates[:,j,:] = np.take(prob_dist[j,:],idx[:,j,:])
    
    del idx
    gc.collect()
    
    prob_candidates = prob_candidates*normal_p
    prob_candidates = prob_candidates.copy()

    del normal_p
    gc.collect()

    prob_candidates = prob_candidates/prob_candidates.sum(axis=2)[:,:,np.newaxis]


    var = np.zeros([x.shape[0],x.shape[1],additional])
    i_steps_norm = i_steps/max_dev_steps*max_dev
    print('calculating probas')
    for k in range(x.shape[1]):
        for i in range(x.shape[0]):
            var[i,k,:] = np.random.choice(i_steps_norm,size=additional,p=prob_candidates[i,k,:])

    var_list = []
    for i in range(additional):
        var_list.append(var[:,:,i].copy())
    return var_list