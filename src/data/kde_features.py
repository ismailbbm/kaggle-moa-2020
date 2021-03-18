import pandas as pd
import numpy as np

from scipy import stats

import joblib
from joblib import Parallel, delayed
import multiprocessing

class kdeFeatures:
    def __init__(self,num_features):
        self.num_features = num_features

    def calculate_kde_kernels(self,X1,X2,ratio_inverse_kde):
        X = pd.concat([X1,X2])
        X_control = X[X['cp_type']==1]
        X_treatment = X[X['cp_type']==0]
        kernels = {}
        cols = self.num_features
        for col in cols:
            #Calculate kernels
            x_control = X_control[col].values
            x_treatment = X_treatment[col].values
            kde_control_kernel = stats.gaussian_kde(x_control)
            kde_treatment_kernel = stats.gaussian_kde(x_treatment)
            kernels[col+'_control'] = kde_control_kernel
            kernels[col+'_treatment'] = kde_treatment_kernel
            
            #Calculate max ratio so that when calculating kde features based on the ratio of treatement/control, we have a threshold for values
            x_control_mean = x_control.mean()
            x_control_std = x_control.std()
            x_treatment_mean = x_treatment.mean()
            #As b is not usually normal we use only a std to create range
            kde_range = [min(x_control_mean - 2*x_control_std, x_treatment_mean - 2*x_control_std),max(x_control_mean + 2*x_control_std, x_treatment_mean + 2*x_control_std)]
            kde_sample = np.arange(kde_range[0],kde_range[1],(kde_range[1]-kde_range[0])/100)
            
            x_control_kde_sample = kde_control_kernel.pdf(kde_sample)
            x_treatment_kde_sample = kde_treatment_kernel.pdf(kde_sample)
            if ratio_inverse_kde:
                max_ratio = (x_control_kde_sample/x_treatment_kde_sample).max()
            else:
                max_ratio = (x_treatment_kde_sample/x_control_kde_sample).max()
            kernels[col+'_ratio'] = max_ratio
            
        return kernels
    
    def build_batch(self,X,kernels,use_log_for_kernel_diff,inverse_kde,use_diff_kde,cpu_count,exclude_c_from_kde,exclude_g_from_kde):
        batch_list = []
        cols = self.num_features
        if exclude_c_from_kde:
            cols = [col for col in cols if not 'c-' in col]
        if exclude_g_from_kde:
            cols = [col for col in cols if not 'g-' in col]
        col_size = len(cols)            

        if col_size>=cpu_count:
            batch_size = int(col_size/cpu_count)
        else:
            batch_size = 1
            cpu_count = col_size
        for i in range(cpu_count):
            if i == cpu_count-1:
                batch_list.append((cols[i*batch_size:],X,kernels,use_log_for_kernel_diff,inverse_kde,use_diff_kde))
            else:
                batch_list.append((cols[i*batch_size:(i+1)*batch_size],X,kernels,use_log_for_kernel_diff,inverse_kde,use_diff_kde))
        return batch_list

    def process_individual_batch(self,batch):
        ratio_multiplier = 10
        cols = batch[0]
        X = batch[1]
        kernels = batch[2]
        use_log_for_kernel_diff = batch[3]
        inverse_kde = batch[4]
        use_diff_kde = batch[5]
        series_list = []
        for col in cols:
            kde_control_kernel = kernels[col+'_control']
            kde_treatment_kernel = kernels[col+'_treatment']
            
            if use_diff_kde:
                a_kde = kde_control_kernel.pdf(X[col].values)
                b_kde = kde_treatment_kernel.pdf(X[col].values)
                a = (b_kde-a_kde)/np.max((a_kde,b_kde),axis=0)
                a = a.clip(-1,1)
                a = np.nan_to_num(a,nan=0.0)
            else:
                if inverse_kde:
                    a = kde_control_kernel.pdf(X[col].values)/kde_treatment_kernel.pdf(X[col].values)
                else:
                    a = kde_treatment_kernel.pdf(X[col].values)/kde_control_kernel.pdf(X[col].values)
                a = np.nan_to_num(a,nan=ratio_multiplier*kernels[col+'_ratio'])
                a = a.clip(0,ratio_multiplier*kernels[col+'_ratio'])
                if use_log_for_kernel_diff:
                    a = np.log1p(a)
                    
            a = pd.Series(a,name=col+'_kde_diff',dtype='float32')
            series_list.append(a)
        return series_list

    def run_batch(self,batch):
        return self.process_individual_batch(batch)

    def process_batch_list(self,batch_list,cpu):
        return joblib.Parallel(n_jobs=cpu)(joblib.delayed(self.run_batch)(batch) for batch in batch_list)

    def process_kde_parallelized(self,X,kernels,use_log_for_kernel_diff,inverse_kde,use_diff_kde,cpu,exclude_c_from_kde,exclude_g_from_kde):
        batch_list = self.build_batch(X,kernels,use_log_for_kernel_diff,inverse_kde,use_diff_kde,cpu,exclude_c_from_kde,exclude_g_from_kde)
        results = self.process_batch_list(batch_list,cpu)
        for series_list in results:
            for s in series_list:
                X[s.name] = s.values
        return X