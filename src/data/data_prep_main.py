import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from scipy import stats
from sklearn import decomposition, cluster
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import joblib
from joblib import Parallel, delayed
import multiprocessing
import gc

from utils import boolean_to_index, index_to_boolean, seed_everything
from data.split_drugs import select_split_drugs
from data.data_pca import calculatePCA
from data.oversampling import calculate_prob_dist_for_data_augmentation, generate_data_augmentation
from data.kde_features import kdeFeatures

class prepareData:
    def __init__(self,path,validation_ratio=0,folds=5,
                 g_removal_count=0, add_c_stats=False, normalization_type='standard',
                 add_kernels=False, use_log_for_kernel_diff=False, inverse_kde=False, ratio_inverse_kde=False, use_diff_kde=False,exclude_c_from_kde=False,exclude_g_from_kde=False,
                 perform_pca=False, pca_variance_threshold=0.95, pca_for_kde=False, pca_for_c=True,
                 use_train_test_for_norm=True,cpu=None,
                 granularity=100,max_dev=0.1,normal_std_dev=0.1,additional=10):
        
        self.path = path
        self.folds = 5
        self.use_log_for_kernel_diff = use_log_for_kernel_diff
        self.normalization_type = normalization_type
        self.add_kernels = add_kernels
        self.add_c_stats = add_c_stats
        self.perform_pca = perform_pca
        self.pca_variance_threshold = pca_variance_threshold
        self.use_log_for_kernel_diff = use_log_for_kernel_diff
        self.inverse_kde = inverse_kde
        self.use_diff_kde = use_diff_kde
        self.exclude_c_from_kde = exclude_c_from_kde
        self.exclude_g_from_kde = exclude_g_from_kde
        
        
        if cpu is None:
            cpu = multiprocessing.cpu_count()
            self.cpu = cpu
        else:
            cpu = min(cpu,multiprocessing.cpu_count())
            self.cpu = cpu
            
        print('import data')
        self._import_data(self.path)
        self._num_features = list(set(self.X_train.columns) - set(['sig_id','cp_type','cp_dose','cp_time']))
        
        print('transform cat features')
        self.X_train = self._transform_cat_features(self.X_train)
        self.X_test = self._transform_cat_features(self.X_test)
        
        print('kde kernels calculations')
        self.kde_features = kdeFeatures(self._num_features)
        self.kde_kernels = self.kde_features.calculate_kde_kernels(self.X_train,self.X_test,ratio_inverse_kde)
        
        print('remove g columns with low variation')
        self.g_to_drop = self._calculate_g_cols_to_drop(self._num_features,self.kde_kernels,g_removal_count)
        self.X_train.drop(self.g_to_drop,inplace=True,axis=1)
        self.X_test.drop(self.g_to_drop,inplace=True,axis=1)
        self._num_features = list(set(self.X_train.columns) - set(['sig_id','cp_type','cp_dose','cp_time']))
        
        if add_kernels:
            print('kde features')
            self.X_train = self.kde_features.process_kde_parallelized(self.X_train,self.kde_kernels,use_log_for_kernel_diff,inverse_kde,use_diff_kde,cpu,exclude_c_from_kde,exclude_g_from_kde)
            self.X_test = self.kde_features.process_kde_parallelized(self.X_test,self.kde_kernels,use_log_for_kernel_diff,inverse_kde,use_diff_kde,cpu,exclude_c_from_kde,exclude_g_from_kde)
            
        if add_c_stats:
            print('add survavilability stats (c)')
            self.X_train = self._add_c_stats(self.X_train)
            self.X_test = self._add_c_stats(self.X_test)
            
        if perform_pca:
            print('perform pca')
            self.pca = calculatePCA()
            self.pca.fit([self.X_train,self.X_test],pca_for_kde,pca_for_c)
            self.X_train = self.pca.transform_pca(self.X_train,pca_variance_threshold)
            self.X_test = self.pca.transform_pca(self.X_test,pca_variance_threshold)
        
        print('normalize features')
        if use_train_test_for_norm:
            _ = self._normalize_features(pd.concat([self.X_train,self.X_test],axis=0))
            self.X_train = self._normalize_features(self.X_train,is_test=True)
            self.X_test = self._normalize_features(self.X_test,is_test=True)
        else:
            self.X_train = self._normalize_features(self.X_train)
            self.X_test = self._normalize_features(self.X_test,is_test=True)
            
        
        if additional>0:
            print('data augmentation')
            X_list = [self.X_train,self.X_test]

            prob_dist = calculate_prob_dist_for_data_augmentation(X_list,granularity)
            self.var_list = generate_data_augmentation(self.X_train,prob_dist,granularity,max_dev=max_dev,normal_std_dev=normal_std_dev,additional=additional)
        
    
    def prepare_test_data(self,path):
        X_test = pd.read_csv(self.path+'test_features.csv')
        X_test = self._transform_cat_features(X_test)
        X_test.drop(self.g_to_drop,inplace=True,axis=1)
        if self.add_kernels:
            X_test = self.kde_features.process_kde_parallelized(X_test,self.kde_kernels,self.use_log_for_kernel_diff,self.inverse_kde,self.use_diff_kde,self.cpu,self.exclude_c_from_kde,self.exclude_g_from_kde)
        if self.add_c_stats:
            X_test = self._add_c_stats(X_test)
        if self.perform_pca:
            X_test = self.pca.transform_pca(X_test,self.pca_variance_threshold)
        X_test = self._normalize_features(X_test,is_test=True)
        
        self.X_test = X_test
        
        
    def _import_data(self,path):
        self.X_train = pd.read_csv(path+'train_features.csv')
        self.X_test = pd.read_csv(path+'test_features.csv')
        self.y_train = pd.read_csv(path+'train_targets_scored.csv')
        self.X_train_additional = pd.read_csv(path+'train_targets_nonscored.csv')
        self.X_train_drugs = pd.read_csv(path+'train_drug.csv')
        self.sample_submission = pd.read_csv(path+'sample_submission.csv')
        
        
    def _transform_cat_features(self,X):
        X['cp_type'] = X['cp_type'].map({'trt_cp':0,'ctl_vehicle':1})
        X['cp_dose'] = X['cp_dose'].map({'D1':0,'D2':1})
        X['cp_time'] = X['cp_time'].map({24:0,48:0.5,72:1})
        return X
    
    def _normalize_features(self,X,is_test=False):
        cols_to_normalize = list(set(self.X_train.columns) - set(['sig_id','cp_type','cp_dose','cp_time']))
        if is_test==False:
            self.normalizer_dict = {}
        for col in cols_to_normalize:
            if is_test:
                scaler = self.normalizer_dict[col]
                a = X[col].values.reshape(-1,1)
                a = (scaler.transform(a)).flatten()
                
                if (self.normalization_type == 'quantile') and (not '_kde_diff' in col):
                    a = a/10.0 + 0.5
                X[col] = a
            else:
                if (self.normalization_type == 'quantile') and (not '_kde_diff' in col):
                    scaler = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")
                else:
                    scaler = MinMaxScaler()
                
                a = X[col].values
                a = scaler.fit_transform(a.reshape(-1, 1))
                self.normalizer_dict[col] = scaler
                
                if (self.normalization_type == 'quantile') and (not '_kde_diff' in col):
                    #Quantilization transforms data on a [-5:+5] range here
                    a = a/10.0 + 0.5
                    
                X[col] = a
                
        return X
    
    def _calculate_g_cols_to_drop(self,cols,kde_kernels,g_removal_count):
        g_to_drop = []
        if g_removal_count==0:
            return g_to_drop
        g_cols = []
        diff_list = []
        for col in cols:
            if 'g-'==col[:2]:
                g_cols.append(col)
                
                kde_control_kernel = kde_kernels[col+'_control']
                kde_treatment_kernel = kde_kernels[col+'_treatment']
                
                rg = np.arange(-10,10,0.1)
                
                vehicle_kde_sample = kde_treatment_kernel.pdf(rg)
                control_kde_sample = kde_control_kernel.pdf(rg)
                
                diff = (np.abs(vehicle_kde_sample-control_kde_sample)).mean()
                
                diff_list.append(diff)

        diff_list_ordered = np.sort(np.array(diff_list))
        thresh = diff_list_ordered[g_removal_count-1]
        
        for col, diff in zip(g_cols,diff_list):
            if diff<=thresh:
                g_to_drop.append(col)
                
        return g_to_drop
    
    def _add_c_stats(self,X):
        all_cols = X.columns
        c_cols = [x for x in all_cols if ('c-' in x) & (not '_kde_diff' in x)]
        X_values = X[c_cols].values
        #Add stats
        X['c_stats_sum'] = X_values.sum(axis=1)
        X['c_stats_mean'] = X_values.mean(axis=1)
        X['c_stats_median'] = np.median(X_values,axis=1)
        X['c_stats_std'] = np.std(X_values,axis=1)
        X['c_stats_kurtosis'] = stats.kurtosis(X_values,axis=1)
        X['c_stats_skew'] = stats.skew(X_values,axis=1)
        
        return X
                   
    def split_train_holdout(self,validation_ratio=0.2,seed=42,threshold=1000):
        if validation_ratio==0:
            X_train = self.X_train.copy()
            X_train = pd.merge(X_train,self.y_train,on='sig_id')
            X_train = pd.merge(X_train,self.X_train_drugs,on='sig_id')
            X_drugs = X_train['drug_id'].values
            X_sig_id = X_train['sig_id'].values
            y_train = X_train[self.y_train.columns.tolist()].iloc[:,1:].copy()
            X_train = X_train[self.X_train.columns.tolist()].iloc[:,1:].copy()
            mask_train = np.isin(self.X_train['sig_id'].values,X_sig_id)
            return  X_train,y_train,X_drugs,X_sig_id,None,None,mask_train
        else:
            seed_everything(seed)
            X = self.X_train.copy()
            X = pd.merge(X,self.y_train,on='sig_id')
            X = pd.merge(X,self.X_train_drugs,on='sig_id')

            #Split control in 2
            mask_control = X.iloc[:,1].values==1
            idx_control = boolean_to_index(mask_control)

            idx_control_train, idx_control_holdout = train_test_split(idx_control,test_size=validation_ratio,random_state=seed)
            X_control_train = X.iloc[idx_control_train]
            X_control_holdout = X.iloc[idx_control_holdout]

            #Identify drugs above threshold
            drugs_count = X['drug_id'].value_counts()
            drugs_above_thresh = drugs_count.loc[drugs_count>threshold].index.values.tolist()
            drugs_above_thresh = list(set(drugs_above_thresh) - set(X[X['cp_type']==1]['drug_id'].values.tolist()))
            if len(drugs_above_thresh)>0:
                idx_drugs_spread = X[X['drug_id'].isin(drugs_above_thresh)].index.values.tolist()
                mask_drugs_spread = index_to_boolean(idx_drugs_spread,X.shape[0])

                idx_drugs_spread_train, idx_drugs_spread_holdout = train_test_split(idx_drugs_spread,test_size=validation_ratio,random_state=seed)
                X_drugs_spread_train = X.iloc[idx_drugs_spread_train]
                X_drugs_spread_holdout = X.iloc[idx_drugs_spread_holdout]

            #Split treatment in 2 by respecting that drugs only appear on 1 group
            #Create list of drugs and how many records they have
            if len(drugs_above_thresh)>0:
                mask_treatment = (X.iloc[:,1].values==0) & (mask_drugs_spread==False)
            else:
                mask_treatment = (X.iloc[:,1].values==0)
            holdout_drugs = select_split_drugs(X[mask_treatment].copy(),validation_ratio)
            
            if len(drugs_above_thresh)==0:
                X_train = pd.concat([X[~(X['drug_id'].isin(holdout_drugs)) & (X['cp_type']==0)],X_control_train],axis=0)
                X_holdout = pd.concat([X[(X['drug_id'].isin(holdout_drugs)) & (X['cp_type']==0)],X_control_holdout],axis=0)
            else:
                X_train = pd.concat([X[~(X['drug_id'].isin(holdout_drugs)) & (X['cp_type']==0) & ~(X['drug_id'].isin(drugs_above_thresh))],X_control_train,X_drugs_spread_train],axis=0)
                X_holdout = pd.concat([X[(X['drug_id'].isin(holdout_drugs)) & (X['cp_type']==0)],X_control_holdout,X_drugs_spread_holdout],axis=0)
            
            #reorder
            X_train = pd.merge(self.X_train['sig_id'],X_train,on=['sig_id'])
            
            y_train = X_train[self.y_train.columns.tolist()].iloc[:,1:].copy()
            y_holdout = X_holdout[self.y_train.columns.tolist()].iloc[:,1:].copy()
            
            X_drugs = X_train['drug_id'].values
            X_sig_id = X_train['sig_id'].values
            X_train = X_train[self.X_train.columns.tolist()].iloc[:,1:].copy()
            X_holdout = X_holdout[self.X_train.columns.tolist()].iloc[:,1:].copy()
            mask_train = np.isin(self.X_train['sig_id'].values,X_sig_id)
            
            return X_train,y_train,X_drugs,X_sig_id,X_holdout,y_holdout,mask_train