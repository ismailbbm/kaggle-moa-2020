import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import seed_everything
from data.split_drugs import select_split_drugs

def create_cv(self,X,y,drugs,sig_ids,threshold=1000,folds=5,seed=42):
        seed_everything(seed)

        y_cols = y.columns.tolist()
        X = X.copy()
        y = y.copy()

        X = pd.concat([X,y],axis=1)
        X['drug_id'] = drugs
        X['sig_id'] = sig_ids

        #Locate drugs
        drugs_count = X['drug_id'].value_counts()
        drugs_below_thresh = drugs_count.loc[drugs_count<=threshold].index.sort_values()
        drugs_above_thresh = drugs_count.loc[drugs_count>threshold].index.sort_values()

        dct_below_thresh = {}; dct_above_thresh = {}
        #Stratify below threshold
        skf= MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        tmp = X.groupby('drug_id')[y_cols].mean().loc[drugs_below_thresh]
        for f,(idxT,idxV) in enumerate( skf.split(tmp,tmp[y_cols])):
            dd = {k:f for k in tmp.index[idxV].values}
            dct_below_thresh.update(dd)

        #stratify above threshold
        skf= MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        tmp = X.loc[X['drug_id'].isin(drugs_above_thresh)].reset_index(drop=True)
        for f,(idxT,idxV) in enumerate( skf.split(tmp,tmp[y_cols])):
            dd = {k:f for k in tmp.sig_id[idxV].values}
            dct_above_thresh.update(dd)

        # ASSIGN FOLDS
        X['fold'] = X['drug_id'].map(dct_below_thresh)
        X.loc[X['fold'].isna(),'fold'] = X.loc[X['fold'].isna(),'sig_id'].map(dct_above_thresh)
        X['fold'] = X['fold'].astype('int8')
        
        oof_assignment = X['fold'].values
        
        oof_idx = []
        for x in np.arange(folds):
            train = np.where(oof_assignment!=x)[0]
            val = np.where(oof_assignment==x)[0]
            oof_idx.append((train,val))
        return oof_idx
    
def create_cv_kfold(self,X,drugs,sig_ids,folds=5,seed=42):
    seed_everything(seed)
    
    X = X.copy()

    X['drug_id'] = drugs
    X['sig_id'] = sig_ids
    
    mask_treatment = X.iloc[:,0].values==0
    mask_control = X.iloc[:,0].values==1
    X_with_treatment = X[mask_treatment].copy()
    X_control = X[mask_control].copy()
    
    drugs_dict = {}
    drugs_already_in_fold = []
    for i in range(folds):
        X_remaining = X_with_treatment[~X_with_treatment['drug_id'].isin(drugs_already_in_fold)].copy()

        if i<folds-1:
            tmp_drugs = select_split_drugs(X_remaining,1.0/(folds-i))
            drugs_already_in_fold = drugs_already_in_fold + tmp_drugs
            dd = {k:i for k in tmp_drugs}
            drugs_dict.update(dd)
        else:
            dd = {k:i for k in X_remaining['drug_id'].values.tolist()}
            drugs_dict.update(dd)
            
    
    X_with_treatment['fold'] = X_with_treatment['drug_id'].map(drugs_dict)
    X_control['fold'] = np.random.randint(0,folds,size=X_control.shape[0])
    
    X_all = pd.concat([X_with_treatment,X_control],axis=0)
    
    X_all = pd.merge(X[['sig_id']],X_all,on='sig_id')
    
    oof_assignment = X_all['fold'].values
    
    oof_idx = []
    for x in np.arange(folds):
        train = np.where(oof_assignment!=x)[0]
        val = np.where(oof_assignment==x)[0]
        oof_idx.append((train,val))
    return oof_idx