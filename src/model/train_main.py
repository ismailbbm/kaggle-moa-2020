import torch
import numpy as np

import gc

from model.models import which_model
from model.fit import NNWrapper
from data.folds_creation import create_cv, create_cv_kfold


def mask_variation(var_list,mask):
    var_list_new = []
    for var in var_list:
        var_list_new.append(var[mask].copy())
    return var_list_new

def train_model(prepared_data, conf, export_path=None):

    folds = conf['model']['folds']
    oof_type = conf['model']['oof_type']
    oof_threshold = conf['model']['oof_threshold']
    n_seeds = conf['model']['n_seeds']
    augment_data = conf['augmentation']['augment_data']
    control_share_in_train = conf['augmentation']['control_share_in_train']
    add_control_from_test = conf['augmentation']['add_control_from_test']
    label_smoothing = conf['model']['label_smoothing']
    oof_loss_limit = conf['model']['oof_loss_limit']
    nn_architecture = conf['model']['nn_architecture']
    fold_type = conf['model']['fold_type']
    validation_ratio = conf['model']['validation_ratio']

    torch.backends.cudnn.deterministic = True

    TabularNN, CFG = which_model(nn_architecture)

    nn_model_list = []
    for i in range(n_seeds):
        seed =  i
        print('\n')
        print('START WITH SEED: {}'.format(seed))

        X_train,y_train,X_drugs,X_sig_id,X_holdout,y_holdout,mask_train = prepared_data.split_train_holdout(validation_ratio=validation_ratio,seed=seed,threshold=oof_threshold)
        if oof_type=='multi':
            oof_idx = create_cv(X_train,y_train,X_drugs,X_sig_id,folds=folds,seed=seed,threshold=oof_threshold)
        else:
            oof_idx = create_cv_kfold(X_train,X_drugs,X_sig_id,folds=folds,seed=seed)

        X_test_control = prepared_data.X_test.values
        mask = X_test_control[:,1]==1
        X_test_control = X_test_control[mask][:,4:].copy()
        X_test_control = X_test_control.astype(np.float32)

        X = X_train.values
        y = y_train.values

        #Get treatment
        mask_treatment = X[:,0]==0

        #Keep only useful columns
        X = X[:,3:].copy()

        CFG.num_features=X.shape[1]
        CFG.target_cols=y.shape[1]
        
        
        #Data augmentation
        var_list = None
        if augment_data:
            del var_list
            gc.collect()
            var_list = mask_variation(prepared_data.var_list,mask_train)


        moa_control_params = {
            'control_share': control_share_in_train,
            'add_control_from_test': add_control_from_test,
            'mask_treatment': mask_treatment,
            'test_control_data': X_test_control,
            'augment_data': augment_data,
            'augment_var_list': var_list,
            'label_smoothing': label_smoothing,
            'oof_loss_limit': oof_loss_limit,
            'fold_type':fold_type
        }

        nn_model = NNWrapper(TabularNN,CFG)
        nn_model.fit(X,y,folds=folds,evaluate=False,oof_idx=oof_idx,moa_control_params=moa_control_params,seed=seed)

        dict_model = {
            'nn_model':nn_model,
            'X_train':X_train,
            'y_train':y_train,
            'X_holdout':X_holdout,
            'y_holdout':y_holdout
        }
        nn_model_list.append(dict_model)

        if not export_path is None:
            print('Saving models')
            for i,model_dict in enumerate(nn_model_list):
                model_dict['nn_model'].save_model(name=export_path+'seed_'+str(i)+'_')

        return nn_model_list