import numpy as np
from sklearn import metrics

from model.postprocessing import post_process, get_available_exclusivity_tuples
from utils import write_yaml_conf

def evaluate_nn_model_list(prepared_data, nn_model_list,ls_floor=1e-5,ls_ceil=1e-5,activation_threshold=0.3,floor=1e-5):
    results = []
    for model_dict in nn_model_list:
        results.append(
            evaluate_nn_model(
                prepared_data,
                model_dict['X_train'],
                model_dict['y_train'],
                model_dict['X_holdout'],
                model_dict['y_holdout'],
                model_dict['nn_model'],
                ls_floor,
                ls_ceil,
                activation_threshold,
                floor
            )
        )
    results = np.array(results)
    results_mean = results.mean(axis=0)
    print("oof: {:.6f} \t oof post: {:.6f} \t oof ls: {:.6f} \t oof excl: {:.6f} \t holdout: {:.6f} \t holdout post: {:.6f} \t holdout ls: {:.6f} \t holdout excl: {:.6f}".format(
        results_mean[0],results_mean[1],results_mean[2],results_mean[3],
        results_mean[4],results_mean[5],results_mean[6],results_mean[7]
    ))
    return results_mean

def evaluate_nn_model(prepared_data, X_train,y_train,X_holdout,y_holdout,nn_model,ls_floor=1e-5,ls_ceil=1e-5,activation_threshold=0.3,floor=1e-5):
    exclusivity_tuples_idx = get_available_exclusivity_tuples(prepared_data)
    #OOF
    y_oof_pred = (y_train.values.copy()).astype(float)
    X_oof = X_train.values 
    y_oof_pred_all = y_oof_pred
    X_oof_all = X_oof
    
    for (_, val_idx), model in zip(nn_model._oof_idx, nn_model.cv_models):
        y_oof_pred[val_idx,...] = nn_model._predict_proba_model(model,nn_model._X[val_idx])
    
    postProcess_oof = post_process(X_oof_all,y_oof_pred_all)
    y_oof_post_processed = postProcess_oof.control_to_zero()
    y_oof_post_processed_ls = postProcess_oof.label_smoothing(ls_floor,ls_ceil)
    y_oof_post_processed_exclusivity = postProcess_oof.exclusivity(exclusivity_tuples_idx,ls_floor,ls_ceil,activation_threshold,floor)
    
    oof_loss = metrics.log_loss(y_train.values.reshape(-1,1),y_oof_pred_all.reshape(-1,1))
    oof_loss_post_processed = metrics.log_loss(y_train.values.reshape(-1,1),y_oof_post_processed.reshape(-1,1))
    oof_loss_post_processed_ls = metrics.log_loss(y_train.values.reshape(-1,1),y_oof_post_processed_ls.reshape(-1,1))
    oof_loss_post_processed_exclusivity = metrics.log_loss(y_train.values.reshape(-1,1),y_oof_post_processed_exclusivity.reshape(-1,1))
    
    #Holdout
    y_holdout_pred = nn_model.predict(X_holdout.values[:,3:])
    
    postProcess_holdout = post_process(X_holdout.values,y_holdout_pred)
    
    y_holdout_pred_post_processed = postProcess_holdout.control_to_zero()
    y_holdout_pred_post_processed_ls = postProcess_holdout.label_smoothing(ls_floor,ls_ceil)
    y_holdout_pred_post_processed_exclusivity = postProcess_holdout.exclusivity(exclusivity_tuples_idx,ls_floor,ls_ceil,activation_threshold,floor)
    
    holdout_loss = metrics.log_loss(y_holdout.values.reshape(-1,1),y_holdout_pred.reshape(-1,1))
    holdout_loss_post_processed = metrics.log_loss(y_holdout.values.reshape(-1,1),y_holdout_pred_post_processed.reshape(-1,1))
    holdout_loss_post_processed_ls = metrics.log_loss(y_holdout.values.reshape(-1,1),y_holdout_pred_post_processed_ls.reshape(-1,1))
    holdout_loss_post_processed_exclusivity = metrics.log_loss(y_holdout.values.reshape(-1,1),y_holdout_pred_post_processed_exclusivity.reshape(-1,1))
    
    return [oof_loss,oof_loss_post_processed,oof_loss_post_processed_ls,oof_loss_post_processed_exclusivity,
        holdout_loss,holdout_loss_post_processed,holdout_loss_post_processed_ls,holdout_loss_post_processed_exclusivity]


def save_config_score(conf, results_mean, path):
    results = {
        'oof': str(results_mean[0]),
        'oof post': str(results_mean[1]),
        'oof ls': str(results_mean[2]),
        'oof excl': str(results_mean[3]),
        'holdout': str(results_mean[4]),
        'holdout post': str(results_mean[5]),
        'holdout ls': str(results_mean[6]),
        'holdout excl': str(results_mean[7])
    }
    conf['results'] = results

    write_yaml_conf(path, conf)