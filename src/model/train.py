import numpy as np
import datetime
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import multiprocessing

from utils import seed_everything
from model.models import TabularNN, CFG


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        bcs_loss = nn.BCEWithLogitsLoss()(x, target)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * bcs_loss + self.smoothing * smooth_loss
        return loss.mean()

class NNWrapper():

    def __init__(self,model_class,cfg):
        self.model_class = model_class
        self.cfg = cfg
        self.clamp = 1e-7 #To avoid having 0 or 1 in logloss
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _evaluate(self,y_actual,y_pred):
        if set([0,1])==set(self.labels):
            return metrics.log_loss(y_actual.reshape(-1,1),y_pred.reshape(-1,1),labels=self.labels)
        else:
            return -np.sum(y_actual.reshape(-1,1)*np.log(y_pred.reshape(-1,1)) + (1-y_actual.reshape(-1,1))*np.log(1-y_pred.reshape(-1,1)))/y_pred.reshape(-1,1).shape[0]
        
    def _numpy_to_tensor(self,n):
        t = (torch.from_numpy(n)).float()
        return t
    
    def _index_to_boolean(self,idx,size):
            mask_array = np.zeros(size)
            mask_array = mask_array==1
            mask_array[idx] = True
            return mask_array
        
    def fit(self,X,y,X_holdout=None,y_holdout=None,folds=5,params=None,evaluate=True,oof_idx=None,seed=42,moa_control_params=None):
        seed_everything(seed=seed)
        #Information on data
        self.labels = list(np.unique(y))
        
        
        #Parameters particular to MoA
        self.moa_control_params = moa_control_params
        
        
        if oof_idx is None:
            self._oof_idx = []
            if (y.ndim == 2) and (y.shape[1] > 1):
                cv = KFold(n_splits=folds, shuffle=True, random_state=42)
                if not self.moa_control_params is None:
                    if ['fold_type']=='multi':
                        cv = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            else:
                cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            for (train_idx, val_idx) in cv.split(X,y):
                self._oof_idx.append((train_idx,val_idx))
        else:
            self._oof_idx = oof_idx
        
        #Train models
        self.cv_models = []
        self._X = X.copy()
        self.total_oof_loss = 0
        self._y = y
        for fold, (train_idx, val_idx) in enumerate(self._oof_idx):
            print("Start training fold {} of {}".format(fold+1,folds))
            if self.moa_control_params is None:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
            else:
                train_mask = self._index_to_boolean(train_idx,X.shape[0])
                val_mask = self._index_to_boolean(val_idx,X.shape[0])
                
                train_mask_non_control = train_mask & self.moa_control_params['mask_treatment']
                self.train_mask_non_control = train_mask_non_control
                train_mask_control = train_mask & (self.moa_control_params['mask_treatment']==False)
                val_mask_non_control = val_mask & self.moa_control_params['mask_treatment']
                
                X_train, X_val = X[train_mask_non_control], X[val_mask_non_control]
                y_train, y_val = y[train_mask_non_control], y[val_mask_non_control]
                
                if self.moa_control_params['add_control_from_test']:
                    self.X_control = np.concatenate([X[train_mask_control],self.moa_control_params['test_control_data']],axis=0)
                else:
                    self.X_control = X[train_mask_control]
            
            model = self._train(X_train,y_train,X_val,y_val)
            self.cv_models.append(model)
            
            if evaluate:
                #Evaluate fold model
                oof_preds = self._predict_proba_model(model,X_val)
                oof_loss = self._evaluate(y_val,oof_preds)
                self.total_oof_loss = self.total_oof_loss + oof_loss/folds
                if not X_holdout is None:
                    holdout_preds = self._predict_proba_model(model,X_holdout)
                    holdout_loss = self._evaluate(y_holdout,holdout_preds)

                if not X_holdout is None:
                    print('Fold {} out of fold score: {:.6f}; holdout score: {:.6f}'.format(fold+1,oof_loss,holdout_loss))
                else:
                    print('Fold {} out of fold score: {:.6f}'.format(fold+1,oof_loss))
             
        #Evaluate whole model
        if evaluate:
            if not X_holdout is None:
                y_pred_holdout = self.predict(X_holdout)
                holdout_loss = self._evaluate(y_holdout,y_pred_holdout)
            if not X_holdout is None:
                print('Total oof score: {:.6f}; holdout score: {:.6f}'.format(self.total_oof_loss,holdout_loss))
            else:
                print('Total oof score: {:.6f}'.format(self.total_oof_loss))

            
    def _prepare_batch_data(self,X,y=None,inference=False):
        if inference:
            dataset = DataLoader(TensorDataset(X),batch_size=X.shape[0],shuffle=False,pin_memory=True,num_workers=multiprocessing.cpu_count()-1)
        else:
            #Before passing it to DataLoader, X,y need to be in the following dimension: items, features
            dataset = DataLoader(TensorDataset(X,y),batch_size=self.cfg.batch_size,shuffle=True,pin_memory=True,num_workers=multiprocessing.cpu_count()-1,drop_last=True)
        return dataset
    
    
    def _calculate_controle_size_to_add(self,X_train):
        #Calculate how many we take from control pool
        control_size_to_add = min(math.floor(X_train.shape[0]*self.moa_control_params['control_share']),self.X_control.shape[0])
        print('Add {} control to training set, representing {:.0f}% of vehicle size'.format(control_size_to_add,100*control_size_to_add/X_train.shape[0]))
        return control_size_to_add
    
    def _moa_add_control_data(self,X_train,y_train,control_size_to_add):
        
        if control_size_to_add>0:
            mask_control_to_add = np.random.choice(np.arange(0,self.X_control.shape[0]), size=control_size_to_add, replace=False)
            X_control_to_add = self.X_control[mask_control_to_add]
            X_train = np.concatenate([X_train,X_control_to_add],axis=0)

            y_control_to_add = np.zeros((X_control_to_add.shape[0],y_train.shape[1]),dtype=y_train.dtype)
            y_train = np.concatenate([y_train,y_control_to_add],axis=0)
        
        X_train_torch = self._numpy_to_tensor(X_train)
        y_train_torch = self._numpy_to_tensor(y_train)
        
        return X_train_torch,y_train_torch
    
    
    def _augment_data(self,X,var_list,mask,epoch):
        var_size = len(var_list)
        i = (epoch-1)%var_size
        var = var_list[i]
        var_val = var[mask]
        X = X + var_val
        return X
        
            
    def _train(self, X_train, y_train, X_val, y_val):
        initial_time = datetime.datetime.now()
        
        X_train_size = X_train.shape[0]

        if not self.moa_control_params is None:
            control_size_to_add = self._calculate_controle_size_to_add(X_train)
            X_train_size += control_size_to_add
        
        #Format data
        X_train_torch = self._numpy_to_tensor(X_train)
        y_train_torch = self._numpy_to_tensor(y_train)
        X_val_torch = self._numpy_to_tensor(X_val)
        y_val_torch = self._numpy_to_tensor(y_val)
        
        #Initialize model
        model = self.model_class(self.cfg,y_train_torch)
        model.to(self.device)
        
        #Initialize model parameters
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
        loss_val_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
        
        if not self.moa_control_params is None:
            if self.moa_control_params['label_smoothing']>0:
                loss_fn = LabelSmoothingCrossEntropy(self.moa_control_params['label_smoothing'])
        
        optimizer = torch.optim.Adam(model.params, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        
        
        total_steps_oneCycle = (int(X_train_size/self.cfg.batch_size)+1)*self.cfg.one_cycle_epochs
        continue_OneCycleLR = True
        scheduler2 = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.3, div_factor=2, 
                                              max_lr=self.cfg.learning_rate*5, total_steps=total_steps_oneCycle,
                                              )
        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.cfg.patience, factor=0.9, threshold=1e-5)
    
        #Print parameters
        if hasattr(self.cfg, 'verbose'):
            verbose = self.cfg.verbose
        else:
            verbose = 1
        
        #Train model
        
        for epoch in range(self.cfg.epochs):
            model.train()
            avg_loss = 0.0
            
            if not self.moa_control_params is None:
                if (self.moa_control_params['augment_data']) & (epoch>0):
                    X_train_tmp = self._augment_data(X_train,
                                                     self.moa_control_params['augment_var_list'],
                                                     self.train_mask_non_control,
                                                     epoch
                                                    )
                else:
                    X_train_tmp = X_train.copy()
                X_train_torch, y_train_torch = self._moa_add_control_data(X_train_tmp, y_train, control_size_to_add)
            
            #We create the train batch dataset here to be able to shuffle it
            dataset = self._prepare_batch_data(X_train_torch,y_train_torch)
            
            #Train trhough batches
            for X_batch, y_batch in dataset:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = model(X_batch)
                
                loss = loss_fn(y_pred, y_batch)
                avg_loss += loss.item() / (len(dataset))
                
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()
                
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()
                
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()
                if (continue_OneCycleLR) and (self.cfg.number_one_cycle>0):
                    scheduler2.step()

            #OneCycle update
            if (self.cfg.number_one_cycle==0):
                continue_OneCycleLR = False
            if (continue_OneCycleLR):
                if (epoch+1)%(self.cfg.one_cycle_epochs*self.cfg.number_one_cycle)==0:
                    print('Finish One Cycle')
                    continue_OneCycleLR=False
                    for g in optimizer.param_groups:
                        g['lr'] = self.cfg.learning_rate
                    
                if ((epoch+1)%self.cfg.one_cycle_epochs==0) and (epoch+1)<(self.cfg.one_cycle_epochs*self.cfg.number_one_cycle):
                    print('Reinitialize One Cycle')
                    scheduler2 = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.3, div_factor=2, 
                                              max_lr=self.cfg.learning_rate*5, total_steps=total_steps_oneCycle,
                                              )
            
            
            #Evaluation
            model.eval()
            y_pred_val = self._predict_proba_model(model,X_val_torch,for_loss=True)
            loss_val = loss_val_fn(y_pred_val, y_val_torch).item()
            
            check_early_stopping, is_new_best = self._should_stop(epoch,loss_val,model,self.cfg.early_stopping,self.cfg.min_delta,avg_loss,self.cfg.min_epochs,self.cfg.hard_patience,self.cfg.ratio_train_val)
            if is_new_best:
                text_new_best = 'New Best'
            else:
                text_new_best = ''
            if (epoch+1)%verbose == 0:
                print('Epoch {}/{} \t train loss: {:.5f} \t val loss: {:.5f} \t time: {}s \t lr: {:.6f} \t {}'.format(epoch+1,self.cfg.epochs,avg_loss,loss_val, (datetime.datetime.now() - initial_time).seconds, optimizer.param_groups[-1]['lr'],text_new_best))
                pass
                
            if not self.cfg.early_stopping is None:
                if check_early_stopping:
                    print("Early stopping, best epoch: {}".format(self._best_epoch+1))
                    model_to_return = self._load_best_model()
                    return model_to_return
                
            if not self.moa_control_params is None: 
                if loss_val < self.moa_control_params['oof_loss_limit']:
                    print('Model decrease below limit')
                    return model

            #Update lr when oneCycle is over
            if not continue_OneCycleLR:
                scheduler1.step(loss_val)
        
        model_to_return = self._load_best_model()
        
        return model_to_return
    
    def _load_best_model(self):
        model_to_return = self.model_class(self.cfg)
        model_to_return.to(self.device)
        model_to_return.load_state_dict(torch.load('tmp_model_state_dict'))
        return model_to_return
    
    
    def save_model(self,name=''):
        for i,model in enumerate(self.cv_models):
            torch.save(model.state_dict(), name+'cv_'+str(i))
        
    def load_model(self,folds,name=''):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cv_models = []
        for i in range(folds):
            model = self.model_class(self.cfg)
            model.to(self.device)
            model.load_state_dict(torch.load(name+'cv_'+str(i)))
            self.cv_models.append(model)
            
    def load_oof_idx(self,oof_idx):
        self._oof_idx = oof_idx
    def load_X(self,X):
        self._X = X
            
    def _should_stop(self,epoch,val_loss,model,patience,min_delta,train_loss,min_epochs=None,hard_patience=None,ratio_train_val=1.1):
        if min_epochs is None:
            min_epochs = 0
        if hard_patience is None:
            hard_patience = epoch + 1
        if epoch==0:
            self._best_val_loss = val_loss
            #self._best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), 'tmp_model_state_dict')
            self._best_epoch = epoch
            self.epochs_since_best = 1
        if val_loss < self._best_val_loss - min_delta:
            self._best_val_loss = val_loss
            #self._best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), 'tmp_model_state_dict')
            self._best_epoch = epoch
            self.epochs_since_best = 1
            return False, True
        else:
            self.epochs_since_best += 1
            if (epoch - self._best_epoch > patience) and (epoch > min_epochs) and ((val_loss>train_loss*ratio_train_val) or (epoch - self._best_epoch > hard_patience)):
                return True, False
            else:
                return False, False

    def _predict_proba_model(self, model, X, for_loss=False):
        if isinstance(X,np.ndarray):
            X = self._numpy_to_tensor(X)
            
        y_pred = []
        
        dataset = self._prepare_batch_data(X,inference=True)
        
        model.eval()
        with torch.no_grad():
            for X_batch in dataset:
                y = model(X_batch[0].to(self.device))
                if for_loss:
                    y_pred.append(y.detach().cpu())
                else:
                    if self.clamp is None:
                        y_pred.append(y.sigmoid().detach().cpu().numpy())
                    else:
                        y_pred.append(y.sigmoid().clamp(self.clamp,1-self.clamp).detach().cpu().numpy())  
        if for_loss:
            y_pred = torch.cat(y_pred,dim=0)
        else:
            y_pred = np.concatenate(y_pred)
        return y_pred
        
    def predict(self,X):
        output_dim = self.cfg.target_cols
        
        y = np.zeros((X.shape[0],output_dim))
        y = np.repeat(y[..., np.newaxis], len(self.cv_models), axis=-1)
        for i,model in enumerate(self.cv_models):
            y[...,i] = self._predict_proba_model(model,X)
        y = y.mean(axis=-1)
        return y
    
    def predict_oof(self):
        y = (self._y.copy()).astype(float)
        for (_, val_idx), model in zip(self._oof_idx, self.cv_models):
            y[val_idx,...] = self._predict_proba_model(model,self._X[val_idx])
        return y