import numpy as np

from model.exclusivity_labels import exclusivity_tuples

class post_process():
    def __init__(self,X,y):
        self.X = X.copy()
        self.y = y.copy()
        
    def control_to_zero(self,y=None):
        if y is None:
            y = self.y.copy()
        control_idx = self.X[:,0]==1
        y[control_idx,:] = 0
        return y
    
    def label_smoothing(self,ls_floor,ls_ceil,with_control=True):
        y = self.y.copy()
        y = y.clip(ls_floor,1-ls_ceil)
        if with_control:
            y = self.control_to_zero(y)
        return y
    
    def exclusivity(self,exclusivity_tuples_idx,ls_floor,ls_ceil,activation_threshold=0.3,floor=1e-5):
        y = self.y.copy()
        y = self.label_smoothing(ls_floor,ls_ceil)
        
        for tup in exclusivity_tuples_idx:
            y_tmp = y[:,tup]

            max_val = np.amax(y_tmp,axis=1)
            max_mask = max_val>=activation_threshold
            max_mask = np.repeat(np.array([max_mask]),y_tmp.shape[1],axis=0).transpose()

            max_idx = np.argmax(y_tmp,axis=1)
            max_idx = np.repeat(np.array([max_idx]),y_tmp.shape[1],axis=0).transpose()
            y_idx = np.repeat(np.array([np.arange(0,len(tup))]),y_tmp.shape[0],axis=0)

            y_mask = (y_idx!=max_idx) & (max_mask)
            y_global_mask = y==-1
            y_global_mask[:,tup] = y_mask
            y[y_global_mask] = np.clip(y[y_global_mask],0,floor)
        return y
    
def get_available_exclusivity_tuples(prepared_data,
                                     exclusivity_tuples=exclusivity_tuples
                                     ):
    available_cols = prepared_data.y_train.columns.tolist()

    exclusivity_tuples_idx = []
    for tup in exclusivity_tuples:
        if all([x in available_cols for x in tup]):
            idx = []
            for col in tup:
                idx.append(available_cols.index(col))
            exclusivity_tuples_idx.append(idx)

    return exclusivity_tuples_idx