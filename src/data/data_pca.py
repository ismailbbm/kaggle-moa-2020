import pandas as pd
import numpy as np

from sklearn import decomposition


class calculatePCA:
    def __init__(self):
        self.pca_cols_dict = {}
        self.pca_dict = {}

    def fit_pca(X_list,pca_for_kde,pca_for_c):
        X = pd.concat(X_list,axis=0)
        all_cols = X.columns
        pca_cols = []
        pca_names = ['g_pca']
        if pca_for_c:
            pca_names.append('c_pca')
        pca_cols.append([x for x in all_cols if ('g-' in x) & (not '_kde_diff' in x)])
        if pca_for_c:
            pca_cols.append([x for x in all_cols if ('c-' in x) & (not '_kde_diff' in x) & (not '_stats' in x)])
        if pca_for_kde:
            pca_cols.append([x for x in all_cols if ('g-' in x) & ('_kde_diff' in x)])
            if pca_for_c:
                pca_cols.append([x for x in all_cols if ('c-' in x) & ('_kde_diff' in x) & (not '_stats' in x)])
            pca_names.append('g_kde_pca')
            if pca_for_c:
                pca_names.append('c_kde_pca')


        for name,cols in zip(pca_names,pca_cols):
            if len(cols)>0:
                X_pca = X[cols]
                pca = decomposition.PCA(n_components=X_pca.shape[1],
                                            whiten=True,
                                            svd_solver='full',
                                            random_state=42
                                            )
                pca.fit(X_pca)
                self.pca_cols_dict[name] = cols
                self.pca_dict[name] = pca    
            
            
    def calculate_pca_components_to_keep(self,explained_variance_ratio_,pca_variance_threshold):
        explained_variance_ratio_cum = explained_variance_ratio_.cumsum()
        return np.argmax(explained_variance_ratio_cum>=pca_variance_threshold) + 1

    def transform_pca(self,X,pca_variance_threshold):
        pca_names = list(self.pca_cols_dict.keys())
        for name in pca_names:
            #Recover cols and fit pca
            cols = self.pca_cols_dict[name]
            pca = self.pca_dict[name]

            #Transform to current data
            X_pca = pca.transform(X[cols])

            #Keep only necessary data + transform into pd
            variance_limit = self._calculate_pca_components_to_keep(pca.explained_variance_ratio_,pca_variance_threshold)
            X_pca = X_pca[:,:variance_limit]
            new_cols = [name+'_'+str(i) for i in range(variance_limit)]
            X_pca = pd.DataFrame(X_pca,columns=new_cols)

            #Adjust X
            X.drop(cols,axis=1,inplace=True)
            X = pd.concat([X,X_pca],axis=1)

        return X