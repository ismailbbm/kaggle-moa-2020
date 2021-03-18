import pandas as pd
import numpy as np

def pong_number(a,size):
    divisor = a//size
    remaining = a%size
    is_even = divisor % 2 == 0
    if is_even:
        return remaining
    else:
        return size-remaining

def select_split_drugs(X,ratio):
    treatment_drugs = X['drug_id'].values
    drug_ids, drug_counts = np.unique(treatment_drugs, return_counts=True)
    drug_id_counts = np.array([drug_ids,drug_counts]).transpose()
    drug_id_counts = drug_id_counts[drug_id_counts[:,1].argsort()]
    n_drugs = drug_id_counts.shape[0]
    
    #Randomly chose drugs that will appear only in holdout
    holdout_drugs = []
    array_size = X.shape[0]
    limit = int(array_size*ratio)
    count_drugs = 0
    random_range = int(1/ratio)+1
    counter = 0
    
    
    while count_drugs<limit:
        choice = random_range*counter + np.random.randint(0,random_range)
        choice = pong_number(choice,n_drugs-1)
        holdout_drugs.append(drug_id_counts[choice,0])
        count_drugs += drug_id_counts[choice,1]
        #holdout_drugs.append(np.take(drug_id_counts,choice,mode='wrap',axis=0)[0])
        #count_drugs += np.take(drug_id_counts,choice,mode='wrap',axis=0)[1]
        counter += 1
        
    return holdout_drugs