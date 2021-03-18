# kaggle-moa

Competition description: https://www.kaggle.com/c/lish-moa

Link to my kaggle profile: https://www.kaggle.com/ismailbbm

Summary: based on cellular response to a particular drug, the goal is to predict which mechanisms of action are perturbed in the cells (e.g. "potassium channel activator")

# Notes on the model and kaggle competitions

Kaggle competitions make available a training dataset and rank model performance based on a private dataset which becomes available after the competition ends, scores on this private dataset are also only available after the end of the competition.

Usually the biggest differentiator between a top solution and a good solution is the ability to construct a model which will generalize very well to unknown data as private data is in general a bit different from the training data.

However in this competition, in my opinion, data leaks from the training dataset to the private dataset were large. Indeed, a significant part of the drugs available in the training set were also present in the private dataset. While I don't have medical expertise, I am still wondering how is it useful scientifically to predict the MoA of drugs for which we already have the data.

The model built here used a cross validation such as the model performance is evaluated only on drugs that were not seen in the training data. Therefore the model minimize overfitting to already seen drugs. This validation scheme was chosen by assuming that the private dataset will be majoritarly constituted of new drugs not present in the private dataset.

# Model

