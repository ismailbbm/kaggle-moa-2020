# kaggle-moa

Competition description: https://www.kaggle.com/c/lish-moa

Link to my kaggle profile: https://www.kaggle.com/ismailbbm

Summary: based on cellular response to a particular drug, the goal is to predict which mechanisms of action are perturbed in the cells (e.g. "potassium channel activator")

# Notes on the model and kaggle competitions

Kaggle competitions make available a training dataset and rank model performance based on a private dataset which becomes available after the competition ends, scores on this private dataset are also only available after the end of the competition.

Usually the biggest differentiator between a top solution and a good solution is the ability to construct a model which will generalize very well to unknown data as private data is in general a bit different from the training data.

However in this competition, in my opinion, data leaks from the training dataset to the private dataset were large. Indeed, a significant part of the drugs available in the training set were also present in the private dataset. While I don't have medical expertise, I am still wondering how is it useful scientifically to predict the MoA of drugs for which we already have the data.

The model built here used a cross validation such as the model performance is evaluated only on drugs that were not seen in the training data. Therefore the model minimize overfitting to already seen drugs. This validation scheme was chosen by assuming that the private dataset will be majoritarly constituted of new drugs not present in the private dataset.

# Project structure

### src

Contains:
* data : preparation scripts
* model : model training and evaluation
* conf.yaml : configuration for the data preparation and the model

### Notebooks

* experiment: notebook to run and evaluate experiments
* run-in-kaggle: as this was a code competition, the submission is a notebook. This is the submission notebook.
* exploration: notebook with the main results of the data exploration


# Model

The model is built upon 3 ideas.

### Multi label binary classification neural network
The model is a dense neural network using pytorch. Neural networks have been performing better than more traditional methods (gradient boosting, logistic regression) for binary classification. The hypothesis is that since we are predicting >100 interdependant binary labels simultaneously, neural nets are better at capturing potential dependencies between the labels.

### Using control drugs to better identify MoA
Splitting the drugs experiments in two groups (control and vehicle) and then plotting the distribution of their impact on cellular response, we can better identify how real drugs differ from control drugs.

![kde.png](https://github.com/ismailbbm/kaggle-moa-2020/images/kde.png)

In red is one of the cellular response distribution from a active drug, blue is from control.
Using this difference, I compute a kernel which will transform the cellular response value into a new value which aims at better identifying cellular responses provoked by a real drug.

![kde_feature.png](https://github.com/ismailbbm/kaggle-moa-2020/images/kde_feature.png)

This feature engineering improved the out of fold score when the validation fold contained unknown drugs. However, this engineering did not produce any improvement when the validation fold contained already seen drugs.


### Excluding some cellular responses

Some of the distribution graphs (control vs vehicle) showed that certain cellular responses did not present differences. Those features were excluded. Optimally, 75 out of 700 features were excluded from the data imporving the scores in both types of cross validation (with validation containing seen drugs and with validation containing not seen drugs).
