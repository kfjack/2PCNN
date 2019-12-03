# 2PCNN

Demonstration code for two-particle correlation neural network (2PCNN). 
See the reference for more information:
https://arxiv.org/abs/1911.02020

## File descriptions:
* prototype_train.py
> construct a 2PCNN prototype model with only energy flow information (pt, eta, phi for 2 particles), 
> loading the jet data, and training with early stopping enabled.
> Recommened to increase the data statistics for a serious training.
* prototype_deploy.py
> similar to the previous code, instead of training, only construct the same model, loading the trained weights,
> and produce the resulting scores and ROC curve.

## The test samples are available at 
https://drive.google.com/drive/u/0/folders/1HojGLS_ODr7E7tndy2vz0fxRB2N10VAR
