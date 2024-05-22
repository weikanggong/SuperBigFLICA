# SuperBigFLICA
The code implements the algorithm in the following paper:
```
Gong, W., Bai, S., Zheng, Y. Q., Smith, S. M., & Beckmann, C. F. (2022). Supervised phenotype discovery from multimodal brain imaging. IEEE Transactions on Medical Imaging, 42(3), 834-849.

```

In python, install pytorch first (https://pytorch.org/get-started/locally/), and then please use the following two functions to perform the analysis:

```
pred_valid, best_model, loss_all_test, best_corr, final_model = SupervisedFLICA(x_train = Data_train, y_train = y_train, x_test = Data_valid, y_test = y_valid,
                                                              dropout=0.2, device = 'cpu',auto_weight = [1,1,1,1], lambdas = [relative_weight, relative_weight, 1-relative_weight, 1-relative_weight], nlat= nIC ,lr=lr, random_seed = 555, maxiter=50, batch_size=512, init_method = 'random')
                   
lat_train,lat_test, spatial_loadings, modality_weights, prediction_weights, pred_train, pred_test = get_model_param(x_train = Data_train, x_test = Data_test, y_train=y_train, best_model = best_model)
```

Please divide your data into train, validation, and test sets.

**1. The first function "SupervisedFLICA" trains the model, and then use a validation set to perform the model selection.**  
Inputs:  
**x_train**: a list, each element is a subject-by-feature matrix of an imaging modality (without NaN), training set.  
**x_test**: a list, each element is a subject-by-feature matrix of an imaging modality (without NaN), test set.  
**y_train**: a matrix, each is subject-by-nIDP (could contain NaN in it), training set nIDP.  
**y_test**: a matrix, each is subject-by-nIDP (could contain NaN in it), test set nIDP.  
**relative_weight**: a weight that balances the imaging reconstuction loss and nIDP prediction loss, you can specify it in (0,1). The smaller the relative_weight, the larger the imaging reconstuction loss.  
**nlat**: the number of components (i.e., ICs) for SuperBigFLICA.  
**lr**: the learning rate (e.g. 0.001)  
**batch_size**: the batch size used for optimization.  
**device**: 'cpu' if uses CPU for training, 'cuda' if use GPU.  
**dropout**: The probability of dropout the imaging data in training. 
**random_seed**: The seed used for the model to reproduce the results.
You can keep other parameters as default.  

Outputs:  
**pred_valid**: the predicted nIDP in the validation set.  
**best_model**: the model that can give the best nIDP prediction in the validation set.  
**loss_all_test**: the training losses.  
**best_corr**: the evaluation metric (If there is only one nIDP, the metric is the correlation; if there are multiple nIDPs, the metric is the sum of correlations that are larger than 0.1).  
**final_model**: the model of the last epoch.


**2. The second function "get_model_param" apply model to the test dataset.**  
Inputs:  

**best_model**: the model used for eval. If you don't want to use the best performed model in the validation set, you can also use the "final_model" output by the first function (the model of the last epoch).  

Outputs:  
**lat_train**: the multimodal shared latent variables (subject-by-nlat), use it to correlate/predict other nIDPs.   
**lat_test**: the multimodal shared latent variables (subject-by-nlat), use it to correlate/predict other nIDPs.   
**spatial_loadings**: a list, each element is a voxel-by-nlat independent spatial loading matrix. The loadings has been z-score normalized by regress the lat_train onto the original data x_train.   
**modality_weights**: a nlat-by-modality matrix, it is the contribution of each modality to each latent component.  
**prediction_weights**: a nlat-by-#nIDP matrix, the trained weights of predicting each of the nIDPs using the latent components.  
**pred_train, pred_test**: the predicted nIDPs by the trained model in training set and test set.  


**3. Some other usages:**  
3.1 Unsupervised IDP discovery: one may not have a specific set of nIDP to predict, but only wants to discover imaging latent features from multimodal data. This is the same as done in the BigFLICA:
```
Gong, W., Beckmann, C. F., & Smith, S. M. (2021). Phenotype discovery from population brain imaging. Medical image analysis, 71, 102050.
```
you can specify "y_train", "y_valid" and "y_test" parameters as random noise in "SupervisedFLICA" and "get_model_param" functions, and set the "relative_weight" parameters as a very large number close to 1, e.g., 0.999999999999, in the above analysis. In this way, the output should be highly similar to the BigFLICA.











