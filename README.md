# SuperBigFLICA: test version 0.1
Gong, W., Bai, S., Zheng, Y. Q., Smith, S. M., & Beckmann, C. F. (2022). Supervised Phenotype Discovery from Multimodal Brain Imaging. IEEE Transactions on Medical Imaging. 

In python, install pytorch first (https://pytorch.org/get-started/locally/), and then please use the following two functions to perform the analysis:

‘’‘
pred_valid, best_model, loss_all_test, best_corr = SupervisedFLICA(x_train = Data_train, y_train = y_train, x_test = Data_valid, y_test = y_valid,
                                                              dropout=0.2, device = 'cpu',auto_weight = [1,1,1,1], lambdas = [relative_weight,relative_weight,1-relative_weight,1-relative_weight],
                                                              nlat= nIC ,lr=lr, random_seed = 555,maxiter=50,batch_size=512, init_method = 'random')
                   
lat_train,lat_test, spatial_loadings, modality_weights, prediction_weights, pred_train, pred_test = get_model_param(x_train = Data_train, x_test = Data_test, best_model = best_model)
’‘’

Please divide your data into train, validation, and test. Then,
The first function trains the model, use a validation set to select the best model.                     
x_train, x_test are lists, each element is a subject-by-feature matrix of a modality (without NaN).
y_train, y_test are matrix, each is subject-by-nIDP (can have NaN).
relative_weight is the weight balances the data recon loss and prediction loss, you can specify it in (0,1).
nlat is the number of components for superbigflica,
lr is the learning rate (e.g. 0.001)
You can keep other parameters as they are. These are values use in my paper.


The second function apply model to the test dataset.

For the output,
lat_train,lat_test are shared latent variables (subject-by-nlat), use it to correlate/predict other nIDPs.
spatial_loadings is a list, each element is a voxel-by-nlat independent spatial loading matrix.
modality_weights is a nlat-by-modality matrix, it is the contribution of each modality to each latent component.
prediction_weights is a nlat-by-#nIDP matrix, shows the trained weights of predicting each of the nIDPs using the latent components.
pred_train, pred_test are the predicted nIDPs by the trained model in training set and test set.
