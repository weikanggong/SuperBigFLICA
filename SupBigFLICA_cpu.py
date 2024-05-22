import numpy as np
import scipy
from scipy import linalg
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
import copy


def nets_zscore(x):
    # x : a nsubject * nfeature numpy matrix
    
    x_zscore=(x-x.mean(axis=0))
    stds=x.std(axis=0)
    index=stds==0
       
    if sum(index)>0:
        stds[index]=0.1
        print('Warning: '+str(sum(index))+' of the features are all zero or constants')
        print('Normalizing them to all zeros ...')
    
    x_zscore=x_zscore/stds
    
    return x_zscore

def g_impute_nan_as_mean(x):
    
    x_mean = np.nanmean(x,axis = 0)
    for i in range(0,x.shape[1]):
        x[np.isnan(x[:,i]),i] = x_mean[i]
    
    return x


def SingleModality_MIGP(x, k = 10 ,subj_batch = 200, n_epoch = 1):
    #x is a nsubject * nfeature matrix
    #Online PCA across the rows of x, to extract k PCs
    #output is a nfeature * k matrix
    
    for j in range(0,n_epoch):

        random.seed(j)
        np.random.seed(j)
        
        print('Epoch: ' + str(j+1)+'...')
        
        d1=x.shape[0]
        
        if j>=1:
            random.seed(j)
            np.random.seed(j)            
            x = x[np.random.permutation(d1),:]
        
        subj_batch = subj_batch + k
        
        ind_end=int(d1/np.float64(subj_batch)-1e-4)+1
        
        if j==0:
            st=int(0*subj_batch)
            en=min(d1,int((0+1)*subj_batch))

            W=x[st:en,:]
            
        for i in range(1,ind_end):
            
            #print(i)
            st=int(i*subj_batch)
            en=min(d1,int((i+1)*subj_batch))
            

            W=np.vstack((W,x[st:en,:]))
            
            d,u=linalg.eigh(np.dot(W,W.T),eigvals=(W.shape[0]-2*k,W.shape[0]-1))
            
            d=np.real(d)
            u=np.real(u)
            indx1=np.argsort(-d)
            d=d[indx1]
            u=u[:,indx1]
      
            W=np.dot(u.T,W)           
    
    
    us = W[0:k,:].T
    
    #u = us / np.sqrt(np.sum(us**2,axis = 0))
    
    return us


def MultiModality_MIGP(x, k = 10 ,subj_batch = 200, n_epoch = 1, zscore = False):
    #x is a list of length k , each is [nsubject * nfeature]
    #doing online pca on feature dimensions
    #equivalent to do a pca on feature concat data nsubject * (nfeature * k)
    #output a nsubject * k matrix
        
    nmod = len(x)
    for i in range(0,nmod):
        print('Multi-Modality MIGP for Modality '+ str(i+1)+'...')
        if i==0:
            if zscore==True:
                uu = SingleModality_MIGP(nets_zscore(x[i]).T, k  ,subj_batch , n_epoch )
            else:
                uu = SingleModality_MIGP(x[i].T, k  ,subj_batch , n_epoch )                
        else:
            if zscore==True:
                uu = SingleModality_MIGP(np.hstack((uu,nets_zscore(x[i]))).T, k  ,subj_batch , n_epoch )
            else:
                uu = SingleModality_MIGP(np.hstack((uu,x[i])).T, k  ,subj_batch , n_epoch )
    
    u = uu[:,0:k]
    
    return u

def nets_svds(x,nComp):
    # x : a nsubject * nfeature numpy matrix
    # nComp : the number of dimension (int), should be < min(x.shape[0], x.shape[1])
    
    if x.shape[0] < x.shape[1]:
        
        cov_mat=np.dot(x,x.T)
        if nComp < x.shape[0]:
            
            d,u=linalg.eigh(cov_mat,eigvals=(x.shape[0]-nComp,x.shape[0]-1))
            d=np.real(d)
            u=np.real(u)
            indx1=np.argsort(-d)
            d=d[indx1]
            u=u[:,indx1]
    
        s = np.sqrt(np.abs(d))
        v = np.dot(x.T , np.dot(u , np.diag(1/s)  ))

    else:
    
        cov_mat=np.dot(x.T,x)
        if nComp < x.shape[1]:
            
            d,v=linalg.eigh(cov_mat,eigvals=(x.shape[1]-nComp,x.shape[1]-1))
            d=np.real(d)
            v=np.real(v)
            indx1=np.argsort(-d)
            d=d[indx1]
            v=v[:,indx1]
            
            s = np.sqrt(np.abs(d));
            u = np.dot(x , np.dot(v , np.diag(1/s)  ))
        
    return u,s,v

def MultiModality_MIGP_faster(x, k = 10 ,subj_batch = 200, n_epoch = 1, zscore = True):
    #x is a list of length k , each is [nsubject * nfeature]
    #doing online pca on feature dimensions
    #equivalent to do a pca on feature concat data nsubject * (nfeature * k)
    #output a nsubject * k matrix
    
    
    nmod=len(x)
    nsub=x[0].shape[0]
    if nmod == 1:        
        u,s,v = nets_svds(x[0],k)
        u = np.dot(u,np.diag(s))
    else:
        cov_mat=np.zeros((nsub,nsub))
        for i in range(0,nmod): 
            dat=nets_zscore(x[i])
            cov_mat=cov_mat+np.dot(dat,dat.T)/dat.shape[1]
        
        if nsub<120000:
            dd,uu=linalg.eigh(cov_mat,eigvals=(nsub-k,nsub-1))
            dd=np.real(dd)
            uu=np.real(uu)
            indx1=np.argsort(-dd)
            dd=dd[indx1]
            uu=uu[:,indx1]
            uu = np.dot(uu,np.diag(dd))
        else:
            uu = SingleModality_MIGP(cov_mat, k  ,subj_batch , n_epoch )
            
        u = uu[:,0:k]
    
    return u

class loss_SuperBigFLICA_regression(nn.Module):
    def __init__(self, ntask, nmod, nsub, device = 'cuda', auto_weight = [1,1,1,1], lambdas = [0.25, 0.25, 0.25, 0.25]):
        super(loss_SuperBigFLICA_regression, self).__init__()
        
        self.ntask = ntask
        self.auto_weight = auto_weight
        self.nmod=nmod
        self.mse = nn.MSELoss()
        self.lambdas = torch.FloatTensor(np.array(lambdas).flatten()).to(device)     
        #weight for reconstruction loss
        self.sigma1 = nn.Parameter(torch.ones(1,nmod,device = device))
        #weight for spatial loadings
        self.sigma2 = nn.Parameter(torch.ones(1,nmod,device = device))
        #weight for mse loss of each task
        self.sigma3 = nn.Parameter(torch.ones(1,ntask,device = device))
        #weight for regression coefs of each task
        self.sigma4 = nn.Parameter(torch.ones(1,ntask,device = device))#l1
        self.sigma5 = nn.Parameter(torch.ones(1,ntask,device = device))#l2
        self.nsub = nsub
               
    def forward(self, recon_x, x_orig, sptial_loadings, y_pred, y_train, pred_weights, lat_train):
        
        batch_prop = recon_x[0].size()[0]/self.nsub
        
        #make these parameters always positive
        sigma1 =self.sigma1**2
        sigma2 =self.sigma2**2
        sigma3 =self.sigma3**2 
        sigma4 =self.sigma4**2
        sigma5 =self.sigma5**2
        
        if self.auto_weight[0] == 0:
            self.sigma1.requires_grad = False
        if self.auto_weight[1] == 0:
            self.sigma2.requires_grad = False
        if self.auto_weight[2] == 0:
            self.sigma3.requires_grad = False
        if self.auto_weight[3] == 0:
            self.sigma4.requires_grad = False
            self.sigma5.requires_grad = False

        loss_recon = 0
        for i in range(0,self.nmod):
            diff = recon_x[i] - x_orig[i]
            loss_recon = loss_recon + (diff * diff).mean() / sigma1[0,i]**2 / 2
        loss_recon = loss_recon + torch.sum(torch.log(sigma1+1)) 

        loss_sptial_loadings = 0
        for i in range(0,self.nmod):         
            loss_sptial_loadings = loss_sptial_loadings + sptial_loadings[i].abs().mean() * batch_prop / sigma2[0,i]
        loss_sptial_loadings = loss_sptial_loadings + 2 * torch.sum(torch.log(sigma2+1)) 
            
        
        index_NaN = torch.isnan(y_train)
        y_train[index_NaN] = y_pred[index_NaN]
        
        diff2 = (y_train - y_pred)**2 / sigma3**2 / 2
        loss_mse = torch.mean(diff2) + torch.sum(torch.log(sigma3 +1))
        
        loss_pred_weights = torch.mean(torch.abs(pred_weights) / sigma4  * batch_prop) + 2 * torch.sum(torch.log(sigma4+1)) #l1
        loss_pred_weights = loss_pred_weights + torch.mean((pred_weights)**2  * batch_prop / sigma5 ** 2 / 2) + torch.sum(torch.log(sigma5+1)) #l2
        
        self.lambdas = self.lambdas / torch.sum(self.lambdas)
        
        l = self.lambdas[0] * loss_recon + self.lambdas[1] * loss_sptial_loadings + self.lambdas[2] * loss_mse + self.lambdas[3] * loss_pred_weights #+ loss_lat
                
        return l, loss_recon, loss_sptial_loadings, loss_mse, loss_pred_weights


import pandas as pd
class load_Multimodal_data1(Dataset):
        def __init__(self, ids, y=None):
            #X is a list of data of size [N,M_k]
            self.ids = ids
            self.filenames = ['connectome_mean_FA_10M.csv','connectome_mean_length_10M.csv',
                              'connectome_sift2_fbc_10M.csv','connectome_streamline_count_10M.csv']
            # self.K = len(X)
            #y is the label of size [N,Q]
            self.y = y
            
            # self.transform = transform
            
        def __len__(self):
            return self.y.shape[0]
        
        def __getitem__(self, index):

            image = []
            for K in range(0,4):
                ff = '/public/home/fuyan/dwi_connectome/' + ff[index] + self.filenames[K]
                # ff = '/public/home/fuyan/dwi_connectome/' + '1000012/' + 'connectome_mean_FA_10M.csv'
                data = pd.read_csv(ff,header=None).to_numpy()
                mask = np.triu(np.ones((data.shape[0], data.shape[1])), 1)
                d = np.triu(data,1)[mask>0]
                d = torch.FloatTensor(d).unsqueeze(0)
                image[K] =  (d - d.mean()) / d.std()               
            
            y = self.y[index,:]
            return image, y
            
class load_Multimodal_data(Dataset):
        def __init__(self, X, y=None, transform=None):
            #X is a list of data of size [N,M_k]
            self.X = X
            self.K = len(X)
            #y is the label of size [N,Q]
            self.y = y
            
            self.transform = transform
            
        def __len__(self):
            return self.X[0].size()[0]
        
        def __getitem__(self, index):
            
            image = []
            for i in range(0,self.K):
                image.append(self.X[i][index,:])
                
            if self.y is not None:
                return image, torch.FloatTensor(self.y[index,:]).float()
            else:
                return image            
            
class SupervisedFLICAmodel(nn.Module):
    def __init__(self,nfea, nlat, ntask, dropout = 0.5, device = 'cpu', init_spatial_loading = None, init_weight_pred=None, init_bais_pred=None):
        
        #nfea: list of length K, number of features in each modality
        #nlat: number of components to extract
        
        super(SupervisedFLICAmodel, self).__init__()
        
        nmod = len(nfea)
        self.ntask=ntask
        self.dropout = dropout
        #initialize the spatial loading of each modality
        if init_spatial_loading is None:
            self.spatial_loading = nn.ParameterList([])
            for i in range(0,nmod):
                self.spatial_loading.append( nn.Parameter(torch.randn(nfea[i],nlat).to(device)) )
                #torch.nn.init.normal_(self.spatial_loading[i])
        else:
            self.spatial_loading = nn.ParameterList([])
            for i in range(0,nmod):
                self.spatial_loading.append( nn.Parameter(torch.FloatTensor(init_spatial_loading[i]).to(device)) )                

        #initialize the modalitity weights (initilize to 1)
        self.mod_weight = nn.Parameter(torch.ones(nlat,nmod).to(device))           
        
        #initalize the prediction weights        
        if init_weight_pred is None:
            self.weight_pred = nn.Parameter(torch.randn(nlat,self.ntask).to(device))
            #torch.nn.init.normal_(self.weight_pred)
            self.bias_pred = nn.Parameter(torch.randn(1,self.ntask).to(device))
            #torch.nn.init.normal_(self.bias_pred)
            
        else:
            self.weight_pred = nn.Parameter(torch.FloatTensor(init_weight_pred).to(device))
            self.bias_pred = nn.Parameter(torch.FloatTensor(init_bais_pred).to(device))

        self.batch_norm = nn.BatchNorm1d(nlat)
        
        #other variables
        self.nlat = nlat    
        self.nmod = nmod   
        
    def forward(self, x, device='cpu'):
        #x is a list of length K [nsub * nfeature]
        
        mod_weight = F.softmax(self.mod_weight,dim=1) 
        
        latents_common = torch.zeros(x[0].size()[0],self.nlat,device=device)
        for i in range(0,self.nmod): 
            dat = F.dropout(x[i], self.dropout, training=self.training)
            latents = dat.matmul(self.spatial_loading[i]).matmul(torch.diag(mod_weight[:,i]))                                                  
            latents_common = latents_common + latents
            
        latents_common = latents_common / self.nmod
        
        #batch norm
        self.batch_norm = self.batch_norm.to(device)
        latents_common = self.batch_norm(latents_common)
        #dropout
        latents_common = F.dropout(latents_common, self.dropout, training=self.training)
        
        output = []  ##the same size as x
        for i in range(0,self.nmod):
            w = (self.spatial_loading[i]).matmul(torch.diag(mod_weight[:,i]))
            output.append( (latents_common.matmul(w.t())) )
        
        #doing prediction         
        pred = latents_common.matmul(self.weight_pred)+self.bias_pred ##it is nsubj_train * 1            

        return output, self.spatial_loading, latents_common, pred, self.weight_pred
                
        
def Initialize_SuperBigFLICA(x_train,y_train,nlat,train_ind = None):
    
    nmod = len(x_train)
    nsubj = x_train[0].shape[0]  
    
    if train_ind is None:
        uu = MultiModality_MIGP_faster(x_train, k = nlat ,subj_batch = 200, n_epoch = 5, zscore = False)
        dd = np.sqrt(np.sum(uu**2,axis = 0, keepdims=True))
        uu = uu / dd       
        
    init_spatial_loading = []
    for i in range(0,nmod):
        init_spatial_loading.append(np.dot(x_train[i].T, uu / dd))
    
    y_train1 = y_train * 1.0
    y_train1[np.isnan(y_train1)] = 0.0
    
    u_share = np.zeros((nsubj,nlat))
    for i in range(0,nmod):
        u_share = u_share + np.dot(x_train[i],init_spatial_loading[i])
    u_share = nets_zscore(u_share)
    u_share = np.hstack((u_share,np.ones((nsubj,1))))
    betas = np.dot(np.dot(np.linalg.pinv(np.dot(u_share.T,u_share)),u_share.T),y_train1)
    init_weight_pred = betas[0:nlat,:]
    init_bais_pred = betas[nlat:(nlat+1),:]

    return init_spatial_loading,init_weight_pred,init_bais_pred, uu


    
def SupervisedFLICA(x_train, y_train, nlat, x_test, y_test, random_seed = 666, train_ind = None, init_method = 'random',
                         dropout = 0.25, device = 'cpu', auto_weight = [1,1,1,1], lambdas = [0.25, 0.25, 0.25, 0.25] ,lr=0.001,batch_size=512,maxiter=50):
    #Inputs:
    #
    #x_train: a list, each is a data matrix of size [N_train,M_k], doing minibatch ICA on N. (N = subject, M_k = features/voxels)
    #         cannot contain NaN.
    #y_train: a data matrix of size [N_train,Q], Q is the number of variables to be predicted, i.e., multitask learning
    #         can contain NaN.
    #nlat: number of latent dimensions
    #x_test: same structure as x_train.
    #y_test: same structure as y_train [N_test,Q].
    #random_seed: specify a random seed for reproducibility.
    #train_ind: leave it as None.
    #init_method: methods of initializing the parameter: 'random' or 'migp'.
    #dropout: the droupout rate.
    #device: 'cpu' or 'cuda' if you want to use gpu.
    #auto_weight: whether doing bayes autoweighting among 4 losses (leave it as default).
    #lambdas: weight on different losses (leave it as equal, i.e., [0.25,0.25,0.25,0.25]).
    #lr: learning rate. try 0.001 to 0.05 
    #batch_size: leave it as 512.
    #maxiter: max number of iterations (usually less than 30).
    
    
    #Outputs:
    #
    #pred_best: best test set prediction [N_test, Q].
    #lat_train: low-dimensional latent variables of training subjects [N_train, nlat].
    #lat_test: low-dimensional latent variables of testing subjects [N_test, nlat].
    #spatial_loadings: spatial loading. [M_k,nlat]
    #modality_weights: weighting of modalities per latent [nlat, n_modalities]
    #prediction_weights: weight of latent to prediction [nlat, Q]
    #pred_train: training set prediction [N_train, Q].
    
    seed = random_seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    nmod = len(x_train)   
    ntask = y_train.shape[1]
    device = torch.device(device)
    
    nfea = []
    for i in range(0,len(x_train)):
        nfea.append(x_train[i].shape[1]) 
        
    print('Data Normalization...')
    is_norm = True
    if is_norm is True:
        for i in range(0,len(x_train)):
            x_mean = x_train[i].mean(axis=0)
            x_stds = x_train[i].std(axis=0)
            x_train[i] = (x_train[i] - x_mean)/x_stds
            x_test[i] = (x_test[i] - x_mean)/x_stds
            print(torch.sum(torch.isnan(x_train[i])))
            print(torch.sum(torch.isnan(x_test[i])))
            
    y_mean = np.nanmean(y_train, axis=0)
    y_stds = np.nanstd(y_train, axis=0)
    y_train = (y_train - y_mean)/y_stds    
    
    print('Done...')
    
    
    train_dataset = load_Multimodal_data(X = x_train, y = y_train)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)           
    test_dataset = load_Multimodal_data(X = x_test, y = y_test)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False)
    
    if init_method == 'random':
        model = SupervisedFLICAmodel(nfea = nfea,nlat = nlat, ntask = ntask, dropout = dropout, 
                             device = device, init_spatial_loading = None,
                            init_weight_pred=None, init_bais_pred=None).to(device)   
    else:
        
        print('Multimodal MIGP initialization...')
        init_spatial_loading,init_weight_pred,init_bais_pred,uu = Initialize_SuperBigFLICA(x_train,y_train,nlat, train_ind)
        print('Done...')
    
        model = SupervisedFLICAmodel(nfea = nfea,nlat = nlat, ntask = ntask, dropout = dropout, 
                                 device = device, init_spatial_loading = init_spatial_loading,
                                init_weight_pred=init_weight_pred, init_bais_pred=init_bais_pred).to(device)    
        
            
    loss_fun_reg = loss_SuperBigFLICA_regression(ntask = ntask, nmod = nmod, device = device,nsub = x_train[0].shape[0],
                                              auto_weight = auto_weight, lambdas = lambdas).to(device)

    
    y_mean1 = torch.FloatTensor(y_mean).to(device)
    y_stds1 = torch.FloatTensor(y_stds).to(device)

    optimizer1 = torch.optim.Adam(loss_fun_reg.parameters(), lr)      
    optimizer = torch.optim.RMSprop(model.parameters(), lr, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maxiter)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=maxiter)
    
#########MRICA main iteration################################################################################            

    epochs=maxiter
    loss_all_train=[]
    loss_all_test=np.zeros((maxiter,4))
    best_corr = -1
    #best_mae = 10000
    
    for epoch in range(0, epochs + 1):
                
        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)
        tt = time.time()
        
        
        model.train()
        train_loss = 0
        train_MAE=0
        #acc_train = 0
        for batch_idx, (data_batch, labels) in enumerate(train_loader):
            
            for i in range(0,nmod):
                data_batch[i] = data_batch[i].to(device)
            labels = labels.to(device)                
            
            if epoch>=0 and epoch<10:
                optimizer.zero_grad()
            elif epoch>=10:
                optimizer.zero_grad()
                optimizer1.zero_grad()
                
            recon_train, spatial_loadings, lat_train, pred_all, weight_pred = model(x = data_batch, device=device)
            
            loss,_,_,_,_ = loss_fun_reg(recon_train,data_batch,spatial_loadings, pred_all, labels, weight_pred, lat_train)
            
            if epoch>=0 and epoch<10:
                loss.backward()
                optimizer.step()
            elif epoch>=10:
                loss.backward()
                optimizer.step()                
                optimizer1.step() 
                
        
            train_loss += loss.item()
            
            pred_all1 = (pred_all * y_stds1) + y_mean1
            labels1 = (labels * y_stds1) + y_mean1
            #print(pred_all1.shape)
            #print(labels.shape)
            
            train_MAE += torch.sum(torch.abs(pred_all1-labels1))/pred_all.size()[1]
            
        train_MAE = train_MAE/y_train.shape[0]
        
        scheduler.step()
        scheduler1.step()       
        
        loss_all_train.append(train_loss)
        
        
        model.eval()    
        
        test_l1 = 0
        test_l2 = 0
        test_l3 = 0
        test_l4 = 0
        
        with torch.no_grad():

            test_loss = 0
            for batch_idx, (data_batch, labels) in enumerate(test_loader):

                for i in range(0,nmod):
                    data_batch[i] = data_batch[i].to(device)
                labels = labels.to(device)                

                recon_train, spatial_loadings, lat_test, pred_all, weight_pred = model(x = data_batch, device=device)

                loss,l1,l2,l3,l4 = loss_fun_reg(recon_train,data_batch,spatial_loadings, pred_all, labels, weight_pred, lat_train)

                test_loss += loss
                test_l1 += l1
                test_l2 += l2
                test_l3 += l3
                test_l4 += l4

                #get test prediction and test latent variable

                if batch_idx==0:
                    pred_all1 = (pred_all * y_stds1) + y_mean1
                    pred_test = torch.Tensor.cpu(pred_all1).detach().numpy()
                    lat_test_all = torch.Tensor.cpu(lat_test).detach().numpy()
                else:
                    pred_all1 = (pred_all * y_stds1) + y_mean1
                    pred_test = np.vstack((pred_test,torch.Tensor.cpu(pred_all1).detach().numpy()))                            
                    lat_test_all = np.vstack((lat_test_all,torch.Tensor.cpu(lat_test).detach().numpy()))  

            #compute evaluation metric
            idx_nonnan = np.isnan(y_test)==0
            corr_test1 = np.zeros((y_test.shape[1],))
            for ij in range(0,y_test.shape[1]):
                if np.sum(idx_nonnan[:,ij])>=10:
                    corr_test1[ij] = scipy.stats.pearsonr(pred_test[idx_nonnan[:,ij],ij],y_test[idx_nonnan[:,ij],ij],)[0]
            if y_test.shape[1]>1:
                corr_test = np.nansum(corr_test1[corr_test1>0.1])
            else:
                corr_test = np.nanmean(corr_test1)
            test_MAE = np.mean(np.abs(pred_test[idx_nonnan].flatten()-y_test[idx_nonnan].flatten()))
            if epoch>=1:
                if corr_test>best_corr:
                #if test_MAE < best_mae:
                    #best_mae = test_MAE * 1.0
                    best_corr = corr_test * 1.0
                    pred_best = pred_test*1.0
                    best_model = copy.deepcopy(model)

            #loss_all_test.append(test_loss)
            loss_all_test[epoch-1,0] = torch.Tensor.cpu(test_l1).detach().numpy()
            loss_all_test[epoch-1,1] = torch.Tensor.cpu(test_l2).detach().numpy()
            loss_all_test[epoch-1,2] = torch.Tensor.cpu(test_l3).detach().numpy()
            loss_all_test[epoch-1,3] = torch.Tensor.cpu(test_l4).detach().numpy()
            
        if y_test.shape[1]>1:
            print('====> Epoch: {} train loss: {:.6f}, test loss: {:.4f}, test MAE: {:.4f}, test Sum of Corr with r>0.1: {:.4f}'
                      .format(epoch, train_loss / len(train_loader.dataset),test_loss/ len(test_loader.dataset), test_MAE, corr_test ))                
                     
        else:
            print('====> Epoch: {} train loss: {:.6f}, test loss: {:.4f}, test MAE: {:.4f}, test Corr: {:.4f}'
                      .format(epoch, train_loss / len(train_loader.dataset),test_loss/ len(test_loader.dataset), test_MAE, corr_test ))                
             
        print('====> Epoch: {} time: {:.3f} s'
                      .format(epoch, time.time() - tt ))     

    best_model.to('cpu')
    last_model = model.to('cpu')
    
    return pred_best, best_model, loss_all_test, best_corr, last_model



def get_model_param(x_train, x_test, y_train, best_model, get_sp_load = 1):
    nmod = len(x_train)
    best_model.eval()        

    modality_weights = F.softmax(torch.Tensor.cpu(best_model.mod_weight),dim=1).detach().numpy()
    
    prediction_weights = torch.Tensor.cpu(best_model.weight_pred).detach().numpy()

    print('Data Normalization...')
    is_norm = True
    if is_norm is True:
        for i in range(0,len(x_train)):
            x_mean = x_train[i].mean(axis=0)
            x_stds = x_train[i].std(axis=0)
            x_train[i] = (x_train[i] - x_mean)/x_stds
            x_test[i] = (x_test[i] - x_mean)/x_stds
            print(torch.sum(torch.isnan(x_train[i])))
            print(torch.sum(torch.isnan(x_test[i])))
            
    y_mean = np.nanmean(y_train, axis=0)
    y_stds = np.nanstd(y_train, axis=0)
    y_train = (y_train - y_mean)/y_stds    
    
    print('Done...')
    
    _,_,lat_train,pred_train,_  = best_model(x = x_train,device='cpu') 
    _,_,lat_test,pred_test,_  = best_model(x = x_test,device='cpu')    

    lat_train = torch.Tensor.cpu(lat_train).detach().numpy()
    lat_test = torch.Tensor.cpu(lat_test).detach().numpy()
    
    pred_train = pred_train.detach().numpy() * y_stds + y_mean
    pred_test = pred_test.detach().numpy() * y_stds + y_mean
    
    # pred_train = torch.Tensor.cpu(pred_train).detach().numpy()
    # pred_test = torch.Tensor.cpu(pred_test).detach().numpy()
    
    spatial_loadings=[]
    if get_sp_load == 1:
        for i in range(0,nmod):
            # spatial_loadings.append(torch.Tensor.cpu(best_model.spatial_loading[i]).detach().numpy())
            spatial_loadings.append( sKPCR_regression(lat_train, x_train[0].numpy(), np.ones((lat_train.shape[0],1))))
                
    return lat_train,lat_test, spatial_loadings, modality_weights, prediction_weights, pred_train, pred_test

        
def sKPCR_regression(X,Y,cov):

    contrast=np.transpose(np.hstack(  ( np.eye(X.shape[1],X.shape[1]) , np.zeros((X.shape[1],cov.shape[1])) ))   )
    contrast=np.array(contrast,dtype='float32')
    
    design=np.hstack((X,cov))
#     print(design)
#     ss = np.linalg.pinv(design).T
    #degree of freedom
    df=design.shape[0]-design.shape[1]
    #
    ss=np.linalg.inv(np.dot(np.transpose(design),design))

    beta=np.dot(np.dot(ss,np.transpose(design)),Y)

    Res=Y-np.dot(design,beta)

    sigma=np.reshape(np.sqrt(np.divide(np.sum(np.square(Res),axis=0),df)),(1,beta.shape[1]))

    tmp1=np.dot(beta.T,contrast)
    tmp2=np.array(np.diag(np.dot(np.dot(contrast.T,ss),contrast)),ndmin=2)

    Tstat=np.divide(tmp1,np.dot(sigma.T,np.sqrt(tmp2)  ))


    return Tstat

def get_res(X,Y):

    X= np.hstack((X, np.ones((X.shape[0], 1))))
    
    beta=np.dot(np.linalg.pinv(X),Y)

    Res=Y-np.dot(X,beta)

    return Res

def nets_zscore(x):
    # x : a nsubject * nfeature numpy matrix
    
    x_zscore=(x-x.mean(axis=0))
    stds=x.std(axis=0)
    index=stds==0
       
    if sum(index)>0:
        stds[index]=0.1
        print('Warning: '+str(sum(index))+' of the features are all zero or constants')
        print('Normalizing them to all zeros ...')
    
    x_zscore=x_zscore/stds
    
    return x_zscore
