import torch
import pandas as pd

import tqdm
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import cdist


def euclidean(x, classes_mean):
    """
    Compute euclidean distance between input features vs class conditioned means.
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = classes_mean.size(0)
    d = x.size(1)
    if d != classes_mean.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    classes_mean = classes_mean.unsqueeze(0).expand(n, m, d)
    x_mu = x - classes_mean
    # return torch.pow(x - support_mean, 2).sum(2)
    return (x_mu**2).sum(2)


def mahalanobis(x, classes_mean, inv_covmat):
    """
    Compute mahalanobis distance between input features vs class conditioned means.
    """

    # create function to calculate Mahalanobis distance
    n = x.size(0)
    d = x.size(1)

    maha_dists = []
    for class_inv_cov, class_mean in zip(inv_covmat, classes_mean):
        x_mu = x - class_mean.unsqueeze(0).expand(n, d)

        mahal = torch.einsum("ij,jk,ki->i", x_mu, class_inv_cov, x_mu.T)
        maha_dists.append(torch.sqrt(mahal))

    return torch.stack(maha_dists, dim=1)


def kernel_dist(x, classes_mean, metric="poly"):
    """
    Compute kernel distance between input features vs class conditioned means.
    """
    from sklearn.metrics.pairwise import pairwise_kernels

    kernel_sim = []
    for class_mean in classes_mean:
        # Note negative sign
        kernel_sim.append(
            -pairwise_kernels(x, np.expand_dims(class_mean, axis=0), metric)
        )
    return torch.from_numpy(np.asarray(kernel_sim).squeeze(axis=2).T)

def normalize_features(feat_tr, feat_te,x):
    means = feat_tr.mean(dim=0)
    feat_normed_tr = feat_tr - means
    feat_normed_te = feat_te - means
    x = x - means
    return feat_normed_tr, feat_normed_te, x

def joint_oc(x):

    """
    Compute the ood scores of the joint OC trained models
    """
    scores = x.pow(2).mean(1).detach()
    #scores = torch.cat([handle(x), handle(features_te)])

    return scores

def uncertainty_certificates(x,features_tr,
                             features_te,
                             hypers,
                             certs_bias=0,
                             get_all=False,
                             mask = None
                             ):

    """
    Compute uncertainty using linear certificates (ours)
    """
    normed_features = hypers["normed"]
    certs_epochs = hypers["certs_epochs"]
    certs_k = hypers['certs_k']
    certs_reg = hypers['certs_reg']
    certs_loss = hypers['certs_loss']
    if mask != None:
        features_tr_all = features_tr.clone()
        features_tr = features_tr[mask]
    else:
        features_tr_all = features_tr
    def target(xx):
        return torch.zeros(xx.size(0), certs_k)
    features_tr, features_te, x= normalize_features(features_tr, features_te, x) if normed_features else (features_tr, features_te, x)
    certificates = torch.nn.Linear(features_tr.size(1), certs_k, bias=certs_bias)
    opt = torch.optim.Adam(certificates.parameters())
    sig = torch.nn.Sigmoid()

    if certs_loss == "bce":
        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    elif certs_loss == "mse":
        loss = torch.nn.L1Loss(reduction="none")


    train_loader, val_loader = torch.utils.data.random_split(TensorDataset(features_tr, features_tr), [0.9, 0.1]) 
    loader_f = DataLoader(train_loader,
                          batch_size=64,
                          shuffle=True)
    loader_val = DataLoader(val_loader, batch_size = len(val_loader))
    pbar = tqdm.tqdm(range(certs_epochs),desc="Loss: ", leave=True)
    prev = 10000
    tol = 0
    for epoch in pbar:
    #for epoch in range(certs_epochs):

        
        for f, _ in loader_f:
            opt.zero_grad()
            error = loss(certificates(f), target(f)).mean()
            penalty = certs_reg * \
                (certificates.weight @ certificates.weight.t() - 
                 torch.eye(certs_k)).pow(2).mean()
            (error + penalty).backward()
            opt.step()
        
        val_error = np.array([loss(certificates(f_val), target(f_val)).detach().numpy().mean() for f_val, _ in loader_val ]).mean()
        pbar.set_description("Loss: %s, val_loss: %s, tol: %s" % (str(error + penalty), str(val_error), str(tol)))
        pbar.refresh()

        if val_error > prev:
            tol += 1
            if tol == 5:
                break
        else:
            prev = val_error
            tol = 0

        
    def handle(features):
        output = certificates(features)
        if certs_loss == "bce":
            return sig(output).pow(2).mean(1).detach()
        if get_all == True:
            return output
        else:
            return output.pow(2).mean(1).detach()
    if get_all == True:
        return torch.cat([handle(x),handle(features_te), handle(features_tr_all)])
    scores = torch.cat([handle(x), handle(features_te)])
    

    return scores
    
def scipy_dist(x, classes_mean, metric=None, **kwargs):
    distances = []
    for class_mean in classes_mean:
        distances.append(
            cdist(x, np.expand_dims(class_mean, axis=0), metric=metric, **kwargs)
        )

    return torch.from_numpy(np.asarray(distances).squeeze(axis=2).T)


def seuclidean(x, classes_mean, class_variances):
    """
    Compute standardised euclidean distance between input features vs class conditioned means.
    """
    distances = []
    for support_class, class_variance in zip(classes_mean, class_variances):
        distances.append(
            cdist(
                x,
                np.expand_dims(support_class, axis=0),
                metric="seuclidean",
                V=class_variance,
            )
        )
    return torch.from_numpy(np.asarray(distances).squeeze(axis=2).T)


def cosine(x, classes_mean, reg_factor=1):
    """calculate pairwise cosine similarity between input features and classes means"""
    x = x.detach().cpu().numpy()
    cosine_sim = []
    if reg_factor == 1.0:
        for class_mean in classes_mean:
            cosine_sim.append(
                -cosine_similarity(x, class_mean.unsqueeze(0).cpu().numpy())
            )
        return torch.from_numpy(np.asarray(cosine_sim).squeeze(axis=2).T)
    else:
        for sample in x:
            sample_dist = []
            for class_mean in classes_mean:
                sample_dist.append(
                    np.dot(sample, class_mean)
                    / (
                        reg_factor
                        * ((np.linalg.norm(sample) * np.linalg.norm(class_mean)))
                    )
                )
            cosine_sim.append(sample_dist)

        return torch.FloatTensor(cosine_sim)

def uncertainty_certificates_cw(x, features_tr,
                             features_te, classes_feats,
                             hypers,certs_bias=0,
                            ):

    """
    Compute uncertainty using linear certificates (ours) but class-wise
    """
    normed_features = hypers["normed"]
    certs_epochs = hypers["certs_epochs"]
    certs_k = hypers['certs_k']
    certs_reg = hypers['certs_reg']
    certs_loss = hypers['certs_loss']
    def target(xx):
        return torch.zeros(xx.size(0), certs_k)
    def handle(features):
        output = certificates(features)
        if certs_loss == "bce":
            return sig(output).pow(2).mean(1).detach()
        else:
            return output.pow(2).mean(1).detach()
    scores = []
    for sub_class in classes_feats:  
        sub_class = torch.Tensor(sub_class)
    #features_tr, features_te, x= normalize_features(features_tr, features_te, x) if normed_features else (features_tr, features_te, x)
        certificates = torch.nn.Linear(sub_class.size(1), certs_k, bias=certs_bias)
        opt = torch.optim.Adam(certificates.parameters())
        sig = torch.nn.Sigmoid()

        if certs_loss == "bce":
            loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        elif certs_loss == "mse":
            loss = torch.nn.L1Loss(reduction="none")

        loader_f = DataLoader(TensorDataset(sub_class, sub_class),
                            batch_size=64,
                            shuffle=True)
        pbar = tqdm.tqdm(range(certs_epochs),desc="Loss: ", leave=True)
        for epoch in pbar:
            for f, _ in loader_f:
                opt.zero_grad()
                error = loss(certificates(f), target(f)).mean()
                penalty = certs_reg * \
                    (certificates.weight @ certificates.weight.t() - 
                    torch.eye(certs_k)).pow(2).mean()
                (error + penalty).backward()
                opt.step()
                pbar.set_description("Loss: %s" % str(error + penalty))
                pbar.refresh()
        scores.append(torch.cat([handle(x), handle(features_te)]))

    return torch.stack(scores)
 
def min_var_feature_deviations(x, features_tr, features_te, hypers):
    pp = hypers['pp']
    epsilon = hypers['epsilon']
    features_val = features_tr[0:int(features_tr.size()[1]*0.1)]
    features_tr = features_tr[int(features_tr.size()[1]*0.1):]
    tr_mu = torch.mean(features_tr, dim = 0)
    val_mu = torch.mean(features_val, dim = 0)
    tr_mu = torch.mean(features_tr, dim = 0)
    val_mu = torch.mean(features_val, dim = 0)
    diff = torch.abs(tr_mu - val_mu)
    onoff = (diff <= torch.quantile(diff, 0.5))
    #print(onoff)
    features_tr = features_tr.T[onoff].T
    features_te = features_te.T[onoff].T
    x = x.T[onoff].T
    train_stds = torch.std(features_tr, dim = 0)
    #train_means = np.mean(features_tr, axis = 0)


    ups = torch.quantile(features_tr, 0.025, dim=0)
    downs = torch.quantile(features_tr, 0.975, dim = 0)
    std_perc = torch.quantile(train_stds, pp)
    #train_stds[train_stds < std_perc] = 1000
    ups[train_stds > std_perc] = -float("inf")
    downs[train_stds > std_perc] = float("inf")
    #uncert = np.sum(np.abs(test_f_2 - train_means) > train_stds*2, axis = 1)
    uncert_te = torch.sum((features_te < (ups - epsilon)) + (features_te > (downs + epsilon)), dim=1)
    uncert_tr = torch.sum((x < (ups - epsilon)) + (x > (downs + epsilon)), dim=1)
    uncert = torch.cat([uncert_tr, uncert_te])
    my_scores = (uncert - uncert.min())/(uncert.max() - uncert.min())
    
    return uncert if my_scores.isnan().sum() >= 1 else my_scores

def MVFD(x, classes_variances, classes_feats, hypers):
    """
    Compute MVFD class wise and take minimum eventaully
    X shape - N x D
    class_var shape - M x D
    class_feats shape - M x N_1 + ... + N_m
    """
    pp = hypers['pp']
    epsilon = hypers['epsilon']
    n = x.size(0) 
    m = classes_variances.size(0)
    d = x.size(1)
    if d != classes_variances.size(1):
        raise Exception
   # print(n, m, d)


    x = x.unsqueeze(1).expand(n, m, d) #NxD -> N x M x D
       # ups shape - N x (M x D)
    ups = torch.stack([torch.quantile(torch.Tensor(class_feats), 0.1, dim=0) for class_feats in classes_feats ]).unsqueeze(0).expand(n,m,d)
    downs = torch.stack([torch.quantile(torch.Tensor(class_feats), 0.9, dim=0) for class_feats in classes_feats ]).unsqueeze(0).expand(n,m,d)
    # std_perc shape - (M) x D
    std_perc = torch.quantile(classes_variances, pp, dim = 1).unsqueeze(1).expand(m,d)
    classes_variances = classes_variances.unsqueeze(0).expand(n, m, d) #N x (M x D)

    ups[classes_variances > std_perc] = -float("inf")
    downs[classes_variances > std_perc] = float("inf")
    # uncert shape - N x M
    uncert = torch.sum((x < (ups - epsilon)) + (x > (downs + epsilon)), dim=2)

    return uncert 

def get_class_variances(classes_feats):
    return torch.stack([torch.var(classes_feat, dim=0) for classes_feat in classes_feats])

def get_inv_covmat(classes_feats, classes_mean):
    assert len(classes_feats) == len(classes_mean)
    inv_covmat = []
    for i in range(len(classes_feats)):
        cov_mat = torch.cov((classes_feats[i] - classes_mean[i]).T ) 
        inv_covmat.append(torch.linalg.inv(cov_mat + np.eye(cov_mat.size(0))*0.001))
    return inv_covmat


def get_classes_mean(classes_feats):
    return torch.stack(
        [torch.mean(cls_feats, dim=0) for cls_feats in classes_feats], dim=0
    )


def get_distances(x, distance, ind_classes_feats, ind_classes_mean, reg_factor=1,
 hypers = None, special_feat = [], layer = "layer_x"):
    """Class conditioned distances. Return [n x c] matrix where n is the number of input features and c the number of classes in ind_classes_feats
    certs_epochs = {10,100}
    certs_reg = {0,1,10}
    certs_k = {100,1000} 
    ind_classes_mean = either this or ind_outputs_te
    ind_classes_feats = either this or ind_outputs_tr
    """

    if distance == "msp":
        # sfmx is already calculated...
        distances = x["softmax"]
    elif distance == "energy":
        return torch.logsumexp(x[layer],dim = 1) 
    
    elif distance == "simple_oc": #return directly the uncert approx.
        features_tr = ind_classes_feats[layer]
        features_te = x[layer] #ood
        ind_outputs = ind_classes_mean[layer]
        distances = uncertainty_certificates(ind_outputs, features_tr, features_te, hypers)
        return distances
    
    elif distance == "simple_oc_OE": #return directly the uncert approx.
        ood_outputs = x[layer]
        ind_outputs_te = ind_classes_mean[layer]
        train_outp = ind_classes_feats
        thres = hypers["threshold"]
        if thres == -1:
            miscl_tr = (torch.max(train_outp["softmax"],dim=1).values >= 0.5)* (torch.argmax(train_outp["softmax"],dim=1) != train_outp["classes"])
        else:
            miscl_tr = (torch.max(train_outp["softmax"],dim=1).values <= thres)* (torch.argmax(train_outp["softmax"],dim=1) != train_outp["classes"])
        if sum(miscl_tr) == 0:
            print("no low conf miscl exists")
            thres_buffer = 0.5
            while sum(miscl_tr) == 0:
                miscl_tr = (torch.max(train_outp["softmax"],dim=1).values >= thres_buffer)* (torch.argmax(train_outp["softmax"],dim=1) != train_outp["classes"])           
                thres_buffer -= 0.1
        
        ind_outputs_tr_miscl = train_outp["layer_x"][miscl_tr]

        cl_tr = (torch.max(train_outp["softmax"],dim=1).values >= 0.0)* (torch.argmax(train_outp["softmax"],dim=1) == train_outp["classes"])
        ind_outputs_tr_cl = train_outp["layer_x"][cl_tr]
        
               
        distances_cl = uncertainty_certificates( ind_outputs_te, ind_outputs_tr_cl,ood_outputs, hypers)
        distances_miscl = 1 /(1+uncertainty_certificates( ind_outputs_te, ind_outputs_tr_miscl,ood_outputs, hypers))


        return distances_cl + distances_miscl


    elif distance == "min_var_feature_deviations": #returns directly the uncert approx.
        features_tr = ind_classes_feats[layer]
        features_te = ind_classes_mean[layer]
        return min_var_feature_deviations(x[layer], features_tr, features_te, hypers)
        
    elif distance == "joint_oc":
        return joint_oc(x)

    elif distance == "mahalanobis_copula":
        inv_covmat = get_inv_covmat(ind_classes_feats, ind_classes_mean)
        distances = mahalanobis(x, ind_classes_mean, inv_covmat)

    elif distance == "mahalanobis":
        inv_covmat = get_inv_covmat(ind_classes_feats, ind_classes_mean)
        distances = mahalanobis(x[layer], ind_classes_mean, inv_covmat)

    elif distance == "MVFD":
        variance = get_class_variances(ind_classes_feats)
        distances = MVFD(x[layer],get_class_variances(ind_classes_feats), ind_classes_feats, hypers)
    elif distance == "oc_cw":
        y_ood = torch.argmax(x["softmax"], dim = 1).detach().numpy()
        y_test = torch.argmax(ind_classes_mean["softmax"], dim = 1).detach().numpy()

        features_tr = ind_classes_feats[layer]
        features_te = ind_classes_mean[layer]
        ind_classes_feats = special_feat
        distances = uncertainty_certificates_cw(x[layer], features_tr, features_te, ind_classes_feats, hypers)

        if hypers["by_pred"]:
            print(y_ood[0:100],len(y_ood), max(y_ood), len(distances) )
            distances_ood = distances.T[np.arange(0,len(y_ood)),y_ood]
            distances_ind =  distances.T[len(y_ood) + np.arange(0,len(y_test)),y_test]
            return torch.concat([distances_ood,distances_ind])    
                   
        return torch.min(distances, dim=0)[0]
    elif distance in {"linear", "poly", "rbf", "sigmoid"}:
        distances = kernel_dist(x[layer], ind_classes_mean, metric=distance)

    elif distance in {
        "rseuclidean",
        "seuclidean",
        "cosine",
        "correlation",
        "chebyshev",
        "braycurtis",
        "euclidean",
    }:
        x = x[layer]
        if distance == "euclidean":
            distances = euclidean(x, ind_classes_mean)
        if distance == "seuclidean":
            variance = get_class_variances(variance)
            distances = seuclidean(x, ind_classes_mean, variance)
        if distance == "cosine":
            distances = cosine(x, ind_classes_mean, reg_factor)
        else:
            distances = scipy_dist(x, ind_classes_mean, metric=distance)
    else:
        raise ValueError(f"{distance} is not supported")
        print(torch.min(distances,dim =1)[0])
    return (
        torch.max(distances, dim=1)[0]
        if distance == "msp"
        else torch.min(distances, dim=1)[0]
    )
