import argparse
import os
import itertools
import pandas as pd
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import src.dataloader
import src.distance_metrics
#from models.crosr import *
#from models.flexmlp import *
#from models.laddernet import *
import datetime

def get_model_outputs(data_loader, model, layer_num=-2, logits_pos = -1):
    classes = []
    logits = []
    layer_x = []
    for batch in data_loader:
        x, y = batch
        x = x.float()
        y = y.long()
        classes += y.squeeze(dim=0)
        with torch.no_grad():
            forward_layer_output = model(x)
            logits += forward_layer_output[logits_pos]
            # layer_num=-1 is pre-logits, layer_num=0 is the first layer
            layer_x += forward_layer_output[layer_num]
    output = {
        "softmax": torch.nn.functional.softmax(torch.stack(logits), dim=1),
        "layer_x": torch.stack(layer_x),
        "classes": torch.stack(classes),
    }

    return output   


def get_accuracy(softmax, classes):
    preds = torch.argmax(softmax, dim=1)
    return (torch.sum(torch.eq(preds, classes)) / len(classes)).detach().cpu().item()


def get_class_features(features, classes):
    class_features = []
    for i in torch.unique(classes):
        idx = torch.where(classes == i)[0]
        class_features.append(features[idx])
    return class_features


def get_class_means(features, classes):
    class_means = []
    for i in torch.unique(classes):
        idx = torch.where(classes == i)[0]
        class_means.append(torch.mean(features[idx], dim=0))
    return torch.stack(class_means)

def get_model_outputs_kickout(data_loader, model,activation,  kick = 640 ):
    """ Kicks out neurons one by one and measures its performance without the efect of each neuron.
    """
    accs = []
    model.eval()
    losses = []
    for i, batch in tqdm.tqdm(enumerate(data_loader)):
        classes = []

        _, y = batch
        y = y.long()
        if len(y.squeeze(dim=0)) != 512:
            
            accs = np.array(accs).reshape(-1,kick).mean(axis=0)
            losses = np.array(losses).reshape(-1,kick).mean(axis=0)
            return accs, losses
 
        classes += y.squeeze(dim=0)
        for j in range(kick):
            logits = []
            x = torch.Tensor(activation["prelogit"][i]).clone()
            with torch.no_grad():

                x[:, j] = 0
                kicked_logits = model.linear(x).detach().clone()

                logits += kicked_logits
            output = {
                "softmax": F.log_softmax(torch.stack(logits), dim=1),
#         "layer_x": torch.stack(layer_x),
                "classes": torch.stack(classes),
                "loss":F.cross_entropy(torch.stack(logits), torch.stack(classes))
                    }

            accuracy = sum(torch.argmax(output['softmax'], dim=1) == output['classes']) / len(output['classes'])
            accs.append(accuracy)
            losses.append(output["loss"])

    accs = np.array(accs).reshape(-1,kick).mean(axis=0)
    losses = np.array(losses).reshape(-1,kick).mean(axis=0)

    return accs, losses 

def get_model_outputs_shap(data_loader, model,activation,  kick = 640):
    """ Calculates the shaply values for prelogit neurons
    """
    model.eval()
    batch_shaps = []
    classes = []
    for i, batch in tqdm.tqdm(enumerate(data_loader)):

        _, y = batch
        y = y.long()
        if len(y.squeeze(dim=0)) != 512:
            output = [] 
            for cl in range(10):
                for b,c in zip(batch_shaps,classes):
                    output.append(b[c==cl].sum(axis=0))

            output = np.array(output).reshape(10,-1,640).sum(axis=1)/(len(classes)*512)
            return output

        classes.append(y.squeeze(dim=0).detach().numpy())
        shapies = np.zeros((512,640))
        for j in range(kick):
            x = torch.Tensor(activation["prelogit"][i]).clone()
            x_orig = torch.Tensor(activation["prelogit"][i]).clone()


            with torch.no_grad():

                x[:, j] = 0
                kicked_logits = model.linear(x).detach().clone()
                logits = model.linear(x_orig).detach().clone()
                shaps = torch.sum((logits - kicked_logits)**2, dim = 1)
                shapies[:,j] = shaps.detach().numpy()

        batch_shaps.append(shapies)



    return None



def get_model_outputs_one(data_loader,acts, model, kick = -1, kick_true = True, cw = False):
    """ A function to kick off different activation neurons. 
        kick : an index or array of indexes to kick out (if cw = True, then needs to have a shape (nr of classes, desired nr of indexes))
        keep kick_true = False to obtain no kicking and normal performance
    """

    classes = []
    logits = []
    xes = []
    model.eval()
    for i, batch in enumerate(data_loader):
        _, y = batch
        y = y.long()
        x = acts["prelogit"][i].copy()

        if len(y.squeeze(dim=0)) != 512:
            return {
        "softmax": F.log_softmax(torch.stack(logits), dim = 1),
        "layer_x":torch.stack(logits),
        "classes": torch.stack(classes),
        "prelogit": torch.stack(xes
        )}
        classes += y.squeeze(dim = 0)
        with torch.no_grad():
            if kick_true and not cw:
                x[:, kick] = 0
            elif cw == True:
                for i in torch.unique(y):
                    idx = torch.where(y.squeeze(dim=0) == i)[0]
                    x[idx.reshape((-1,1)),kick[i]] = 0
   
            

            xes += torch.Tensor(x)
            kicked_logits = model.linear(torch.Tensor(x)).detach().clone()
            logits += kicked_logits
    output = {
        "softmax": F.log_softmax(torch.stack(logits), dim = 1),
        "layer_x": torch.stack(logits),
        "classes": torch.stack(classes),
        "prelogit":torch.stack(xes)
    }

    return output
def get_all_outputs(model,
    kick_num,
    kick_true,
    ind_train_dataloader,
    ind_test_dataloader,
    ood_test_dataloader,
    ind_tr_acts,
    ind_te_acts,
    ood_te_acts,
    layer = "layer_x"
): #äkki ka seda et ood ja test ei lükka välja hoopis
    if ind_te_acts == None:
        logits_pos = kick_num + 1
        layer_num = kick_num
        return get_all_outputs_basic(model, layer_num,logits_pos, ind_train_dataloader, ind_test_dataloader, ood_test_dataloader, layer )
    ind_outputs = get_model_outputs_one(ind_train_dataloader, ind_tr_acts, model, kick_num, kick_true)
    ind_classes_feats = get_class_features(ind_outputs[layer], ind_outputs["classes"])

    ind_classes_mean = get_class_means(ind_outputs[layer], ind_outputs["classes"])
    features_tr = ind_outputs[layer].clone()
    ind_outputs = get_model_outputs_one(ind_test_dataloader, ind_te_acts, model, kick_num, kick_true,False)
    # Validation accuracy
    ind_accuracy = get_accuracy(ind_outputs["softmax"], ind_outputs["classes"])
    ood_outputs = get_model_outputs_one(ood_test_dataloader, ood_te_acts, model, kick_num, kick_true,False)


    return features_tr, ind_outputs, ood_outputs, ind_classes_feats, ind_classes_mean, ind_accuracy

def get_all_outputs_basic(model,
    layer_num,
    logits_pos,
    ind_train_dataloader,
    ind_test_dataloader,
    ood_test_dataloader,
    layer = "layer_x"
):
    ind_outputs = get_model_outputs(ind_train_dataloader, model, layer_num, logits_pos)
    ind_classes_feats = get_class_features(ind_outputs[layer], ind_outputs["classes"])
    ind_classes_mean = get_class_means(ind_outputs[layer], ind_outputs["classes"])
    ind_outputs_tr = ind_outputs.copy()
    ind_outputs_te = get_model_outputs(ind_test_dataloader, model,layer_num,logits_pos)
    # Validation accuracy
    ind_accuracy = get_accuracy(ind_outputs_te["softmax"], ind_outputs_te["classes"])
    ood_outputs = get_model_outputs(ood_test_dataloader, model, layer_num, logits_pos)

    return ind_outputs_tr, ind_outputs_te, ood_outputs, ind_classes_feats, ind_classes_mean, ind_accuracy

def get_all_outputs_without_model(
    ind_tr_acts,
    ind_te_acts,
    ood_acts,
    ind_tr_classes
):
    ind_outputs = ind_tr_acts
    ind_classes_feats = get_class_features(ind_outputs, ind_tr_classes)
    ind_classes_mean = get_class_means(ind_outputs, ind_tr_classes)
    ind_outputs_tr = ind_outputs
    ind_outputs_te = ind_te_acts
    # Validation accuracy
    ind_accuracy = 0
    ood_outputs = ood_acts

    return ind_outputs_tr, ind_outputs_te, ood_outputs, ind_classes_feats, ind_classes_mean, ind_accuracy



def ood_experiment(
    all_outputs,
    ind_dataset_name,
    ood_dataset_name,
    distance,
    layer="layer_x",
    hypers = None,
):
    ind_outputs_tr, ind_outputs_te, ood_outputs, ind_classes_feats, ind_classes_mean, ind_accuracy = all_outputs
    
    no_no_distances = ["simple_oc", "min_var_feature_deviations", "oc_cw", "simple_oc_OE"]
    if distance not in no_no_distances: 
        ind_distances = distance_metrics.get_distances(
            ind_outputs_te,
            distance=distance,
            ind_classes_feats=ind_classes_feats,
            ind_classes_mean=ind_classes_mean,
            hypers = hypers,
        )

    ood_distances = distance_metrics.get_distances(
            #ood_outputs["softmax"] if distance == "msp" else ood_outputs[layer],
            ood_outputs,
            distance=distance,
            ind_classes_feats = ind_outputs_tr if distance in no_no_distances else ind_classes_feats,
            ind_classes_mean = ind_outputs_te if distance in no_no_distances else ind_classes_mean,
            hypers = hypers,
            special_feat=ind_classes_feats,
            layer=layer
        )
    total_distances = ood_distances.cpu().numpy() if distance in no_no_distances else  torch.cat([ind_distances, ood_distances]).cpu().numpy()
    
    if distance in ["msp","mahalanobis_copula"]:
        classes = [1] * len(ind_distances) + [0] * len(ood_distances)
    elif distance in no_no_distances:
        classes = [0] * len(ind_outputs_te[layer]) + [1] * len(ood_outputs[layer])
        total_distances = total_distances if sum(total_distances) == 0 else (total_distances - total_distances.min()) / (total_distances.max() - total_distances.min())
    
    else:
        classes = [0] * len(ind_distances) + [1] * len(ood_distances)
    #print(len(classes), len(total_distances), ind_accuracy)
    auc = metrics.roc_auc_score(classes, total_distances)
    #print(auc)

    return {
        "ind": ind_dataset_name,
        "ood": ood_dataset_name,
        "distance": distance,
        "id_accuracy": ind_accuracy,
        "auc": auc,
    }, classes, total_distances
    

def get_ind_and_ood_dataloaders(opt, ood_dataset):
    ind_train_dataloader = DataLoader(
        dataloader.get_dataset(opt["ind_dataset"], opt["data_dir"], True,imgsz = opt["imgsz"]),
        batch_size=opt["batch_size"],
        shuffle=False,
        num_workers=opt["num_workers"],
        
    )

    ind_test_dataloader = DataLoader(
        dataloader.get_dataset(opt["ind_dataset"], opt["data_dir"], False,imgsz = opt["imgsz"]),
        batch_size=opt["batch_size"],
        shuffle=False,
        num_workers=opt["num_workers"],
    )
    
    ood_test_dataloader = DataLoader(
        dataloader.get_dataset(ood_dataset, opt["data_dir"], False,imgsz = opt["imgsz"]),
        batch_size=opt["batch_size"],
        shuffle=False,
        num_workers=opt["num_workers"],
    )
    return ind_train_dataloader, ind_test_dataloader, ood_test_dataloader




#total_distances /= 100000
def make_uncert_plot(total_distances, classes,auc,folder = "plots/", log_scale = False, title = "", save_img = False, normix = False):
    plt.rcParams["axes.titlesize"] = 8
    total_distances = total_distances if sum(total_distances) == 0 else (total_distances - total_distances.min())/(total_distances.max()-total_distances.min()) if normix == True else total_distances
    in_insts = total_distances[np.array(classes) ==0]
    out_insts = total_distances[np.array(classes) == 1]
    counts, bins = np.histogram(total_distances, 100)
    counts_in, _ = np.histogram(in_insts, bins, density = True)
    counts_out, _ = np.histogram(out_insts, bins, density = True)
    plt.bar(bins[:-1], counts_in,width=0.01, color = "red", label = "in", alpha = 0.5)
    plt.bar(bins[:-1], counts_out,width = 0.01 , color = "blue", label = "out", alpha = 0.5)
    plt.xscale("log") if log_scale == True else None
    title = title + " " +  str(np.round(auc,3))
    plt.title(title ,loc="center", wrap = True)
    plt.legend()
    plt.savefig(folder + title + ".png", format = "png") if save_img == True else None
    



def make_exp(opt, dist_opts, model, layer = "prelogit", kick_true = False, kick_num = -1, ext = 0, data = None):

  #  ckpt_path = os.path.join(
  #      "..",
  #      "model_logs",
  #      opt["experiment_name"],
  #      opt["model"],
  #      opt["ind_dataset"],
  #      f"num_reverse_layers={opt['num_reverse_layers']}",
  #      "checkpoints",
  #      "last.ckpt",
  #  )
#
  #  model = eval(
  #      f"{opt['model']}.load_from_checkpoint('{ckpt_path}', in_channels={opt['in_channels']}, out_channels={opt['num_classes']})"
  #  )
    results = {}
    model = None if model == None else model.eval()
    for ood_dataset in opt["ood_dataset"]:
        print(ood_dataset)

        if model == None:
            all_outs = get_all_outputs_without_model(ind_tr_acts= data[0] ,ind_te_acts = data[1] ,ood_acts= data[2], ind_tr_classes=data[3]) 


        else:
            (ind_train_dataloader, ind_test_dataloader, ood_test_dataloader) = get_ind_and_ood_dataloaders(opt, ood_dataset)
            all_outs = get_all_outputs(model,kick_num,kick_true, ind_train_dataloader, ind_test_dataloader, ood_test_dataloader\
                            ,ind_tr_acts=  None,ind_te_acts = None,ood_te_acts= None) 


        for distance in opt["distances"]:
            print(distance)
            dist_opt = list(itertools.product(*dist_opts[distance].values()))
            keys = dist_opts[distance].keys()
            dist_hypers = [dict(zip(keys, i)) for i in dist_opt]
            for hyper_set in dist_hypers:

    
                results_piccolo, classes, total_distances = ood_experiment(
                                                    all_outputs=all_outs,
                                                    ind_dataset_name=opt["ind_dataset"],
                                                    ood_dataset_name=ood_dataset,
                                                    distance=distance,
                                                    layer= "layer_x",
                                                    hypers = hyper_set
                                                )
                title = f"in_{opt['ind_dataset']}_out_{ood_dataset}_{distance}_{results_piccolo['auc']}"
                results_piccolo["kick_out"] = kick_true
                results_piccolo['hypers'] = hyper_set
                results_piccolo['model_name'] = opt["model_name"]
                results_piccolo["epoch"] = ext
                results_piccolo["title"] = title
                if not os.path.exists( f"plots/{distance}_{opt['ind_dataset']}_{ood_dataset}/"):
                    os.makedirs(f"plots/{distance}_{opt['ind_dataset']}_{ood_dataset}/")
                make_uncert_plot(total_distances, classes, results_piccolo["auc"],log_scale = False,title = f"in_{opt['ind_dataset']}_out_{ood_dataset}_{distance}", folder = f"plots/{distance}_{opt['ind_dataset']}_{ood_dataset}/",save_img=True)
                plt.clf()
                print(results_piccolo)
                df = pd.DataFrame({k: [v] for k, v in results_piccolo.items()})
                df["time"] = datetime.datetime.now()
                if not os.path.exists(opt["outf"]):
                    os.makedirs(opt["outf"])
                df.to_csv(os.path.join(opt["outf"], f"results_{opt['model_name']}.csv"), mode="a", header = False, index=False)
                results = {}

   