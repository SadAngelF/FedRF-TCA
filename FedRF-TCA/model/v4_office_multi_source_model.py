from random import sample
from turtle import Turtle, forward
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
from torchvision import models, transforms as T
import torch
from TorchRandomF import RFF_perso
from model.linear_classifier import bottleneck
from vggnet import svdv2, geometric_approximation
import collections
import math

"""
    Based on Horizontal Setting of Federated Learning, which means that 
    Source data and Target data are in the same featre space, with a domian shift between them.
"""

class federated_office_source_and_target_model(nn.Module):
    def __init__(self, feature_dim, m, n_features, sigma, gamma, kernel='rbf', class_num=31):
        """
        Source forward process:
            X -> conv1 -> conv2 -> source_rf_map -> source_tca_layer -> federated_server -> classifier
        Input:
            feature_dim: faeture dimention of the output of feature extractor;
            m: final feature dimention of transfer feature, top m eigenvectors of Equation;
            n_features: number of random features;
            sigma: variance of Gaussian kernel ;
            gamma: regularization parameter;
            kernel='rbf': kernel function;
        Output:
            fs, ft, ys, yt
            
        (1) Target party broadcast Sigma_i_y_i to all source parties;
        (2) Source party passes Sigma_i_y_i to Target by turns;
        (3) In the end, all Source parties' classifier vote for the result of Target classification. 

        Strategy 2:
        (1) Target party broadcast Sigma_i_y_i to all source parties;
        (2) Source party passes Sigma_i_y_i to Target at the same time, to sum up for Target's aigning features with Source parties;
        (3) In the end, all Source parties' classifier vote for the result of Target classification. 
        """
        super(federated_office_source_and_target_model, self).__init__()
        # feature extractor
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.conv_layers1 = torch.nn.Sequential(*(list(resnet50.children())[0:7])).requires_grad_(False)
        self.source1_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.source2_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.source3_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)
        self.target_conv_layers2 = torch.nn.Sequential(*(list(resnet50.children())[7:9])).requires_grad_(True)

        # random feature map part, with unified random weights
        self.source1_rf_map = federated_rf_map(n_features=n_features, sigma=sigma, feature_dim=feature_dim, kernel=kernel, domain='source')
        self.source2_rf_map = federated_rf_map(n_features=n_features, sigma=sigma, feature_dim=feature_dim, kernel=kernel, domain='source')
        self.source3_rf_map = federated_rf_map(n_features=n_features, sigma=sigma, feature_dim=feature_dim, kernel=kernel, domain='source')
        self.target_rf_map = federated_rf_map(n_features=n_features, sigma=sigma, feature_dim=feature_dim, kernel=kernel, domain='target')
        self.target_rf_map.update_rf_map(self.source1_rf_map.get_rf_map())
        self.source2_rf_map.update_rf_map(self.source1_rf_map.get_rf_map())
        self.source3_rf_map.update_rf_map(self.source1_rf_map.get_rf_map())

        # Wrf calculating, transfer key part
        self.Wrf_source1 = nn.Linear(2*n_features, m, bias=False)
        self.Wrf_source2 = nn.Linear(2*n_features, m, bias=False)
        self.Wrf_source3 = nn.Linear(2*n_features, m, bias=False)
        self.Wrf_target = nn.Linear(2*n_features, m, bias=False)
        
        # classifier
        self.source1_classifier = bottleneck(input_features_dim=m, output=class_num).requires_grad_(True)
        self.source2_classifier = bottleneck(input_features_dim=m, output=class_num).requires_grad_(True)
        self.source3_classifier = bottleneck(input_features_dim=m, output=class_num).requires_grad_(True)
        self.target_classifier = bottleneck(input_features_dim=m, output=class_num).requires_grad_(True)

    def forward(self, x_s1, x_s2, x_s3, x_t):
        """"""
        x_s1 = self.conv_layers1(x_s1)
        x_s2 = self.conv_layers1(x_s2)
        x_s3 = self.conv_layers1(x_s3)
        x_t = self.conv_layers1(x_t)

        x_s1 = self.source1_conv_layers2(x_s1)
        x_s2 = self.source2_conv_layers2(x_s2)
        x_s3 = self.source3_conv_layers2(x_s3)
        x_t = self.target_conv_layers2(x_t)

        x_s1 = torch.flatten(x_s1, start_dim=1)
        x_s2 = torch.flatten(x_s2, start_dim=1)
        x_s3 = torch.flatten(x_s3, start_dim=1)
        x_t = torch.flatten(x_t, start_dim=1)

        # each party runs independently
        x_s1 = self.sample_norm(x_s1)
        x_s2 = self.sample_norm(x_s2)
        x_s3 = self.sample_norm(x_s3)
        x_t = self.sample_norm(x_t)
        source1_Sigma_i, source1_Sigma_yi = self.source1_rf_map(x_s1)
        source2_Sigma_i, source2_Sigma_yi = self.source2_rf_map(x_s2)
        source3_Sigma_i, source3_Sigma_yi = self.source3_rf_map(x_s3)
        target_Sigma_i, target_Sigma_yi = self.target_rf_map(x_t)

        # generate new transfer features
        fs1 = self.Wrf_source1(source1_Sigma_i.T)
        fs1 = self.sample_norm(fs1)
        ys1 = self.source1_classifier(fs1)
        
        fs2 = self.Wrf_source2(source2_Sigma_i.T)
        fs2 = self.sample_norm(fs2)
        ys2 = self.source2_classifier(fs2)

        fs3 = self.Wrf_source3(source3_Sigma_i.T)
        fs3 = self.sample_norm(fs3)
        ys3 = self.source3_classifier(fs3)

        ft = self.Wrf_target(target_Sigma_i.T)
        ft = self.sample_norm(ft)
        yt = self.target_classifier(ft)
        # yt = self.multi_prediction_for_target(ft)

        return source1_Sigma_yi, source2_Sigma_yi, source3_Sigma_yi, target_Sigma_yi, ys1, ys2, ys3, yt # fs1, fs2, fs3, ft, # avoid user data leakage


    def multi_prediction_for_target(self, ft):
        # self.target_classifier = self.source_classifier
        yt1 = self.source1_classifier(ft)
        yt2 = self.source2_classifier(ft)
        yt3 = self.source3_classifier(ft)
        return torch.cat([yt1.unsqueeze(1), yt2.unsqueeze(1), yt3.unsqueeze(1), ], dim=1)

    # CLassifier Layer Aggregation
    def FedAvg_classifier(self, subset_list=None):
        models_num = 3 
        models_list = [self.source1_classifier, self.source2_classifier, self.source3_classifier]        
        if subset_list != None:
            models_num = len(subset_list)
            models_list = [models_list[subset_i] for subset_i in subset_list]
        models_state_dict = [x.state_dict() for x in models_list]
        weight_keys = list(models_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(models_num):
                key_sum = key_sum + models_state_dict[i][key]
            fed_state_dict[key] = key_sum / models_num
        # update fed weights to fed model
        self.source1_classifier.load_state_dict(fed_state_dict)
        self.source2_classifier.load_state_dict(fed_state_dict)
        self.source3_classifier.load_state_dict(fed_state_dict)
        self.target_classifier.load_state_dict(fed_state_dict)  

    def Weighted_FedAvg_classifier(self, subset_list=None, ns_list=None):
        domain_num_list = [1,1,1] 
        models_list = [self.source1_classifier, self.source2_classifier, self.source3_classifier]        
        if subset_list != None and ns_list != None:
            domain_num_list = [ns_list[domain_i] for domain_i in subset_list]
            models_list = [models_list[subset_i] for subset_i in subset_list]
        models_state_dict = [x.state_dict() for x in models_list]
        weight_keys = list(models_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(domain_num_list)):
                key_sum = key_sum + (domain_num_list[i] / sum(domain_num_list)) * models_state_dict[i][key]
            fed_state_dict[key] = key_sum
        # update fed weights to fed model
        self.source1_classifier.load_state_dict(fed_state_dict)
        self.source2_classifier.load_state_dict(fed_state_dict)
        self.source3_classifier.load_state_dict(fed_state_dict)
        self.target_classifier.load_state_dict(fed_state_dict)  

    # W_rf Layer Aggregation
    def FedAvg_transfer_layer(self, subset_list=None):
        models_num = 4
        models_list = [self.Wrf_source1, self.Wrf_source2, self.Wrf_source3, self.Wrf_target]
        if subset_list != None:
            models_num = len(subset_list) + 1
            models_list = [models_list[subset_i] for subset_i in subset_list] + [self.Wrf_target]
        models_state_dict = [x.state_dict() for x in models_list]
        weight_keys = list(models_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(models_num):
                key_sum = key_sum + models_state_dict[i][key]
            fed_state_dict[key] = key_sum / models_num
        # update fed weights to fed model
        if subset_list == None:
            self.Wrf_source1.load_state_dict(fed_state_dict)
            self.Wrf_source2.load_state_dict(fed_state_dict)
            self.Wrf_source3.load_state_dict(fed_state_dict)
        else:
            if 0 in subset_list:
                self.Wrf_source1.load_state_dict(fed_state_dict)
            if 1 in subset_list:
                self.Wrf_source2.load_state_dict(fed_state_dict)
            if 2 in subset_list:
                self.Wrf_source3.load_state_dict(fed_state_dict)      
        self.Wrf_target.load_state_dict(fed_state_dict)

    def Weighted_FedAvg_transfer_layer(self, subset_list=None, ns_list=None, nt=None):
        domain_num_list = [1,1,1,1] 
        models_list = [self.Wrf_source1, self.Wrf_source2, self.Wrf_source3, self.Wrf_target]
        if subset_list != None:
            domain_num_list = [ns_list[domain_i] for domain_i in subset_list]
            domain_num_list.append(nt)
            models_list = [models_list[subset_i] for subset_i in subset_list] + [self.Wrf_target]
        models_state_dict = [x.state_dict() for x in models_list]
        weight_keys = list(models_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(domain_num_list)):
                key_sum = key_sum + (domain_num_list[i] / sum(domain_num_list)) * models_state_dict[i][key]
            fed_state_dict[key] = key_sum #  / models_num
        # update fed weights to fed model
        if subset_list == None:
            self.Wrf_source1.load_state_dict(fed_state_dict)
            self.Wrf_source2.load_state_dict(fed_state_dict)
            self.Wrf_source3.load_state_dict(fed_state_dict)
        else:
            if 0 in subset_list:
                self.Wrf_source1.load_state_dict(fed_state_dict)
            if 1 in subset_list:
                self.Wrf_source2.load_state_dict(fed_state_dict)
            if 2 in subset_list:
                self.Wrf_source3.load_state_dict(fed_state_dict)    
        self.Wrf_target.load_state_dict(fed_state_dict)

    def FedAvg_only_target_classifier(self, cls_acc1, cls_acc2, cls_acc3, cls_acc4):
        models_num = 3 # source and target, 2 parties
        # model_weights = [math.log(cls_acc1), math.log(cls_acc2), math.log(cls_acc3), math.log(cls_acc4)]
        # model_weights = [1/cls_acc1, 1/cls_acc2, 1/cls_acc3, 1/cls_acc4]
        model_weights = [cls_acc1, cls_acc2, cls_acc3]
        model_constant = sum(model_weights)
        models_list = [self.source1_classifier, self.source2_classifier, self.source3_classifier]
        models_state_dict = [x.state_dict() for x in models_list]
        weight_keys = list(models_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(models_num):
                key_sum = key_sum + (model_weights[i] / model_constant) * models_state_dict[i][key]
            fed_state_dict[key] = key_sum
        # update fed weights to fed model
        self.target_classifier.load_state_dict(fed_state_dict)
        return model_constant, (model_weights[0] / model_constant), (model_weights[1] / model_constant), (model_weights[2] / model_constant)
        
    def FedAvg_pair_transfer_layer(self, iter_num):
        models_num = 2 # source and target, 2 parties
        # models_list = [self.Wrf_source1, self.Wrf_source2, self.Wrf_source3, self.Wrf_source4, self.Wrf_target]
        if iter_num == 0:
            models_list = [self.Wrf_source1, self.Wrf_target]
        elif iter_num == 1:
            models_list = [self.Wrf_source2, self.Wrf_target]
        elif iter_num == 2:
            models_list = [self.Wrf_source3, self.Wrf_target]
        models_state_dict = [x.state_dict() for x in models_list]
        weight_keys = list(models_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(models_num):
                key_sum = key_sum + models_state_dict[i][key]
            fed_state_dict[key] = key_sum / models_num
        # update fed weights to fed model
        if iter_num == 0:
            self.Wrf_source1.load_state_dict(fed_state_dict)
        elif iter_num == 1:
            self.Wrf_source2.load_state_dict(fed_state_dict)
        elif iter_num == 2:
            self.Wrf_source3.load_state_dict(fed_state_dict)      
        self.Wrf_target.load_state_dict(fed_state_dict)
    
    def mmd_loss_source1(self, source_Sigma_yi, target_Sigma_yi_detach):
        Sigma = source_Sigma_yi + target_Sigma_yi_detach
        mmd_loss_part = self.Wrf_source1(Sigma.T)
        return torch.mm(mmd_loss_part, mmd_loss_part.T)

    def mmd_loss_source2(self, source_Sigma_yi, target_Sigma_yi_detach):
        Sigma = source_Sigma_yi + target_Sigma_yi_detach
        mmd_loss_part = self.Wrf_source2(Sigma.T)
        return torch.mm(mmd_loss_part, mmd_loss_part.T)

    def mmd_loss_source3(self, source_Sigma_yi, target_Sigma_yi_detach):
        Sigma = source_Sigma_yi + target_Sigma_yi_detach
        mmd_loss_part = self.Wrf_source3(Sigma.T)
        return torch.mm(mmd_loss_part, mmd_loss_part.T)

    def mmd_loss_source4(self, source_Sigma_yi, target_Sigma_yi_detach):
        Sigma = source_Sigma_yi + target_Sigma_yi_detach
        mmd_loss_part = self.Wrf_source4(Sigma.T)
        return torch.mm(mmd_loss_part, mmd_loss_part.T)

    def mmd_loss_target(self, source_Sigma_yi_detach, target_Sigma_yi):
        Sigma = source_Sigma_yi_detach + target_Sigma_yi
        mmd_loss_part = self.Wrf_target(Sigma.T)
        return torch.mm(mmd_loss_part, mmd_loss_part.T)

    def forward_rf_map(self, x):
        x = self.conv_layers1(x)
        x = self.source1_conv_layers2(x)

        x = torch.flatten(x, start_dim=1)

        # each party runs independently
        x = self.sample_norm(x)
        source1_Sigma_i, source1_Sigma_yi = self.source1_rf_map(x)
        return source1_Sigma_i, source1_Sigma_yi
    
    def init_W_rf(self, W_rf_s1, W_rf_s2, W_rf_s3, W_rf_s4, W_rf_t):
        with torch.no_grad():
            self.Wrf_source1.weight.copy_(W_rf_s1.T)
            self.Wrf_source2.weight.copy_(W_rf_s2.T)
            self.Wrf_source3.weight.copy_(W_rf_s3.T)
            self.Wrf_source4.weight.copy_(W_rf_s4.T)
            self.Wrf_target.weight.copy_(W_rf_t.T)
        
    def sample_norm(self, x_new):
        x_new = x_new.T
        x_new = x_new / (torch.linalg.norm(x_new, dim=0) + 1e-5)
        x_new = x_new.T
        return x_new

class federated_rf_map(nn.Module):
    def __init__(self, n_features, sigma, kernel='rbf', feature_dim=2048, domain='source'):
        super(federated_rf_map, self).__init__()
        self.rf_map = RFF_perso(sigma, n_features, kernel=kernel)
        self.rf_map.fit(feature_dim)
        self.domain = domain

    def forward(self, X):
        Xdevice = X.device
        ns_i = X.size(dim=0)
        X = self.sample_norm(X)

        # compute Sigma_i on i-th iteration
        assert not torch.any(torch.isnan(X))
        Sigma_i = self.rf_map.transform(X).T # Sigma_i: (2*n_features, n)
        assert not torch.any(torch.isnan(Sigma_i))

        # init y_i vector
        if self.domain == 'source':
            yi = 1/ns_i * torch.ones(ns_i, 1)
        elif self.domain == 'target':
            yi = -1/ns_i * torch.ones(ns_i, 1)
        yi = yi.to(Xdevice)

        Sigmai_yi = torch.mm(Sigma_i, yi)
                
        return Sigma_i, Sigmai_yi

    def get_rf_map(self):
        return self.rf_map
    
    def update_rf_map(self, rf_map):
        self.rf_map = rf_map

    def sample_norm(self, x_new):
        x_new = x_new.T
        x_new = x_new / (torch.linalg.norm(x_new, dim=0) + 1e-5)
        x_new = x_new.T
        return x_new
