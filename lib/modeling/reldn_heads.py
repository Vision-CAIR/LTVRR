# modified by Sherif Abdelkarim on Jan 2020

import numpy as np
from numpy import linalg as la
import math
import logging
import json

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import nn as mynn
import random

from core.config import cfg
import utils.net as net_utils
from modeling.sparse_targets_rel import FrequencyBias
from utils import focal_loss

logger = logging.getLogger(__name__)

RG = np.random.default_rng()

def create_one_hot(y, classes, device_id):
    y_onehot = torch.FloatTensor(y.size(0), classes).cuda(device_id)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot


def cumix(sbj_vis_embeddings, sbj_labels, obj_vis_embeddings, obj_labels, prd_vis_embeddings, prd_labels, indices_1, indices_2, indices_3, device_id):
    
    if cfg.random_lamda:
        lamda = torch.from_numpy(RG.beta(0.8, 0.8, [indices_1.shape[0], 1])).float()
    else:
        lamda = 0.65

    alpha1 = torch.randint(0, 2, [indices_1.shape[0], 1]).cuda(device_id)
    # alpha2 = torch.randint(0, 2, [indices_1.shape[0]]).cuda(device_id)
    # alpha3 = torch.randint(0, 2, [indices_1.shape[0]]).cuda(device_id)

    if cfg.mixup:
        mixed_sbj_embeddings = lamda * sbj_vis_embeddings[indices_1] + (1 - lamda)*(sbj_vis_embeddings[indices_2])
        mixed_obj_embeddings = lamda * obj_vis_embeddings[indices_1] + (1 - lamda)*(obj_vis_embeddings[indices_2])
        mixed_prd_embeddings = lamda * prd_vis_embeddings[indices_1] + (1 - lamda)*(prd_vis_embeddings[indices_2])

        sbj_one_hot_labels = create_one_hot(sbj_labels, cfg.MODEL.NUM_CLASSES -1, device_id)
        prd_one_hot_labels = create_one_hot(prd_labels, cfg.MODEL.NUM_PRD_CLASSES +1, device_id)
        obj_one_hot_labels = create_one_hot(obj_labels, cfg.MODEL.NUM_CLASSES -1, device_id)

        mixed_sbj_labels = lamda * sbj_one_hot_labels[indices_1] + (1 - lamda)*(sbj_one_hot_labels[indices_2])
        mixed_obj_labels = lamda * obj_one_hot_labels[indices_1] + (1 - lamda)*(obj_one_hot_labels[indices_2])
        mixed_prd_labels = lamda * prd_one_hot_labels[indices_1] + (1 - lamda)*(prd_one_hot_labels[indices_2])

    else:
        mixed_sbj_embeddings = lamda * sbj_vis_embeddings[indices_1] + (1 - lamda)*(alpha1 * sbj_vis_embeddings[indices_2] + \
               (1 - alpha1) * sbj_vis_embeddings[indices_3])
        mixed_obj_embeddings = lamda * obj_vis_embeddings[indices_1] + (1 - lamda)*(alpha1 * obj_vis_embeddings[indices_2] + \
               (1 - alpha1) * obj_vis_embeddings[indices_3])
        mixed_prd_embeddings = lamda * prd_vis_embeddings[indices_1] + (1 - lamda)*(alpha1 * prd_vis_embeddings[indices_2] + \
               (1 - alpha1) * prd_vis_embeddings[indices_3])

        sbj_one_hot_labels = create_one_hot(sbj_labels, cfg.MODEL.NUM_CLASSES -1, device_id)
        prd_one_hot_labels = create_one_hot(prd_labels, cfg.MODEL.NUM_PRD_CLASSES +1, device_id)
        obj_one_hot_labels = create_one_hot(obj_labels, cfg.MODEL.NUM_CLASSES -1, device_id)

        mixed_sbj_labels = lamda * sbj_one_hot_labels[indices_1] + (1 - lamda)*(alpha1 * sbj_one_hot_labels[indices_2] + \
                (1 - alpha1) * sbj_one_hot_labels[indices_3])
        mixed_obj_labels = lamda * obj_one_hot_labels[indices_1] + (1 - lamda)*(alpha1 * obj_one_hot_labels[indices_2] + \
                (1 - alpha1) * obj_one_hot_labels[indices_3])
        mixed_prd_labels = lamda * prd_one_hot_labels[indices_1] + (1 - lamda)*(alpha1 * prd_one_hot_labels[indices_2] + \
                (1 - alpha1) * prd_one_hot_labels[indices_3])

    return mixed_sbj_embeddings, mixed_sbj_labels, mixed_obj_embeddings, mixed_obj_labels, mixed_prd_embeddings, mixed_prd_labels



class reldn_head(nn.Module):
    def __init__(self, dim_in, all_obj_vecs=None, all_prd_vecs=None):
        super().__init__()
            
        num_prd_classes = cfg.MODEL.NUM_PRD_CLASSES + 1
            
        if cfg.MODEL.RUN_BASELINE:
            # only run it on testing mode
            self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])
            return
        
        self.obj_vecs = all_obj_vecs
        self.prd_vecs = all_prd_vecs

        # add subnet
        self.prd_feats = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.LeakyReLU(0.1))
        self.prd_vis_embeddings = nn.Sequential(
            nn.Linear(1024 * 3, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024))
        if not cfg.MODEL.USE_SEM_CONCAT:
            self.prd_sem_embeddings = nn.Sequential(
                nn.Linear(cfg.MODEL.INPUT_LANG_EMBEDDING_DIM, 1024),
                nn.LeakyReLU(0.1),
                nn.Linear(1024, 1024))
        else:
            self.prd_sem_hidden = nn.Sequential(
                nn.Linear(cfg.MODEL.INPUT_LANG_EMBEDDING_DIM, 1024),
                nn.LeakyReLU(0.1),
                nn.Linear(1024, 1024))
            self.prd_sem_embeddings = nn.Linear(3 * 1024, 1024)
        
        self.so_vis_embeddings = nn.Linear(dim_in // 3, 1024)
        self.so_sem_embeddings = nn.Sequential(
            nn.Linear(cfg.MODEL.INPUT_LANG_EMBEDDING_DIM, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024))
            
        if cfg.MODEL.USE_FREQ_BIAS:
            # Assume we are training/testing on only one dataset
            if len(cfg.TRAIN.DATASETS):
                self.freq_bias = FrequencyBias(cfg.TRAIN.DATASETS[0])
            else:
                self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # spo_feat is concatenation of SPO
    def forward(self, spo_feat, prd_weights, sbj_labels=None, obj_labels=None, sbj_feat=None, obj_feat=None, prd_labels=None):

        device_id = spo_feat.get_device()
        if sbj_labels is not None:
            sbj_labels = Variable(torch.from_numpy(sbj_labels.astype('int64'))).cuda(device_id)
        if obj_labels is not None:
            obj_labels = Variable(torch.from_numpy(obj_labels.astype('int64'))).cuda(device_id)
        if prd_labels is not None:
            prd_labels = Variable(torch.from_numpy(prd_labels.astype('int64'))).cuda(device_id)
            
        if cfg.MODEL.RUN_BASELINE:
            assert sbj_labels is not None and obj_labels is not None
            prd_cls_scores = self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels, obj_labels), 1))
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)
            return prd_cls_scores, None, None, None, None, None
        
        if spo_feat.dim() == 4:
            spo_feat = spo_feat.squeeze(3).squeeze(2)

        # prd_labels = rel_ret['all_prd_labels_int32']
        # sbj_labels = rel_ret['all_sbj_labels_int32']
        # obj_labels = rel_ret['all_obj_labels_int32']
        
        sbj_vis_embeddings = self.so_vis_embeddings(sbj_feat)
        obj_vis_embeddings = self.so_vis_embeddings(obj_feat)
        
        prd_hidden = self.prd_feats(spo_feat)
        prd_features = torch.cat((sbj_vis_embeddings.detach(), prd_hidden, obj_vis_embeddings.detach()), dim=1)
        prd_vis_embeddings = self.prd_vis_embeddings(prd_features)

        ds_obj_vecs = self.obj_vecs
        ds_obj_vecs = Variable(torch.from_numpy(ds_obj_vecs.astype('float32'))).cuda(device_id)
        so_sem_embeddings = self.so_sem_embeddings(ds_obj_vecs)
        so_sem_embeddings = F.normalize(so_sem_embeddings, p=2, dim=1)  # (#prd, 1024)
        so_sem_embeddings.t_()

        sbj_vis_embeddings_n = F.normalize(sbj_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
        # print('!! SBJ_VIS_EMBEDDINGS: !! ', sbj_vis_embeddings.shape)
        sbj_sim_matrix = torch.mm(sbj_vis_embeddings_n, so_sem_embeddings)  # (#bs, #prd)
        # print('!! SBJ_SIM_MATRIX: !! ', sbj_sim_matrix.shape)
        sbj_cls_scores = cfg.MODEL.NORM_SCALE * sbj_sim_matrix
        
        obj_vis_embeddings_n = F.normalize(obj_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
        obj_sim_matrix = torch.mm(obj_vis_embeddings_n, so_sem_embeddings)  # (#bs, #prd)
        obj_cls_scores = cfg.MODEL.NORM_SCALE * obj_sim_matrix
        
        if not cfg.MODEL.USE_SEM_CONCAT:
            ds_prd_vecs = self.prd_vecs
            ds_prd_vecs = Variable(torch.from_numpy(ds_prd_vecs.astype('float32'))).cuda(device_id)
            prd_sem_embeddings = self.prd_sem_embeddings(ds_prd_vecs)
            prd_sem_embeddings = F.normalize(prd_sem_embeddings, p=2, dim=1)  # (#prd, 1024)
            prd_vis_embeddings_n = F.normalize(prd_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
            prd_sim_matrix = torch.mm(prd_vis_embeddings_n, prd_sem_embeddings.t_())  # (#bs, #prd)
            prd_cls_scores = cfg.MODEL.NORM_SCALE * prd_sim_matrix
        else:
            ds_prd_vecs = self.prd_vecs
            ds_prd_vecs = Variable(torch.from_numpy(ds_prd_vecs.astype('float32'))).cuda(device_id)
            prd_sem_hidden = self.prd_sem_hidden(ds_prd_vecs)  # (#prd, 1024)
            # get sbj vis embeddings and expand to (#bs, #prd, 1024)
            sbj_vecs = self.obj_vecs[sbj_labels]  # (#bs, cfg.MODEL.INPUT_LANG_EMBEDDING_DIM)
            sbj_vecs = Variable(torch.from_numpy(sbj_vecs.astype('float32'))).cuda(device_id)
            if len(list(sbj_vecs.size())) == 1:  # sbj_vecs should be 2d
                sbj_vecs.unsqueeze_(0)
            sbj_sem_embeddings = self.so_sem_embeddings(sbj_vecs)  # (#bs, 1024)
            sbj_sem_embeddings = sbj_sem_embeddings.unsqueeze(1).expand(
                sbj_vecs.shape[0], ds_prd_vecs.shape[0], 1024)  # (#bs, 1024) --> # (#bs, 1, 1024) --> # (#bs, #prd, 1024)
            # get obj vis embeddings and expand to (#bs, #prd, 1024)
            obj_vecs = self.obj_vecs[obj_labels]  # (#bs, cfg.MODEL.INPUT_LANG_EMBEDDING_DIM)
            obj_vecs = Variable(torch.from_numpy(obj_vecs.astype('float32'))).cuda(device_id)
            if len(list(obj_vecs.size())) == 1:  # obj_vecs should be 2d
                obj_vecs.unsqueeze_(0)
            obj_sem_embeddings = self.so_sem_embeddings(obj_vecs)  # (#bs, 1024)
            obj_sem_embeddings = obj_sem_embeddings.unsqueeze(1).expand(
                obj_vecs.shape[0], ds_prd_vecs.shape[0], 1024)  # (#bs, 1024) --> # (#bs, 1, 1024) --> # (#bs, #prd, 1024)
            # expand prd hidden feats to (#bs, #prd, 1024)
            prd_sem_hidden = prd_sem_hidden.unsqueeze(0).expand(
                sbj_vecs.shape[0], ds_prd_vecs.shape[0], 1024)  # (#prd, 1024) --> # (1, #prd, 1024) --> # (#bs, #prd, 1024)
            # now feed semantic SPO features into the last prd semantic layer
            spo_sem_feat = torch.cat(
                (sbj_sem_embeddings.detach(),
                 prd_sem_hidden,
                 obj_sem_embeddings.detach()),
                dim=2)  # (#bs, #prd, 3 * 1024)
            # get prd scores
            prd_sem_embeddings = self.prd_sem_embeddings(spo_sem_feat)  # (#bs, #prd, 1024)
            prd_sem_embeddings = F.normalize(prd_sem_embeddings, p=2, dim=2)  # (#bs, #prd, 1024)
            prd_vis_embeddings = F.normalize(prd_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
            prd_vis_embeddings = prd_vis_embeddings.unsqueeze(-1)  # (#bs, 1024) --> (#bs, 1024, 1)
            prd_sim_matrix = torch.bmm(prd_sem_embeddings, prd_vis_embeddings).squeeze(-1)  # bmm((#bs, #prd, 1024), (#bs, 1024, 1)) = (#bs, #prd, 1) --> (#bs, #prd)
            prd_cls_scores = cfg.MODEL.NORM_SCALE * prd_sim_matrix
        
        mixed_sbj_cls_scores = None
        mixed_obj_cls_scores = None
        mixed_prd_cls_scores = None
        mixed_sbj_labels = None
        mixed_obj_labels = None
        mixed_prd_labels = None

        if cfg.cumix and self.training:
            bs = sbj_vis_embeddings.shape[0]
            probs = prd_weights[prd_labels]
            indices = np.argsort(-probs, kind='quicksort')     ### in decreasing order, means tail classes are ahead

            if cfg.aug_percent == 50:
                partition_num = bs // 2
                indices_1 = indices[0:partition_num]
                indices_2 = np.random.permutation(indices_1)
                indices_3 = indices[partition_num: partition_num*2]
            elif cfg.aug_percent == 70:
                partition_num = int(bs * 0.7)
                indices_1 = indices[0:partition_num]
                indices_2 = np.random.permutation(indices_1)
                indices_3 = np.random.permutation(indices_1)
            elif cfg.aug_percent == 60:
                partition_num = int(bs * 0.6)
                indices_1 = indices[0:partition_num]
                indices_2 = np.random.permutation(indices_1)
                indices_3 = np.random.permutation(indices_1)
            else:
                partition_num = bs // 3
                indices_1 = indices[0:partition_num]
                indices_2 = indices[partition_num : partition_num*2]
                indices_3 = indices[partition_num*2 : partition_num*3]

            mixed_sbj_embeddings, mixed_sbj_labels, mixed_obj_embeddings, mixed_obj_labels, mixed_prd_embeddings, mixed_prd_labels  = \
                cumix(sbj_vis_embeddings, sbj_labels, obj_vis_embeddings, obj_labels, prd_vis_embeddings, prd_labels, indices_1, indices_2, indices_3, device_id)

            # mixed_obj_labels = mixed_obj_labels.cuda(device_id)
            # mixed_sbj_labels = mixed_sbj_labels.cuda(device_id)
            # mixed_prd_labels = mixed_prd_labels.cuda(device_id)

            mixed_sbj_vis_embeddings = F.normalize(mixed_sbj_embeddings, p=2, dim=1)  # (#bs, 1024)
            mixed_sbj_sim_matrix = torch.mm(mixed_sbj_vis_embeddings, so_sem_embeddings)  # (#bs, #prd)
            mixed_sbj_cls_scores = cfg.MODEL.NORM_SCALE * mixed_sbj_sim_matrix

            mixed_obj_vis_embeddings = F.normalize(mixed_obj_embeddings, p=2, dim=1)  # (#bs, 1024)
            mixed_obj_sim_matrix = torch.mm(mixed_obj_vis_embeddings, so_sem_embeddings)  # (#bs, #prd)
            mixed_obj_cls_scores = cfg.MODEL.NORM_SCALE * mixed_obj_sim_matrix

            mixed_prd_vis_embeddings = F.normalize(mixed_prd_embeddings, p=2, dim=1)  # (#bs, 1024)
            mixed_prd_sim_matrix = torch.mm(mixed_prd_vis_embeddings, prd_sem_embeddings)  # (#bs, #prd)
            mixed_prd_cls_scores = cfg.MODEL.NORM_SCALE * mixed_prd_sim_matrix


        if cfg.MODEL.USE_FREQ_BIAS:
            assert sbj_labels is not None and obj_labels is not None
            prd_cls_scores = prd_cls_scores + self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels, obj_labels), 1))
            
        if not self.training:
            sbj_cls_scores = F.softmax(sbj_cls_scores, dim=1)
            obj_cls_scores = F.softmax(obj_cls_scores, dim=1)
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)
        
        return prd_cls_scores, sbj_cls_scores, obj_cls_scores, mixed_sbj_cls_scores, mixed_obj_cls_scores, mixed_prd_cls_scores, \
                mixed_sbj_labels, mixed_obj_labels, mixed_prd_labels


def manual_CE(predictions, labels):
    loss = -torch.mean(torch.sum(labels * torch.log_softmax(predictions,dim=1),dim=1))
    return loss

def manual_log_softmax(pred, weight):
    e_x = torch.exp(pred - torch.max(pred))
    # print('!! e_x shape !! ', e_x.shape)
    # print('!! weight shape !! ', weight.shape)
    return e_x / torch.sum(weight * e_x, 1).unsqueeze(1)

def eql_loss_so(sbj_cls_scores, sbj_labels_int32, obj_weight):
    device_id = sbj_cls_scores.get_device()
    sbj_labels = Variable(torch.from_numpy(sbj_labels_int32.astype('int64'))).cuda(device_id)

    weight = Variable(torch.from_numpy(obj_weight)).cuda(device_id)
    one_hot_labels = create_one_hot(sbj_labels, cfg.MODEL.NUM_CLASSES - 1, device_id)

    probs = weight[sbj_labels]
    threshold = torch.FloatTensor([1 if a_ > 4.3e-04 else 0 for a_ in probs]).cuda(device_id)
    threshold = threshold.unsqueeze(1)

    beta = torch.empty((sbj_labels.shape[0], 1)).fill_(0.9)
    beta = torch.bernoulli(beta).cuda(device_id)

    w_k = 1. - beta * threshold * (1. - one_hot_labels)

    softmax_output = manual_log_softmax(sbj_cls_scores, w_k)
    loss = -torch.mean(torch.sum(one_hot_labels * softmax_output,dim=1))

    prd_cls_preds = sbj_cls_scores.max(dim=1)[1].type_as(sbj_labels)
    accuracy_cls_prd = prd_cls_preds.eq(sbj_labels).float().mean(dim=0)

    return loss, accuracy_cls_prd


def eql_loss_prd(prd_cls_scores, prd_labels_int32, prd_weight):
    device_id = prd_cls_scores.get_device()
    prd_labels = Variable(torch.from_numpy(prd_labels_int32.astype('int64'))).cuda(device_id)

    weight = Variable(torch.from_numpy(prd_weight)).cuda(device_id)
    one_hot_labels = create_one_hot(prd_labels, cfg.MODEL.NUM_PRD_CLASSES + 1, device_id)

    probs = weight[prd_labels]
    threshold = torch.FloatTensor([1 if a_ > 4.3e-04 else 0 for a_ in probs]).cuda(device_id)
    threshold = threshold.unsqueeze(1)

    beta = torch.empty((prd_labels.shape[0], 1)).fill_(0.9)
    beta = torch.bernoulli(beta).cuda(device_id)

    w_k = 1. - beta * threshold * (1. - one_hot_labels)

    softmax_output = manual_log_softmax(prd_cls_scores, w_k)
    loss = -torch.mean(torch.sum(one_hot_labels * softmax_output,dim=1))

    prd_cls_preds = prd_cls_scores.max(dim=1)[1].type_as(prd_labels)
    accuracy_cls_prd = prd_cls_preds.eq(prd_labels).float().mean(dim=0)

    return loss, accuracy_cls_prd


def add_cls_loss(cls_scores, labels, weight=None):
    if cfg.MODEL.LOSS == 'cross_entropy':
        return F.cross_entropy(cls_scores, labels)
    elif cfg.MODEL.LOSS == 'weighted_cross_entropy':
        return F.cross_entropy(cls_scores, labels, weight=weight)
    elif cfg.MODEL.LOSS == 'focal':
        cls_scores_exp = cls_scores.unsqueeze(2)
        cls_scores_exp = cls_scores_exp.unsqueeze(3)
        labels_exp = labels.unsqueeze(1)
        labels_exp = labels_exp.unsqueeze(2)
        return focal_loss.focal_loss(cls_scores_exp, labels_exp, alpha=cfg.MODEL.ALPHA, gamma=cfg.MODEL.GAMMA, reduction='mean')
    elif cfg.MODEL.LOSS == 'weighted_focal':
        cls_scores_exp = cls_scores.unsqueeze(2)
        cls_scores_exp = cls_scores_exp.unsqueeze(3)
        labels_exp = labels.unsqueeze(1)
        labels_exp = labels_exp.unsqueeze(2)
        weight = weight.unsqueeze(0)
        weight = weight.unsqueeze(2)
        weight = weight.unsqueeze(3)
        return focal_loss.focal_loss(cls_scores_exp, labels_exp, alpha=cfg.MODEL.ALPHA, gamma=cfg.MODEL.GAMMA, reduction='mean', weight_ce=weight)
    else:
        raise NotImplementedError

def add_hubness_loss(cls_scores):
    # xp_yall_prob   (batch_size, num_classes)
    # xp_yall_prob.T (num_classes, batch_size
    # xp_yall_prob.expand(0, 1, -1, 1)
    # xp_yall_probT_average_reshape = xp_yall_probT_reshaped.mean(axis=2)
    # hubness_dist = xp_yall_probT_average_reshape - hubness_blob
    # hubness_dist_sqr = hubness_dist.pow(2)
    # hubness_dist_sqr_scaled = hubness_dist_sqr * cfg.TRAIN.HUBNESS_SCALE
    cls_scores = F.softmax(cls_scores, dim=1)
    hubness_blob = 1./cls_scores.size(1)
    cls_scores_T = cls_scores.transpose(0, 1)
    cls_scores_T = cls_scores_T.unsqueeze(1).unsqueeze(3).expand(-1, 1, -1, 1)
    cls_scores_T = cls_scores_T.mean(dim=2, keepdim=True)
    hubness_dist = cls_scores_T - hubness_blob
    hubness_dist = hubness_dist.pow(2) * cfg.TRAIN.HUBNESS_SCALE
    hubness_loss = hubness_dist.mean()
    return hubness_loss


def reldn_losses(prd_cls_scores, prd_labels_int32, fg_only=False, weight=None):
    device_id = prd_cls_scores.get_device()
    prd_labels = Variable(torch.from_numpy(prd_labels_int32.astype('int64'))).cuda(device_id)
    if cfg.MODEL.LOSS == 'weighted_cross_entropy' or cfg.MODEL.LOSS == 'weighted_focal':
        weight = Variable(torch.from_numpy(weight)).cuda(device_id)
    loss_cls_prd = add_cls_loss(prd_cls_scores, prd_labels, weight=weight)
    # class accuracy
    prd_cls_preds = prd_cls_scores.max(dim=1)[1].type_as(prd_labels)
    accuracy_cls_prd = prd_cls_preds.eq(prd_labels).float().mean(dim=0)

    return loss_cls_prd, accuracy_cls_prd


def reldn_so_losses(sbj_cls_scores, obj_cls_scores, sbj_labels_int32, obj_labels_int32):
    device_id = sbj_cls_scores.get_device()

    sbj_labels = Variable(torch.from_numpy(sbj_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_sbj = add_cls_loss(sbj_cls_scores, sbj_labels)
    sbj_cls_preds = sbj_cls_scores.max(dim=1)[1].type_as(sbj_labels)
    accuracy_cls_sbj = sbj_cls_preds.eq(sbj_labels).float().mean(dim=0)
    
    obj_labels = Variable(torch.from_numpy(obj_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_obj = add_cls_loss(obj_cls_scores, obj_labels)
    obj_cls_preds = obj_cls_scores.max(dim=1)[1].type_as(obj_labels)
    accuracy_cls_obj = obj_cls_preds.eq(obj_labels).float().mean(dim=0)
    
    return loss_cls_sbj, loss_cls_obj, accuracy_cls_sbj, accuracy_cls_obj
