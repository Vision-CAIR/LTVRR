# modified by Sherif Abdelkarim on Jan 2020

from functools import wraps
import importlib
import logging
import math
import numpy as np
from numpy import linalg as la
import copy
import gensim
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.relpn_heads as relpn_heads
import modeling.reldn_heads as reldn_heads
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils
import utils.fpn as fpn_utils
from utils.memory_utils import *
import cv2
from torchvision.utils import save_image
import torchvision
from torchvision import transforms
import random
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

def _augment_gt_boxes_by_perturbation(unique_gt_boxes, im_width, im_height):
    num_gt = unique_gt_boxes.shape[0]
    num_rois = 30
    rois = np.zeros((num_rois, 4), dtype=np.float32)
    cnt = 0
    for i in range(num_gt):
        box = unique_gt_boxes[i]
        box_width = box[2] - box[0] + 1
        box_height = box[3] - box[1] + 1
        x_offset_max = (box_width - 1) // 2
        y_offset_max = (box_height - 1) // 2
        for _ in range(num_rois // num_gt):
            x_min_offset = np.random.uniform(low=-x_offset_max, high=x_offset_max)
            y_min_offset = np.random.uniform(low=-y_offset_max, high=y_offset_max)
            x_max_offset = np.random.uniform(low=-x_offset_max, high=x_offset_max)
            y_max_offset = np.random.uniform(low=-y_offset_max, high=y_offset_max)

            new_x_min = min(max(np.round(box[0] + x_min_offset), 0), im_width - 1)
            new_y_min = min(max(np.round(box[1] + y_min_offset), 0), im_height - 1)
            new_x_max = min(max(np.round(box[2] + x_max_offset), 0), im_width - 1)
            new_y_max = min(max(np.round(box[3] + y_max_offset), 0), im_height - 1)

            new_box = np.array(
                [new_x_min, new_y_min, new_x_max, new_y_max]).astype(np.float32)
            rois[cnt] = new_box
            cnt += 1

    return rois

def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


def get_obj_prd_gn_relco_vecs(dataset_name):
    gn_word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        cfg.DATA_DIR + '/word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)
    logger.info('Model loaded.')
    # change everything into lowercase
    all_keys = list(gn_word2vec_model.vocab.keys())
    for key in all_keys:
        new_key = key.lower()
        gn_word2vec_model.vocab[new_key] = gn_word2vec_model.vocab.pop(key)
    temp_dict = {x.replace('_', '-'): y for x, y in gn_word2vec_model.vocab.items()}
    gn_word2vec_model.vocab.update(temp_dict)
    logger.info('Wiki words converted to lowercase.')

    relco_word2vec_model = gensim.models.Word2Vec.load(
            cfg.DATA_DIR + '/word2vec_model/vg_300d_skipgram_rel')
    print('Model loaded.')
    relco_vec_mean = relco_word2vec_model.wv.syn0.mean(axis=0)
    # change everything into lowercase
    for key in relco_word2vec_model.wv.vocab.keys():
        new_key = key.lower()
        relco_word2vec_model.wv.vocab[new_key] = relco_word2vec_model.wv.vocab.pop(key)
    temp_dict = {x.replace('_', '-'): y for x, y in relco_word2vec_model.wv.vocab.items()}
    relco_word2vec_model.wv.vocab.update(temp_dict)

    print('Relco words converted to lowercase.')

    if dataset_name.find('vrd') >= 0:
        with open(cfg.DATA_DIR + '/vrd/objects.json') as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/vrd/predicates.json') as f:
            prd_cats = json.load(f)
    elif dataset_name.find('vg80k') >= 0:
        with open(cfg.DATA_DIR + '/vg80k/seed{}/objects.json'.format(cfg.RNG_SEED)) as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/vg80k/seed{}/predicates.json'.format(cfg.RNG_SEED)) as f:
            prd_cats = json.load(f)
    elif dataset_name.find('vg8k') >= 0:
        with open(cfg.DATA_DIR + '/vg8k/seed{}/objects.json'.format(cfg.RNG_SEED)) as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/vg8k/seed{}/predicates.json'.format(cfg.RNG_SEED)) as f:
            prd_cats = json.load(f)
    elif dataset_name.find('vg') >= 0:
        with open(cfg.DATA_DIR + '/vg/seed{}/objects.json'.format(cfg.RNG_SEED)) as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/vg/seed{}/predicates.json'.format(cfg.RNG_SEED)) as f:
            prd_cats = json.load(f)
    elif dataset_name.find('gvqa20k') >= 0:
        with open(cfg.DATA_DIR + '/gvqa20k/seed{}/objects.json'.format(cfg.RNG_SEED)) as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/gvqa20k/seed{}/predicates.json'.format(cfg.RNG_SEED)) as f:
            prd_cats = json.load(f)
    elif dataset_name.find('gvqa10k') >= 0:
        with open(cfg.DATA_DIR + '/gvqa10k/seed{}/objects.json'.format(cfg.RNG_SEED)) as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/gvqa10k/seed{}/predicates.json'.format(cfg.RNG_SEED)) as f:
            prd_cats = json.load(f)
    elif dataset_name.find('gvqa') >= 0:
        with open(cfg.DATA_DIR + '/gvqa/seed{}/objects.json'.format(cfg.RNG_SEED)) as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/gvqa/seed{}/predicates.json'.format(cfg.RNG_SEED)) as f:
            prd_cats = json.load(f)
    else:
        raise NotImplementedError
    # represent background with the word 'unknown'
    # obj_cats.insert(0, 'unknown')
    prd_cats.insert(0, 'unknown')
    all_obj_vecs = np.zeros((len(obj_cats), 600), dtype=np.float32)

    for r, obj_cat in enumerate(obj_cats):
        obj_vecs_gn = np.zeros(300, dtype=np.float32)
        obj_words = obj_cat.split()
        for word in obj_words:
            raw_vec = gn_word2vec_model[word]
            obj_vecs_gn += (raw_vec / la.norm(raw_vec))
        obj_vecs_gn /= len(obj_words)
        # obj_vecs_gn /= la.norm(obj_vecs_gn)

        all_obj_vecs[r][:300] = obj_vecs_gn

    for r, obj_cat in enumerate(obj_cats):
        obj_vecs_relco = np.zeros(300, dtype=np.float32)
        obj_words = obj_cat.split()
        for word in obj_words:
            if word in relco_word2vec_model.wv.vocab:
                raw_word = relco_word2vec_model[word]
                obj_vecs_relco += (raw_word / la.norm(raw_word))
            else:
                obj_vecs_relco += \
                    (relco_vec_mean / la.norm(relco_vec_mean))
        obj_vecs_relco /= len(obj_words)
        # obj_vecs_relco /= la.norm(obj_vecs_relco)

        all_obj_vecs[r][300:] = obj_vecs_relco

    logger.info('Object label vectors loaded.')

    all_prd_vecs = np.zeros((len(prd_cats), 600), dtype=np.float32)

    for r, prd_cat in enumerate(prd_cats):
        prd_vecs_gn = np.zeros(300, dtype=np.float32)
        prd_words = prd_cat.split()
        for word in prd_words:
            raw_vec = gn_word2vec_model[word]
            prd_vecs_gn += (raw_vec / la.norm(raw_vec))
        prd_vecs_gn /= len(prd_words)
        # prd_vecs_gn /= la.norm(prd_vecs_gn)

        all_prd_vecs[r][:300] = prd_vecs_gn

    for r, prd_cat in enumerate(prd_cats):
        prd_vecs_relco = np.zeros(300, dtype=np.float32)
        prd_words = prd_cat.split()
        for word in prd_words:
            if word in relco_word2vec_model.wv.vocab:
                raw_word = relco_word2vec_model[word]
                prd_vecs_relco += (raw_word / la.norm(raw_word))
            else:
                prd_vecs_relco += \
                    (relco_vec_mean / la.norm(relco_vec_mean))
        prd_vecs_relco /= len(prd_words)
        # prd_vecs_relco /= la.norm(prd_vecs_relco)

        all_prd_vecs[r][300:] = prd_vecs_relco

    logger.info('Predicate label vectors loaded.')
    return all_obj_vecs, all_prd_vecs, obj_cats, prd_cats


def get_obj_prd_vecs(dataset_name):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        cfg.DATA_DIR + '/word2vec_model/GoogleNews-vectors-negative300.bin', binary=True)
    logger.info('Model loaded.')
    # change everything into lowercase
    all_keys = list(word2vec_model.vocab.keys())
    for key in all_keys:
        new_key = key.lower()
        word2vec_model.vocab[new_key] = word2vec_model.vocab.pop(key)
    temp_dict = {x.replace('_', '-'): y for x, y in word2vec_model.vocab.items()}
    word2vec_model.vocab.update(temp_dict)
    logger.info('Wiki words converted to lowercase.')

    if dataset_name.find('vrd') >= 0:
        with open(cfg.DATA_DIR + '/vrd/objects.json') as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/vrd/predicates.json') as f:
            prd_cats = json.load(f)
    elif dataset_name.find('vg80k') >= 0:
        with open(cfg.DATA_DIR + '/vg80k/seed{}/objects.json'.format(cfg.RNG_SEED)) as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/vg80k/seed{}/predicates.json'.format(cfg.RNG_SEED)) as f:
            prd_cats = json.load(f)
    elif dataset_name.find('vg8k') >= 0:
        with open(cfg.DATA_DIR + '/vg8k/seed{}/objects.json'.format(cfg.RNG_SEED)) as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/vg8k/seed{}/predicates.json'.format(cfg.RNG_SEED)) as f:
            prd_cats = json.load(f)
    elif dataset_name.find('vg') >= 0:
        with open(cfg.DATA_DIR + '/vg/seed{}/objects.json'.format(cfg.RNG_SEED)) as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/vg/seed{}/predicates.json'.format(cfg.RNG_SEED)) as f:
            prd_cats = json.load(f)
    elif dataset_name.find('gvqa20k') >= 0:
        with open(cfg.DATA_DIR + '/gvqa20k/seed{}/objects.json'.format(cfg.RNG_SEED)) as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/gvqa20k/seed{}/predicates.json'.format(cfg.RNG_SEED)) as f:
            prd_cats = json.load(f)
    elif dataset_name.find('gvqa10k') >= 0:
        with open(cfg.DATA_DIR + '/gvqa10k/seed{}/objects.json'.format(cfg.RNG_SEED)) as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/gvqa10k/seed{}/predicates.json'.format(cfg.RNG_SEED)) as f:
            prd_cats = json.load(f)
    elif dataset_name.find('gvqa') >= 0:
        with open(cfg.DATA_DIR + '/gvqa/seed{}/objects.json'.format(cfg.RNG_SEED)) as f:
            obj_cats = json.load(f)
        with open(cfg.DATA_DIR + '/gvqa/seed{}/predicates.json'.format(cfg.RNG_SEED)) as f:
            prd_cats = json.load(f)
    else:
        raise NotImplementedError
    # represent background with the word 'unknown'
    # obj_cats.insert(0, 'unknown')
    prd_cats.insert(0, 'unknown')
    all_obj_vecs = np.zeros((len(obj_cats), 300), dtype=np.float32)
    for r, obj_cat in enumerate(obj_cats):
        obj_words = obj_cat.split()
        for word in obj_words:
            raw_vec = word2vec_model[word]
            all_obj_vecs[r] += (raw_vec / la.norm(raw_vec))
        all_obj_vecs[r] /= len(obj_words)
    logger.info('Object label vectors loaded.')
    all_prd_vecs = np.zeros((len(prd_cats), 300), dtype=np.float32)
    for r, prd_cat in enumerate(prd_cats):
        prd_words = prd_cat.split()
        for word in prd_words:
            raw_vec = word2vec_model[word]
            all_prd_vecs[r] += (raw_vec / la.norm(raw_vec))
        all_prd_vecs[r] /= len(prd_words)
    logger.info('Predicate label vectors loaded.')
    return all_obj_vecs, all_prd_vecs, obj_cats, prd_cats



def get_freq_from_dict(freq_dict, categories):
    freqs = np.zeros(len(categories))
    for i, cat in enumerate(categories):
        if cat in freq_dict.keys():
            freqs[i] = freq_dict[cat]
        else:
            freqs[i] = 0
    return freqs


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # Region Proposal Network
        if not cfg.TRAIN.USE_GT_BOXES and cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)
            
        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
            self.Conv_Body.dim_out, 4096, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            # self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
        if not cfg.TRAIN.USE_GT_BOXES:
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out)
            
        self.Prd_RCNN = copy.deepcopy(self)
        if not cfg.TRAIN.USE_GT_BOXES:
            del self.Prd_RCNN.RPN
            del self.Prd_RCNN.Box_Outs
        
        # initialize word vectors
        ds_name = cfg.TRAIN.DATASETS[0] if len(cfg.TRAIN.DATASETS) else cfg.TEST.DATASETS[0]

        if cfg.MODEL.INPUT_LANG_EMBEDDING_DIM == 300:
            self.obj_vecs, self.prd_vecs, obj_categories, prd_categories = get_obj_prd_vecs(ds_name)
        elif cfg.MODEL.INPUT_LANG_EMBEDDING_DIM == 600:
            self.obj_vecs, self.prd_vecs, obj_categories, prd_categories = get_obj_prd_gn_relco_vecs(ds_name)
        else:
            raise NotImplementedError

        # RelPN
        self.RelPN = relpn_heads.generic_relpn_outputs()
        # RelDN
        self.RelDN = reldn_heads.reldn_head(self.Box_Head.dim_out * 3, self.obj_vecs, self.prd_vecs)  # concat of SPO

        self.prd_weights = None
        self.obj_weights = None

        if cfg.DATASET == 'gvqa10k':
            freq_prd_path = cfg.DATA_DIR + '/gvqa/reduced_data/10k/seed{}/predicates_freqs.json'.format(
                cfg.RNG_SEED)
            freq_obj_path = cfg.DATA_DIR + '/gvqa/reduced_data/10k/seed{}/objects_freqs.json'.format(
                cfg.RNG_SEED)
        elif cfg.DATASET == 'gvqa20k':
            freq_prd_path = cfg.DATA_DIR + '/gvqa/reduced_data/20k/seed{}/predicates_freqs.json'.format(
                cfg.RNG_SEED)
            freq_obj_path = cfg.DATA_DIR + '/gvqa/reduced_data/20k/seed{}/objects_freqs.json'.format(
                cfg.RNG_SEED)
        elif cfg.DATASET == 'gvqa':
            freq_prd_path = cfg.DATA_DIR + '/gvqa/seed{}/predicates_freqs.json'.format(
                cfg.RNG_SEED)
            freq_obj_path = cfg.DATA_DIR + '/gvqa/seed{}/objects_freqs.json'.format(
                cfg.RNG_SEED)
        elif cfg.DATASET == 'vg80k':
            freq_prd_path = cfg.DATA_DIR + '/vg/predicates_freqs.json'
            freq_obj_path = cfg.DATA_DIR + '/vg/objects_freqs.json'
        elif cfg.DATASET == 'vg8k':
            freq_prd_path = cfg.DATA_DIR + '/vg8k/seed{}/train_predicates_freqs.json'.format(
                cfg.RNG_SEED)
            freq_obj_path = cfg.DATA_DIR + '/vg8k/seed{}/train_objects_freqs.json'.format(
                cfg.RNG_SEED)
        else:
            raise NotImplementedError

        self.prd_freq_dict = json.load(open(freq_prd_path))
        self.obj_freq_dict = json.load(open(freq_obj_path))

        no_bg_prd_categories = prd_categories[1:]

        assert len(no_bg_prd_categories) == cfg.MODEL.NUM_PRD_CLASSES

        self.prd_categories = no_bg_prd_categories
        self.obj_categories = obj_categories

        self.freq_prd = get_freq_from_dict(self.prd_freq_dict, self.prd_categories)
        self.freq_obj = get_freq_from_dict(self.obj_freq_dict, self.obj_categories)

        # if cfg.MODEL.LOSS == 'weighted_cross_entropy' or cfg.MODEL.LOSS == 'weighted_focal':
        logger.info('loading frequencies')

        freq_prd = self.freq_prd + 1
        freq_obj = self.freq_obj + 1
        prd_weights = np.sum(freq_prd) / freq_prd
        obj_weights = np.sum(freq_obj) / freq_obj
        ## More weight given to the classes belonging to few classes spectrum

        self.prd_weights = (prd_weights / np.mean(prd_weights)).astype(np.float32)
        self.obj_weights = (obj_weights / np.mean(obj_weights)).astype(np.float32)
        temp = np.zeros(shape=self.prd_weights.shape[0] + 1, dtype=np.float32) 
        temp[1:] = self.prd_weights
        temp[0] = min(self.prd_weights)
        self.prd_weights = temp

        self._init_modules()

    def _init_modules(self):
        # VGG16 imagenet pretrained model is initialized in VGG16.py
        if cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS != '':
            logger.info("Loading pretrained weights from %s", cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS)
            resnet_utils.load_pretrained_imagenet_weights(self)
                
        if cfg.RESNETS.VRD_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.VRD_PRETRAINED_WEIGHTS)
        if cfg.VGG16.VRD_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.VGG16.VRD_PRETRAINED_WEIGHTS)
            
        if cfg.RESNETS.VG_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.RESNETS.VG_PRETRAINED_WEIGHTS)
        if cfg.VGG16.VG_PRETRAINED_WEIGHTS != '':
            self.load_detector_weights(cfg.VGG16.VG_PRETRAINED_WEIGHTS)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

        if cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS != '' or \
            cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS != '' or cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS != '':
            if cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.RESNETS.VRD_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.VGG16.VRD_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.RESNETS.VG_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            if cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS != '':
                logger.info("loading prd pretrained weights from %s", cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS)
                checkpoint = torch.load(cfg.VGG16.VG_PRD_PRETRAINED_WEIGHTS, map_location=lambda storage, loc: storage)
            # not using the last softmax layers
            del checkpoint['model']['Box_Outs.cls_score.weight']
            del checkpoint['model']['Box_Outs.cls_score.bias']
            del checkpoint['model']['Box_Outs.bbox_pred.weight']
            del checkpoint['model']['Box_Outs.bbox_pred.bias']
            net_utils.load_ckpt(self.Prd_RCNN, checkpoint['model'])
            if cfg.TRAIN.FREEZE_PRD_CONV_BODY:
                for p in self.Prd_RCNN.Conv_Body.parameters():
                    p.requires_grad = False
            if cfg.TRAIN.FREEZE_PRD_BOX_HEAD:
                for p in self.Prd_RCNN.Box_Head.parameters():
                    p.requires_grad = False
    
    def load_detector_weights(self, weight_name):
        logger.info("loading pretrained weights from %s", weight_name)
        checkpoint = torch.load(weight_name, map_location=lambda storage, loc: storage)
        if not cfg.VGG16.INCLUDE_CLASSIFIER or cfg.TRAIN.USE_GT_BOXES:
            del checkpoint['model']['Box_Outs.cls_score.weight']
            del checkpoint['model']['Box_Outs.cls_score.bias']
            del checkpoint['model']['Box_Outs.bbox_pred.weight']
            del checkpoint['model']['Box_Outs.bbox_pred.bias']
        net_utils.load_ckpt(self, checkpoint['model'])
        # freeze everything above the rel module
        for p in self.Conv_Body.parameters():
            p.requires_grad = False
        if not cfg.TRAIN.USE_GT_BOXES:
            for p in self.RPN.parameters():
                p.requires_grad = False
        if not cfg.MODEL.UNFREEZE_DET:
            for p in self.Box_Head.parameters():
                p.requires_grad = False
            if cfg.VGG16.INCLUDE_CLASSIFIER and not cfg.TRAIN.USE_GT_BOXES:
                for p in self.Box_Outs.parameters():
                    p.requires_grad = False


    def forward(self, data, im_info, dataset_name=None, roidb=None, use_gt_labels=False, include_feat=False, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, dataset_name, roidb, use_gt_labels, include_feat, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, dataset_name, roidb, use_gt_labels, include_feat, **rpn_kwargs)

    def _forward(self, data, im_info, dataset_name=None, roidb=None, use_gt_labels=False, include_feat=False,  **rpn_kwargs):
        im_data = data
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))
        if dataset_name is not None:
            dataset_name = blob_utils.deserialize(dataset_name)
        else:
            dataset_name = cfg.TRAIN.DATASETS[0] if self.training else cfg.TEST.DATASETS[0]  # assuming only one dataset per run
            

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables


        blob_conv = self.Conv_Body(im_data)
        blob_conv_prd = self.Prd_RCNN.Conv_Body(im_data)

        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
            blob_conv_prd = blob_conv_prd[-self.num_roi_levels:]

        if not cfg.TRAIN.USE_GT_BOXES:
            rpn_ret = self.RPN(blob_conv, im_info, roidb)

            if cfg.MODEL.SHARE_RES5 and self.training:
                box_feat, res5_feat = self.Box_Head(blob_conv, rpn_ret, use_relu=True)
            else:
                box_feat = self.Box_Head(blob_conv, rpn_ret, use_relu=True)
            cls_score, bbox_pred = self.Box_Outs(box_feat)
        
        # now go through the predicate branch
        use_relu = False if cfg.MODEL.NO_FC7_RELU else True
        if self.training:
            if cfg.TRAIN.USE_GT_BOXES:
                # we always feed one image per batch during training
                assert len(roidb) == 1
                im_scale = im_info.data.numpy()[:, 2][0]
                im_w = im_info.data.numpy()[:, 1][0]
                im_h = im_info.data.numpy()[:, 0][0]
                sbj_boxes = roidb[0]['sbj_gt_boxes']
                obj_boxes = roidb[0]['obj_gt_boxes']
                sbj_all_boxes = _augment_gt_boxes_by_perturbation(sbj_boxes, im_w, im_h)
                obj_all_boxes = _augment_gt_boxes_by_perturbation(obj_boxes, im_w, im_h)
                det_all_boxes = np.vstack((sbj_all_boxes, obj_all_boxes))
                det_all_boxes = np.unique(det_all_boxes, axis=0)
                det_all_rois = det_all_boxes * im_scale
                repeated_batch_idx = 0 * blob_utils.ones((det_all_rois.shape[0], 1))
                det_all_rois = np.hstack((repeated_batch_idx, det_all_rois))
                rel_ret = self.RelPN(det_all_rois, None, None, im_info, dataset_name, roidb)
            else:
                fg_inds = np.where(rpn_ret['labels_int32'] > 0)[0]
                det_rois = rpn_ret['rois'][fg_inds]
                det_labels = rpn_ret['labels_int32'][fg_inds]
                det_scores = F.softmax(cls_score[fg_inds], dim=1)
                rel_ret = self.RelPN(det_rois, det_labels, det_scores, im_info, dataset_name, roidb)
            sbj_feat = self.Box_Head(blob_conv, rel_ret, rois_name='sbj_rois', use_relu=use_relu)
            # print('!! SBJ_FEAT SHAPE: !! ', sbj_feat.shape)    #### (_, 4096)
            obj_feat = self.Box_Head(blob_conv, rel_ret, rois_name='obj_rois', use_relu=use_relu)
        else:
            if roidb is not None:
                im_scale = im_info.data.numpy()[:, 2][0]
                im_w = im_info.data.numpy()[:, 1][0]
                im_h = im_info.data.numpy()[:, 0][0]
                sbj_boxes = roidb['sbj_gt_boxes']
                obj_boxes = roidb['obj_gt_boxes']
                sbj_rois = sbj_boxes * im_scale
                obj_rois = obj_boxes * im_scale
                repeated_batch_idx = 0 * blob_utils.ones((sbj_rois.shape[0], 1))
                sbj_rois = np.hstack((repeated_batch_idx, sbj_rois))
                obj_rois = np.hstack((repeated_batch_idx, obj_rois))
                rel_rois = box_utils.rois_union(sbj_rois, obj_rois)
                rel_ret = {}
                rel_ret['sbj_rois'] = sbj_rois
                rel_ret['obj_rois'] = obj_rois
                rel_ret['rel_rois'] = rel_rois
                if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
                    lvl_min = cfg.FPN.ROI_MIN_LEVEL
                    lvl_max = cfg.FPN.ROI_MAX_LEVEL
                    rois_blob_names = ['sbj_rois', 'obj_rois', 'rel_rois']
                    for rois_blob_name in rois_blob_names:
                        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
                        target_lvls = fpn_utils.map_rois_to_fpn_levels(
                            rel_ret[rois_blob_name][:, 1:5], lvl_min, lvl_max)
                        fpn_utils.add_multilevel_roi_blobs(
                            rel_ret, rois_blob_name, rel_ret[rois_blob_name], target_lvls,
                            lvl_min, lvl_max)
                if use_gt_labels:
                    sbj_labels = roidb['sbj_gt_classes']  # start from 0
                    obj_labels = roidb['obj_gt_classes']  # start from 0
                    sbj_scores = np.ones_like(sbj_labels, dtype=np.float32)
                    obj_scores = np.ones_like(obj_labels, dtype=np.float32)
                else:
                    sbj_det_feat = self.Box_Head(blob_conv, rel_ret, rois_name='sbj_rois', use_relu=True)
                    sbj_cls_scores, _ = self.Box_Outs(sbj_det_feat)
                    sbj_cls_scores = sbj_cls_scores.data.cpu().numpy()
                    obj_det_feat = self.Box_Head(blob_conv, rel_ret, rois_name='obj_rois', use_relu=True)
                    obj_cls_scores, _ = self.Box_Outs(obj_det_feat)
                    obj_cls_scores = obj_cls_scores.data.cpu().numpy()
                    sbj_labels = np.argmax(sbj_cls_scores[:, 1:], axis=1)
                    obj_labels = np.argmax(obj_cls_scores[:, 1:], axis=1)
                    sbj_scores = np.amax(sbj_cls_scores[:, 1:], axis=1)
                    obj_scores = np.amax(obj_cls_scores[:, 1:], axis=1)
                rel_ret['sbj_scores'] = sbj_scores.astype(np.float32, copy=False)
                rel_ret['obj_scores'] = obj_scores.astype(np.float32, copy=False)
                rel_ret['sbj_labels'] = sbj_labels.astype(np.int32, copy=False) + 1  # need to start from 1
                rel_ret['obj_labels'] = obj_labels.astype(np.int32, copy=False) + 1  # need to start from 1
                rel_ret['all_sbj_labels_int32'] = sbj_labels.astype(np.int32, copy=False)
                rel_ret['all_obj_labels_int32'] = obj_labels.astype(np.int32, copy=False)
                sbj_feat = self.Box_Head(blob_conv, rel_ret, rois_name='sbj_rois', use_relu=use_relu)
                obj_feat = self.Box_Head(blob_conv, rel_ret, rois_name='obj_rois', use_relu=use_relu)
            else:
                score_thresh = cfg.TEST.SCORE_THRESH
                while score_thresh >= -1e-06:  # a negative value very close to 0.0
                    det_rois, det_labels, det_scores = \
                        self.prepare_det_rois(rpn_ret['rois'], cls_score, bbox_pred, im_info, score_thresh)
                    rel_ret = self.RelPN(det_rois, det_labels, det_scores, im_info, dataset_name, roidb)
                    valid_len = len(rel_ret['rel_rois'])
                    if valid_len > 0:
                        break
                    logger.info('Got {} rel_rois when score_thresh={}, changing to {}'.format(
                        valid_len, score_thresh, score_thresh - 0.01))
                    score_thresh -= 0.01
                det_feat = self.Box_Head(blob_conv, rel_ret, rois_name='det_rois', use_relu=use_relu)
                sbj_feat = det_feat[rel_ret['sbj_inds']]
                obj_feat = det_feat[rel_ret['obj_inds']]

        rel_feat = self.Prd_RCNN.Box_Head(blob_conv_prd, rel_ret, rois_name='rel_rois', use_relu=use_relu)

        concat_feat = torch.cat((sbj_feat, rel_feat, obj_feat), dim=1)
        # print('!! Concat feat shape: !! ', concat_feat.shape)

        if cfg.MODEL.USE_FREQ_BIAS or cfg.MODEL.RUN_BASELINE or cfg.MODEL.USE_SEM_CONCAT:
            sbj_labels = rel_ret['all_sbj_labels_int32']
            obj_labels = rel_ret['all_obj_labels_int32']
        else:
            sbj_labels = None
            obj_labels = None
        
        # when MODEL.USE_SEM_CONCAT, memory runs out if the whole batch is fed once
        # so we need to feed the batch twice if it's big
        gn_size = 1000
        if cfg.MODEL.USE_SEM_CONCAT and concat_feat.shape[0] > gn_size:
            group = int(math.ceil(concat_feat.shape[0] / gn_size))
            prd_cls_scores = None
            sbj_cls_scores = None
            obj_cls_scores = None
            for i in range(group):
                end = int(min((i + 1) * gn_size, concat_feat.shape[0]))
                concat_feat_i = concat_feat[i * gn_size : end]
                sbj_labels_i = sbj_labels[i * gn_size : end] if sbj_labels is not None else None
                obj_labels_i = obj_labels[i * gn_size : end] if obj_labels is not None else None
                
                sbj_feat_i = sbj_feat[i * gn_size : end]
                obj_feat_i = obj_feat[i * gn_size : end]
                prd_cls_scores_i, sbj_cls_scores_i, obj_cls_scores_i = \
                    self.RelDN(concat_feat_i, sbj_labels_i, obj_labels_i, sbj_feat_i, obj_feat_i)
                if prd_cls_scores is None:
                    prd_cls_scores = prd_cls_scores_i
                    sbj_cls_scores = sbj_cls_scores_i
                    obj_cls_scores = obj_cls_scores_i
                else:
                    prd_cls_scores = torch.cat((prd_cls_scores, prd_cls_scores_i))
                    sbj_cls_scores = torch.cat((sbj_cls_scores, sbj_cls_scores_i)) if sbj_cls_scores_i is not None else sbj_cls_scores
                    obj_cls_scores = torch.cat((obj_cls_scores, obj_cls_scores_i)) if obj_cls_scores_i is not None else obj_cls_scores
        else:
            if self.training:
                prd_cls_scores, sbj_cls_scores, obj_cls_scores, mixed_sbj_cls_scores, mixed_obj_cls_scores, mixed_prd_cls_scores, mixed_sbj_labels, mixed_obj_labels, mixed_prd_labels = \
                        self.RelDN(concat_feat, self.prd_weights, rel_ret['all_sbj_labels_int32'], rel_ret['all_obj_labels_int32'], sbj_feat, obj_feat, rel_ret['all_prd_labels_int32'])
            else:
                prd_cls_scores, sbj_cls_scores, obj_cls_scores, mixed_sbj_cls_scores, mixed_obj_cls_scores, mixed_prd_cls_scores, mixed_sbj_labels, mixed_obj_labels, mixed_prd_labels = \
                        self.RelDN(concat_feat, self.prd_weights, rel_ret['all_sbj_labels_int32'], rel_ret['all_obj_labels_int32'], sbj_feat, obj_feat)

        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            if not cfg.TRAIN.USE_GT_BOXES:
                # rpn loss
                rpn_kwargs.update(dict(
                    (k, rpn_ret[k]) for k in rpn_ret.keys()
                    if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
                ))
                loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
                if cfg.FPN.FPN_ON:
                    for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                        return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                        return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
                else:
                    return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                    return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox
                # bbox loss
                loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                    cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                    rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
                return_dict['losses']['loss_cls'] = loss_cls
                return_dict['losses']['loss_bbox'] = loss_bbox
                return_dict['metrics']['accuracy_cls'] = accuracy_cls

            if cfg.MODEL.LOSS == 'eql':
                loss_cls_prd, accuracy_cls_prd = reldn_heads.eql_loss_prd(
                    prd_cls_scores, rel_ret['all_prd_labels_int32'], self.prd_weights)
            else:
                loss_cls_prd, accuracy_cls_prd = reldn_heads.reldn_losses(
                    prd_cls_scores, rel_ret['all_prd_labels_int32'], weight=self.prd_weights)
            return_dict['losses']['loss_cls_prd'] = loss_cls_prd
            return_dict['metrics']['accuracy_cls_prd'] = accuracy_cls_prd

            if cfg.cumix:
                return_dict['losses']['loss_cls_prd'] += reldn_heads.manual_CE(mixed_prd_cls_scores, mixed_prd_labels)

            if cfg.MODEL.USE_SEPARATE_SO_SCORES:
                if cfg.MODEL.LOSS == 'eql':
                    loss_cls_sbj, accuracy_cls_sbj = reldn_heads.eql_loss_so(
                        sbj_cls_scores, rel_ret['all_sbj_labels_int32'], self.obj_weights)
                else:
                    loss_cls_sbj, accuracy_cls_sbj = reldn_heads.reldn_losses(
                        sbj_cls_scores, rel_ret['all_sbj_labels_int32'], weight=self.obj_weights)
                return_dict['losses']['loss_cls_sbj'] = loss_cls_sbj
                return_dict['metrics']['accuracy_cls_sbj'] = accuracy_cls_sbj

                if cfg.MODEL.LOSS == 'eql':
                    loss_cls_obj, accuracy_cls_obj = reldn_heads.eql_loss_so(
                        obj_cls_scores, rel_ret['all_obj_labels_int32'], self.obj_weights)
                else:
                    loss_cls_obj, accuracy_cls_obj = reldn_heads.reldn_losses(
                        obj_cls_scores, rel_ret['all_obj_labels_int32'], weight=self.obj_weights)
                return_dict['losses']['loss_cls_obj'] = loss_cls_obj
                return_dict['metrics']['accuracy_cls_obj'] = accuracy_cls_obj

                if cfg.cumix:
                    return_dict['losses']['loss_cls_sbj'] += reldn_heads.manual_CE(mixed_sbj_cls_scores, mixed_sbj_labels)
                    return_dict['losses']['loss_cls_obj'] += reldn_heads.manual_CE(mixed_obj_cls_scores, mixed_obj_labels)

            if cfg.TRAIN.HUBNESS:
                if cfg.TRAIN.HUB_REL_ONLY:
                    loss_hubness_prd = reldn_heads.add_hubness_loss(prd_cls_scores)
                    return_dict['losses']['loss_hubness_prd'] = loss_hubness_prd
                else:
                    loss_hubness_prd = reldn_heads.add_hubness_loss(prd_cls_scores)
                    loss_hubness_sbj = reldn_heads.add_hubness_loss(sbj_cls_scores)
                    loss_hubness_obj = reldn_heads.add_hubness_loss(obj_cls_scores)
                    return_dict['losses']['loss_hubness_prd'] = loss_hubness_prd
                    return_dict['losses']['loss_hubness_sbj'] = loss_hubness_sbj
                    return_dict['losses']['loss_hubness_obj'] = loss_hubness_obj

                    if cfg.cumix:
                        return_dict['losses']['loss_hubness_prd'] += reldn_heads.add_hubness_loss(mixed_prd_cls_scores)
                        return_dict['losses']['loss_hubness_sbj'] += reldn_heads.add_hubness_loss(mixed_sbj_cls_scores)
                        return_dict['losses']['loss_hubness_obj'] += reldn_heads.add_hubness_loss(mixed_obj_cls_scores)



            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)
        else:
            # Testing
            return_dict['sbj_rois'] = rel_ret['sbj_rois']
            return_dict['obj_rois'] = rel_ret['obj_rois']
            return_dict['sbj_labels'] = rel_ret['sbj_labels']
            return_dict['obj_labels'] = rel_ret['obj_labels']
            return_dict['sbj_scores'] = rel_ret['sbj_scores']
            return_dict['sbj_scores_out'] = sbj_cls_scores
            return_dict['obj_scores'] = rel_ret['obj_scores']
            return_dict['obj_scores_out'] = obj_cls_scores
            return_dict['prd_scores'] = prd_cls_scores
            if include_feat:
                return_dict['sbj_feat'] = sbj_feat
                return_dict['obj_feat'] = obj_feat
                return_dict['prd_feat'] = concat_feat

        return return_dict
    
    def get_roi_inds(self, det_labels, lbls):
        lbl_set = np.array(lbls)
        inds = np.where(np.isin(det_labels, lbl_set))[0]
        return inds
    
    def prepare_det_rois(self, rois, cls_scores, bbox_pred, im_info, score_thresh=cfg.TEST.SCORE_THRESH):
        im_info = im_info.data.cpu().numpy()
        # NOTE: 'rois' is numpy array while
        # 'cls_scores' and 'bbox_pred' are pytorch tensors
        scores = cls_scores.data.cpu().numpy().squeeze()
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy().squeeze()
        
        assert rois.shape[0] == scores.shape[0] == box_deltas.shape[0]
        
        det_rois = np.empty((0, 5), dtype=np.float32)
        det_labels = np.empty((0), dtype=np.float32)
        det_scores = np.empty((0), dtype=np.float32)
        for im_i in range(cfg.TRAIN.IMS_PER_BATCH):
            # get all boxes that belong to this image
            inds = np.where(abs(rois[:, 0] - im_i) < 1e-06)[0]
            # unscale back to raw image space
            im_boxes = rois[inds, 1:5] / im_info[im_i, 2]
            im_scores = scores[inds]
            # In case there is 1 proposal
            im_scores = im_scores.reshape([-1, im_scores.shape[-1]])
            # In case there is 1 proposal
            im_box_deltas = box_deltas[inds]
            im_box_deltas = im_box_deltas.reshape([-1, im_box_deltas[inds].shape[-1]])

            im_scores, im_boxes = self.get_det_boxes(im_boxes, im_scores, im_box_deltas, im_info[im_i][:2] / im_info[im_i][2])
            im_scores, im_boxes, im_labels = self.box_results_with_nms_and_limit(im_scores, im_boxes, score_thresh)
            
            batch_inds = im_i * np.ones(
                (im_boxes.shape[0], 1), dtype=np.float32)
            im_det_rois = np.hstack((batch_inds, im_boxes * im_info[im_i, 2]))
            det_rois = np.append(det_rois, im_det_rois, axis=0)
            det_labels = np.append(det_labels, im_labels, axis=0)
            det_scores = np.append(det_scores, im_scores, axis=0)
        
        return det_rois, det_labels, det_scores

    def get_det_boxes(self, boxes, scores, box_deltas, h_and_w):

        if cfg.TEST.BBOX_REG:
            if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
                # Remove predictions for bg class (compat with MSRA code)
                box_deltas = box_deltas[:, -4:]
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # (legacy) Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * cfg.TRAIN.BBOX_NORMALIZE_STDS \
                             + cfg.TRAIN.BBOX_NORMALIZE_MEANS
            pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
            pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, h_and_w)
            if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
                pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
            # Map scores and predictions back to the original set of boxes
            scores = scores[inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]
            
        return scores, pred_boxes
    
    def box_results_with_nms_and_limit(self, scores, boxes, score_thresh=cfg.TEST.SCORE_THRESH):
        num_classes = cfg.MODEL.NUM_CLASSES
        cls_boxes = [[] for _ in range(num_classes)]
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > score_thresh)[0]
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4:(j + 1) * 4]
            dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
            if cfg.TEST.SOFT_NMS.ENABLED:
                nms_dets, _ = box_utils.soft_nms(
                    dets_j,
                    sigma=cfg.TEST.SOFT_NMS.SIGMA,
                    overlap_thresh=cfg.TEST.NMS,
                    score_thresh=0.0001,
                    method=cfg.TEST.SOFT_NMS.METHOD
                )
            else:
                keep = box_utils.nms(dets_j, cfg.TEST.NMS)
                nms_dets = dets_j[keep, :]
            # add labels
            label_j = np.ones((nms_dets.shape[0], 1), dtype=np.float32) * j
            nms_dets = np.hstack((nms_dets, label_j))
            # Refine the post-NMS boxes using bounding-box voting
            if cfg.TEST.BBOX_VOTE.ENABLED:
                nms_dets = box_utils.box_voting(
                    nms_dets,
                    dets_j,
                    cfg.TEST.BBOX_VOTE.VOTE_TH,
                    scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
                )
            cls_boxes[j] = nms_dets

        # Limit to max_per_image detections **over all classes**
        if cfg.TEST.DETECTIONS_PER_IM > 0:
            image_scores = np.hstack(
                [cls_boxes[j][:, -2] for j in range(1, num_classes)]
            )
            if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
                image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
                for j in range(1, num_classes):
                    keep = np.where(cls_boxes[j][:, -2] >= image_thresh)[0]
                    cls_boxes[j] = cls_boxes[j][keep, :]

        im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
        boxes = im_results[:, :-2]
        scores = im_results[:, -2]
        labels = im_results[:, -1]

        return scores, boxes, labels

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = RoIPoolFunction(resolution, resolution, sc)(bl_in, rois)
                    elif method == 'RoICrop':
                        # Warning!: Not check if implementation matches Detectron
                        grid_xy = net_utils.affine_grid_gen(
                            rois, bl_in.size()[2:], self.grid_size)
                        grid_yx = torch.stack(
                            [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                        xform_out = RoICropFunction()(bl_in, Variable(grid_yx).detach())
                        if cfg.CROP_RESIZE_WITH_MAX_POOL:
                            xform_out = F.max_pool2d(xform_out, 2, 2)
                    elif method == 'RoIAlign':
                        xform_out = RoIAlignFunction(
                            resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
            elif method == 'RoICrop':
                grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    xform_out = F.max_pool2d(xform_out, 2, 2)
            elif method == 'RoIAlign':
                xform_out = RoIAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
        return blob_conv

    @check_inference
    def mask_net(self, blob_conv, rpn_blob):
        """For inference"""
        mask_feat = self.Mask_Head(blob_conv, rpn_blob)
        mask_pred = self.Mask_Outs(mask_feat)
        return mask_pred

    @check_inference
    def keypoint_net(self, blob_conv, rpn_blob):
        """For inference"""
        kps_feat = self.Keypoint_Head(blob_conv, rpn_blob)
        kps_pred = self.Keypoint_Outs(kps_feat)
        return kps_pred

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
