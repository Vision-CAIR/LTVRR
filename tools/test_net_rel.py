"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time
from six.moves import cPickle as pickle

import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
#from core.test_engine_rel import run_inference, get_features_for_centroids

import utils.logging
from datasets import task_evaluation_rel as task_evaluation
from evaluation.generate_detections_csv import generate_csv_file_from_det_obj, generate_topk_csv_from_det_obj, generate_boxes_csv_from_det_obj
from evaluation.frequency_based_analysis_of_methods import get_metrics_from_csv, get_many_medium_few_scores, get_wordsim_metrics_from_csv

import numpy as np
import json

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')
    parser.add_argument(
        '--do_val', dest='do_val', help='do evaluation', action='store_true')
    parser.add_argument(
        '--use_gt_boxes', dest='use_gt_boxes', help='use gt boxes for sgcls/prdcls', action='store_true')
    parser.add_argument(
        '--use_gt_labels', dest='use_gt_labels', help='use gt boxes for sgcls/prdcls', action='store_true')

    parser.add_argument(
        '--cutoff_medium', dest='cutoff_medium', help='ratio of medium classes', type=float, default=0.80)

    parser.add_argument(
        '--cutoff_many', dest='cutoff_many', help='ratio of many classes', type=float, default=0.95)

    parser.add_argument(
        '--seed', dest='seed',
        help='Value of seed here will overwrite seed in cfg file',
        type=int)

    return parser.parse_args()


def get_obj_and_prd_categories():
    from datasets.dataset_catalog_rel import ANN_FN3
    from datasets.dataset_catalog_rel import DATASETS

    predicates_path = DATASETS[cfg.TEST.DATASETS[0]][ANN_FN3]
    objects_path = DATASETS[cfg.TEST.DATASETS[0]][ANN_FN3].replace('predicates', 'objects', 1)
    logger.info('Loading predicates from: ' + predicates_path)
    logger.info('Loading objects from: ' + objects_path)
    with open(predicates_path) as f:
        prd_categories = json.load(f)
    with open(objects_path) as f:
        obj_categories = json.load(f)
    return obj_categories, prd_categories


def get_obj_and_prd_frequencies():
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

    logger.info('Loading predicates frequencies from: ' + freq_prd_path)
    logger.info('Loading objects frequencies from: ' + freq_obj_path)

    prd_freq_dict = json.load(open(freq_prd_path))
    obj_freq_dict = json.load(open(freq_obj_path))

    return obj_freq_dict, prd_freq_dict

if __name__ == '__main__':
    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)


    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    cfg.DATASET = args.dataset
    if args.dataset == "vrd":
        cfg.TEST.DATASETS = ('vrd_val',)
        cfg.MODEL.NUM_CLASSES = 101
        cfg.MODEL.NUM_PRD_CLASSES = 70  # exclude background
    elif args.dataset == "vg":
        cfg.TEST.DATASETS = ('vg_val',)
        cfg.MODEL.NUM_CLASSES = 151
        cfg.MODEL.NUM_PRD_CLASSES = 50  # exclude background
    elif args.dataset == "vg80k":
        cfg.TEST.DATASETS = ('vg80k_test',)
        cfg.MODEL.NUM_CLASSES = 53305 # includes background
        cfg.MODEL.NUM_PRD_CLASSES = 29086  # excludes background
    elif args.dataset == "vg8k":
        cfg.TEST.DATASETS = ('vg8k_test',)
        cfg.MODEL.NUM_CLASSES = 5331 # includes background
        cfg.MODEL.NUM_PRD_CLASSES = 2000  # excludes background
    elif args.dataset == "gvqa20k":
        cfg.TEST.DATASETS = ('gvqa20k_test',)
        cfg.MODEL.NUM_CLASSES = 1704 # includes background
        cfg.MODEL.NUM_PRD_CLASSES = 310  # exclude background
    elif args.dataset == "gvqa10k":
        cfg.TEST.DATASETS = ('gvqa10k_test',)
        cfg.MODEL.NUM_CLASSES = 1704 # includes background
        cfg.MODEL.NUM_PRD_CLASSES = 310  # exclude background
    elif args.dataset == "gvqa":
        cfg.TEST.DATASETS = ('gvqa_test',)
        cfg.MODEL.NUM_CLASSES = 1704 # includes background
        cfg.MODEL.NUM_PRD_CLASSES = 310  # exclude background
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    if args.seed:
        cfg.RNG_SEED = args.seed
    assert_and_infer_cfg()

    data_dir = '{}/{}/'.format(cfg.DATA_DIR, cfg.DATASET)
    ann_dir = '{}seed{}/'.format(data_dir, cfg.RNG_SEED)
    # The import has to happen after setting up the config to avoid loading default cfg values
    from core.test_engine_rel import run_inference

    obj_categories, prd_categories = get_obj_and_prd_categories()
    obj_freq_dict, prd_freq_dict = get_obj_and_prd_frequencies()

    if not cfg.MODEL.RUN_BASELINE:
        assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
            'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))
    
    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True
    #print('Generating Centroids')
    #all_results = get_features_for_centroids(args) 
    #print('Done!')
    if args.use_gt_boxes:
        if args.use_gt_labels:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_prdcls.pkl')
            csv_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_prdcls.csv')
        else:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_sgcls.pkl')
            csv_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_sgcls.csv')
    else:
        det_file = os.path.join(args.output_dir, 'rel_detections.pkl')
        csv_file = os.path.join(args.output_dir, 'rel_detections.csv')
    if os.path.exists(det_file):
        logger.info('Loading results from {}'.format(det_file))
        with open(det_file, 'rb') as f:
            all_results = pickle.load(f)
        # logger.info('Starting evaluation now...')
        # task_evaluation.eval_rel_results(all_results, args.output_dir, args.do_val)
    else:
        if not torch.cuda.is_available():
            sys.exit("Need a CUDA device to run the code.")
        assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)
        all_results = run_inference(
                        args,
                        ind_range=args.range,
                        multi_gpu_testing=args.multi_gpu_testing,
                        check_expected_results=True)
        all_results = all_results[0]
    print('all_results', len(all_results))
    print('all_results', all_results[0].keys())
    #all_results = all_results[0]
    freq_prd = (np.zeros(cfg.MODEL.NUM_PRD_CLASSES))
    freq_obj = (np.zeros(cfg.MODEL.NUM_CLASSES))
    generate_csv_file_from_det_obj(all_results, csv_file, obj_categories, prd_categories, obj_freq_dict, prd_freq_dict)
    logger.info('Saved CSV to: ' + csv_file)
    get_metrics_from_csv(csv_file, get_mr=True)

    cutoffs = [args.cutoff_medium, args.cutoff_many]
    get_many_medium_few_scores(csv_file, cutoffs, cfg.DATASET, data_dir, ann_dir, syn=True)

    csv_file_topk = os.path.join(os.path.dirname(csv_file), 'rel_detections_gt_boxes_prdcls_topk.csv')
    generate_topk_csv_from_det_obj(all_results, csv_file_topk, obj_categories, prd_categories, 250)
    logger.info('Saved topk CSV to: ' + csv_file_topk)

    csv_file_boxes = os.path.join(os.path.dirname(csv_file), 'rel_detections_gt_boxes_prdcls_boxes.csv')
    generate_boxes_csv_from_det_obj(all_results, csv_file_boxes, obj_categories, prd_categories, obj_freq_dict, prd_freq_dict)
    logger.info('Saved boxes CSV to: ' + csv_file_boxes)

    if cfg.DATASET.find('gvqa') >= 0:
        from evaluation.add_word_similarity_to_csv import add_similarity_to_detections
        logger.info('Adding word similarity to CSV')
        add_similarity_to_detections(csv_file)
        csv_file_w = os.path.join(os.path.dirname(csv_file), 'rel_detections_gt_boxes_prdcls_wrd_sim.csv')
        get_wordsim_metrics_from_csv(csv_file_w)
