# Written by Sherif Abdelkarim on Feb 2020

import argparse
import pandas as pd
import os
from core.config import cfg

sim_path = cfg.DATA_DIR + 'gvqa/similarity_matrices/'
lch = pd.read_pickle(sim_path + 'obj_sim_lch.pkl')
wup = pd.read_pickle(sim_path + 'obj_sim_wup.pkl')
res = pd.read_pickle(sim_path + 'obj_sim_res.pkl')
jcn = pd.read_pickle(sim_path + 'obj_sim_jcn.pkl')
lin = pd.read_pickle(sim_path + 'obj_sim_lin_norm.pkl')
path = pd.read_pickle(sim_path + 'obj_sim_path_norm.pkl')

obj_w2v = pd.read_pickle(sim_path + 'obj_sim_w2v_gn_norm.pkl')
prd_w2v = pd.read_pickle(sim_path + 'prd_sim_w2v_gn_norm.pkl')


def add_similarity_to_detections(csv_file):
    df = pd.read_csv(csv_file)

    for label in ['sbj', 'obj']:
        df[label + '_lch'] = lch.lookup(df['gt_' + label], df['det_' + label])
        df[label + '_wup'] = wup.lookup(df['gt_' + label], df['det_' + label])
        df[label + '_res'] = res.lookup(df['gt_' + label], df['det_' + label])
        df[label + '_jcn'] = jcn.lookup(df['gt_' + label], df['det_' + label])
        df[label + '_lin'] = lin.lookup(df['gt_' + label], df['det_' + label])
        df[label + '_path'] = path.lookup(df['gt_' + label], df['det_' + label])
        df[label + '_w2v_gn'] = obj_w2v.lookup(df['gt_' + label], df['det_' + label])

    for label in ['rel']:
        df[label + '_w2v_gn'] = prd_w2v.lookup(df['gt_' + label], df['det_' + label])

    out_dir = os.path.dirname(csv_file)
    df.to_csv(os.path.join(out_dir, 'rel_detections_gt_boxes_prdcls_wrd_sim.csv'))


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')
    parser.add_argument(
        '--csv_file', dest='csv_file', required=True,
        help='Detections CSV file')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    add_similarity_to_detections(args.csv_file)
