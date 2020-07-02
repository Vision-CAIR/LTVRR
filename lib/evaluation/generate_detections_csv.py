# Written by Sherif Abdelkarim on Feb 2020


import argparse
import csv
import os
# import pickle
from six.moves import cPickle as pickle
import numpy as np
from tqdm import tqdm
# if not os.path.exists('./reports/' + data):
#     os.mkdir('./reports/' + data)
# if not os.path.exists('./reports/' + data + '/' + split):
#     os.mkdir('./reports/' + data + '/' + split)
# report_path = './reports/' + data + '/' + split + '/'
# csv_path = report_path + 'csv_files/'

# if not os.path.exists(report_path):
#     os.mkdir(report_path)
# if not os.path.exists(csv_path):
#     os.mkdir(csv_path)

def generate_boxes_csv_from_det_obj(detections, csv_path, obj_categores, prd_categories, obj_freq_dict, prd_freq_dict):
    with open(csv_path, 'w', newline='') as csvfile:
        total_test_iters = len(detections)
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        spamwriter.writerow(['image_id',
                             'gt_rel',
                             'det_rel',
                             'rel_freq_gt',
                             'rel_freq_det',
                             'rel_rank',
                             'gt_sbj',
                             'det_sbj',
                             'sbj_freq_gt',
                             'sbj_freq_det',
                             'sbj_rank',
                             'sbj_box_0',
                             'sbj_box_1',
                             'sbj_box_2',
                             'sbj_box_3',
                             'gt_obj',
                             'det_obj',
                             'obj_freq_gt',
                             'obj_freq_det',
                             'obj_rank',
                             'obj_box_0',
                             'obj_box_1',
                             'obj_box_2',
                             'obj_box_3'])

        for i in range(0, total_test_iters):
            # image_idx = detections[i]['image_idx']
            # image_id = detections[i]['image_id']
            image_id = detections[i]['image'].split('/')[-1].split('.')[0]

            det_scores_prd_all = detections[i]['prd_scores'][:, 1:]
            det_labels_rel_all = np.argsort(-det_scores_prd_all, axis=1)

            det_scores_sbj_all = detections[i]['sbj_scores_out']
            det_labels_sbj_all = np.argsort(-det_scores_sbj_all, axis=1)

            det_scores_obj_all = detections[i]['obj_scores_out']
            det_labels_obj_all = np.argsort(-det_scores_obj_all, axis=1)

            for j in range(len(detections[i]['gt_sbj_labels'])):
                gt_labels_sbj_idx = detections[i]['gt_sbj_labels'][j]
                gt_labels_obj_idx = detections[i]['gt_obj_labels'][j]
                gt_labels_rel_idx = detections[i]['gt_prd_labels'][j]

                gt_labels_sbj = obj_categores[detections[i]['gt_sbj_labels'][j]]
                gt_labels_obj = obj_categores[detections[i]['gt_obj_labels'][j]]
                gt_labels_rel = prd_categories[detections[i]['gt_prd_labels'][j]]

                # det_labels_sbj = det_labels_sbj_all[j, 0]
                det_labels_sbj = obj_categores[det_labels_sbj_all[j, 0]]
                sbj_rank = np.where(det_labels_sbj_all[j, :] == gt_labels_sbj_idx)[0][0]

                # det_labels_obj = det_labels_obj_all[j, 0]
                det_labels_obj = obj_categores[det_labels_obj_all[j, 0]]
                obj_rank = np.where(det_labels_obj_all[j, :] == gt_labels_obj_idx)[0][0]

                det_labels_rel = prd_categories[det_labels_rel_all[j, 0]]
                rel_rank = np.where(det_labels_rel_all[j, :] == gt_labels_rel_idx)[0][0]

                if gt_labels_sbj in obj_freq_dict.keys():
                    gt_sbj_freq = obj_freq_dict[gt_labels_sbj]
                else:
                    gt_sbj_freq = 0
                if gt_labels_obj in obj_freq_dict.keys():
                    gt_obj_freq = obj_freq_dict[gt_labels_obj]
                else:
                    gt_obj_freq = 0
                if gt_labels_rel in prd_freq_dict.keys():
                    gt_rel_freq = prd_freq_dict[gt_labels_rel]
                else:
                    gt_rel_freq = 0

                if det_labels_sbj in obj_freq_dict.keys():
                    det_sbj_freq = obj_freq_dict[det_labels_sbj]
                else:
                    det_sbj_freq = 0
                if det_labels_obj in obj_freq_dict.keys():
                    det_obj_freq = obj_freq_dict[det_labels_obj]
                else:
                    det_obj_freq = 0
                if det_labels_rel in prd_freq_dict.keys():
                    det_rel_freq = prd_freq_dict[det_labels_rel]
                else:
                    det_obj_freq = 0

                sbj_box_0, sbj_box_1, sbj_box_2, sbj_box_3 = detections[i]['gt_sbj_boxes'][j, :]
                obj_box_0, obj_box_1, obj_box_2, obj_box_3 = detections[i]['gt_obj_boxes'][j, :]

                spamwriter.writerow(
                    [image_id,

                     gt_labels_rel,
                     det_labels_rel,
                     gt_rel_freq,
                     det_rel_freq,
                     rel_rank,

                     gt_labels_sbj,
                     det_labels_sbj,
                     gt_sbj_freq,
                     det_sbj_freq,
                     sbj_rank,
                     sbj_box_0,
                     sbj_box_1,
                     sbj_box_2,
                     sbj_box_3,

                     gt_labels_obj,
                     det_labels_obj,
                     gt_obj_freq,
                     det_obj_freq,
                     obj_rank,
                     obj_box_0,
                     obj_box_1,
                     obj_box_2,
                     obj_box_3])

def generate_topk_csv_from_det_obj(detections, csv_path, obj_categories, prd_categories, k):
    with open(csv_path, 'w', newline='') as csvfile:
        total_test_iters = len(detections)
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        spamwriter.writerow(['image_id',
                             'box_id',
                             'det_id',

                             'gt_rel',
                             'det_rel',

                             'gt_sbj',
                             'det_sbj',

                             'gt_obj',
                             'det_obj'])

        for i in range(0, total_test_iters):
            # image_idx = detections[i]['image_idx']
            # image_id = detections[i]['image_id']
            image_id = detections[i]['image'].split('/')[-1].split('.')[0]

            det_scores_prd_all = detections[i]['prd_scores'][:, 1:]
            det_labels_rel_all = np.argsort(-det_scores_prd_all, axis=1)

            det_scores_sbj_all = detections[i]['sbj_scores_out']
            det_labels_sbj_all = np.argsort(-det_scores_sbj_all, axis=1)

            det_scores_obj_all = detections[i]['obj_scores_out']
            det_labels_obj_all = np.argsort(-det_scores_obj_all, axis=1)

            for j in range(len(detections[i]['gt_sbj_labels'])):
                box_id = j
                gt_labels_sbj_idx = detections[i]['gt_sbj_labels'][j]
                gt_labels_obj_idx = detections[i]['gt_obj_labels'][j]
                gt_labels_rel_idx = detections[i]['gt_prd_labels'][j]

                gt_labels_sbj = obj_categories[detections[i]['gt_sbj_labels'][j]]
                gt_labels_obj = obj_categories[detections[i]['gt_obj_labels'][j]]
                gt_labels_rel = prd_categories[detections[i]['gt_prd_labels'][j]]

                # det_labels_sbj = det_labels_sbj_all[j, 0]
                # topk_sbj = obj_categories[det_labels_sbj_all[j, :k]]

                # det_labels_obj = det_labels_obj_all[j, 0]
                # topk_obj = obj_categories[det_labels_obj_all[j, :k]]

                # topk_rel = prd_categories[det_labels_rel_all[j, :k]]

                for m in range(k):
                    detection_id = m
                    det_sbj = obj_categories[det_labels_sbj_all[j, m]]
                    det_obj = obj_categories[det_labels_obj_all[j, m]]
                    det_rel = prd_categories[det_labels_rel_all[j, m]]
                    spamwriter.writerow(
                        [image_id,
                         box_id,
                         detection_id,

                         gt_labels_rel,
                         det_rel,

                         gt_labels_sbj,
                         det_sbj,

                         gt_labels_obj,
                         det_obj])


def generate_csv_file_from_det_obj(detections, csv_path, obj_categores, prd_categories, obj_freq_dict, prd_freq_dict):
    with open(csv_path, 'w', newline='') as csvfile:
        total_test_iters = len(detections)
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        spamwriter.writerow(['image_id',
                             'gt_rel',
                             'det_rel',
                             'rel_freq_gt',
                             'rel_freq_det',
                             'rel_rank',
                             'gt_sbj',
                             'det_sbj',
                             'sbj_freq_gt',
                             'sbj_freq_det',
                             'sbj_rank',
                             'gt_obj',
                             'det_obj',
                             'obj_freq_gt',
                             'obj_freq_det',
                             'obj_rank'])

        for i in range(0, total_test_iters):
            # image_idx = detections[i]['image_idx']
            # image_id = detections[i]['image_id']
            image_id = detections[i]['image'].split('/')[-1].split('.')[0]

            det_scores_prd_all = detections[i]['prd_scores'][:, 1:]
            det_labels_rel_all = np.argsort(-det_scores_prd_all, axis=1)

            det_scores_sbj_all = detections[i]['sbj_scores_out']
            det_labels_sbj_all = np.argsort(-det_scores_sbj_all, axis=1)

            det_scores_obj_all = detections[i]['obj_scores_out']
            det_labels_obj_all = np.argsort(-det_scores_obj_all, axis=1)

            for j in range(len(detections[i]['gt_sbj_labels'])):
                gt_labels_sbj_idx = detections[i]['gt_sbj_labels'][j]
                gt_labels_obj_idx = detections[i]['gt_obj_labels'][j]
                gt_labels_rel_idx = detections[i]['gt_prd_labels'][j]

                gt_labels_sbj = obj_categores[detections[i]['gt_sbj_labels'][j]]
                gt_labels_obj = obj_categores[detections[i]['gt_obj_labels'][j]]
                gt_labels_rel = prd_categories[detections[i]['gt_prd_labels'][j]]

                # det_labels_sbj = det_labels_sbj_all[j, 0]
                det_labels_sbj = obj_categores[det_labels_sbj_all[j, 0]]
                sbj_rank = np.where(det_labels_sbj_all[j, :] == gt_labels_sbj_idx)[0][0]

                # det_labels_obj = det_labels_obj_all[j, 0]
                det_labels_obj = obj_categores[det_labels_obj_all[j, 0]]
                obj_rank = np.where(det_labels_obj_all[j, :] == gt_labels_obj_idx)[0][0]

                det_labels_rel = prd_categories[det_labels_rel_all[j, 0]]
                rel_rank = np.where(det_labels_rel_all[j, :] == gt_labels_rel_idx)[0][0]

                if gt_labels_sbj in obj_freq_dict.keys():
                    gt_sbj_freq = obj_freq_dict[gt_labels_sbj]
                else:
                    gt_sbj_freq = 0
                if gt_labels_obj in obj_freq_dict.keys():
                    gt_obj_freq = obj_freq_dict[gt_labels_obj]
                else:
                    gt_obj_freq = 0
                if gt_labels_rel in prd_freq_dict.keys():
                    gt_rel_freq = prd_freq_dict[gt_labels_rel]
                else:
                    gt_rel_freq = 0

                if det_labels_sbj in obj_freq_dict.keys():
                    det_sbj_freq = obj_freq_dict[det_labels_sbj]
                else:
                    det_sbj_freq = 0
                if det_labels_obj in obj_freq_dict.keys():
                    det_obj_freq = obj_freq_dict[det_labels_obj]
                else:
                    det_obj_freq = 0
                if det_labels_rel in prd_freq_dict.keys():
                    det_rel_freq = prd_freq_dict[det_labels_rel]
                else:
                    det_obj_freq = 0
                # print('rel_rank', rel_rank)
                # print('det_labels_rel_all[j, :]', det_labels_rel_all[j, :])
                # print('gt_labels_rel', gt_labels_rel)
                # exit()
                # det_labels_rel = detections[i]['prd_labels'][j]
                # print(det_labels_rel[:10])
                # print(gt_labels_rel)
                # print(np.where(det_labels_rel == gt_labels_rel))
                # try:
                #    rel_rank = np.where(det_labels_rel == gt_labels_rel)[0][0]
                # except IndexError as e:
                #    rel_rank = 251

                # try:
                #    sbj_rank = np.where(det_labels_sbj == gt_labels_sbj)[0][0]
                # except IndexError as e:
                #    sbj_rank = 251

                # try:
                #    obj_rank = np.where(det_labels_obj == gt_labels_obj)[0][0]
                # except IndexError as e:
                #    obj_rank = 251

                # rel_top1 = gt_labels_rel in det_labels_rel[:1]
                # rel_top5 = gt_labels_rel in det_labels_rel[:5]
                # rel_top10 = gt_labels_rel in det_labels_rel[:10]

                # sbj_top1 = gt_labels_sbj in det_labels_sbj[:1]
                # sbj_top5 = gt_labels_sbj in det_labels_sbj[:5]
                # sbj_top10 = gt_labels_sbj in det_labels_sbj[:10]

                # obj_top1 = gt_labels_obj in det_labels_obj[:1]
                # obj_top5 = gt_labels_obj in det_labels_obj[:5]
                # obj_top10 = gt_labels_obj in det_labels_obj[:10]

                spamwriter.writerow(
                    [image_id,

                     gt_labels_rel,
                     det_labels_rel,
                     gt_rel_freq,
                     det_rel_freq,
                     rel_rank,

                     gt_labels_sbj,
                     det_labels_sbj,
                     gt_sbj_freq,
                     det_sbj_freq,
                     sbj_rank,

                     gt_labels_obj,
                     det_labels_obj,
                     gt_obj_freq,
                     det_obj_freq,
                     obj_rank])


def generate_csv_file_from_det_file(detections_file, csv_file, obj_categories, prd_categories, obj_freq_dict, prd_freq_dict):
    detections = pickle.load(open(detections_file, 'rb'))
    generate_csv_file_from_det_obj(detections, csv_file, obj_categories, prd_categories, obj_freq_dict, prd_freq_dict)


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Generate csv files')
    parser.add_argument(
        '--pkl_file', dest='pkl_file', required=True,
        help='Detections Pickle file')

    return parser.parse_args()

