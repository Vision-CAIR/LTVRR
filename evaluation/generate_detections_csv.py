import csv
import os
import pickle


data = ''
split = ''
model = ''

if not os.path.exists('./reports/' + data):
    os.mkdir('./reports/' + data)
if not os.path.exists('./reports/' + data + '/' + split):
    os.mkdir('./reports/' + data + '/' + split)
report_path = './reports/' + data + '/' + split + '/'
csv_path = report_path + 'csv_files/'

if not os.path.exists(report_path):
    os.mkdir(report_path)
if not os.path.exists(csv_path):
    os.mkdir(csv_path)

def generate_csv_file_from_det_file(detections, suffix, pred_freq, obj_freq, csv_path):
    with open(csv_path + suffix + '.csv', 'w', newline='') as csvfile:
        total_test_iters = len(detections['image_idx'])
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        spamwriter.writerow(['image_id',
                             'gt_rel',
                             'det_rel',
                             'rel_freq_gt',
                             'rel_rank_gt',
                             'gt_sbj',
                             'det_sbj',
                             'sbj_freq_gt',
                             'sbj_rank_gt',
                             'gt_obj',
                             'det_obj',
                             'obj_freq_gt',
                             'obj_rank_gt'])

        for i in range(0, total_test_iters):
            image_idx = detections['image_idx'][i]
            image_id = detections['image_id'][i]
            for j in range(len(detections['gt_labels_sbj'][i])):
                gt_labels_sbj = detections['gt_labels_sbj'][i][j]
                gt_labels_obj = detections['gt_labels_obj'][i][j]
                gt_labels_rel = detections['gt_labels_rel'][i][j]
                det_labels_sbj = detections['labels_sbj'][i][j]
                det_labels_obj = detections['labels_obj'][i][j]
                det_labels_rel = detections['labels_rel'][i][j]

                sbj_gt_rank = det_labels_sbj.index(gt_labels_sbj)
                obj_gt_rank = det_labels_obj.index(gt_labels_obj)
                rel_gt_rank = det_labels_rel.index(gt_labels_rel)

                spamwriter.writerow(
                    [image_id,
                     gt_labels_rel,
                     det_labels_rel[0],
                     pred_freq[gt_labels_rel],
                     rel_gt_rank,
                     gt_labels_sbj,
                     det_labels_sbj[0],
                     obj_freq[gt_labels_sbj],
                     sbj_gt_rank
                     gt_labels_obj,
                     det_labels_obj[0],
                     obj_freq[gt_labels_obj],
                     obj_gt_rank])



