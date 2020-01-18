import csv
import os
# import pickle
from six.moves import cPickle as pickle
import numpy as np

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

def generate_csv_file_from_det_file(detections, pred_freq, obj_freq, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        total_test_iters = len(detections['image_idx'])
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        spamwriter.writerow(['image_id',
                             'gt_rel',
                             'det_rel',
                             'rel_freq_gt',
                             'rel_rank',
                             'gt_sbj',
                             'det_sbj',
                             'sbj_freq_gt',
                             'sbj_rank',
                             'gt_obj',
                             'det_obj',
                             'obj_freq_gt',
                             'obj_rank'])

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

                #print(det_labels_rel[:10])
                #print(gt_labels_rel)
                #print(np.where(det_labels_rel == gt_labels_rel))
                try:
                    rel_rank = np.where(det_labels_rel == gt_labels_rel)[0][0]
                except IndexError as e:
                    rel_rank = 251

                try:
                    sbj_rank = np.where(det_labels_sbj == gt_labels_sbj)[0][0]
                except IndexError as e:
                    sbj_rank = 251

                try:
                    obj_rank = np.where(det_labels_obj == gt_labels_obj)[0][0]
                except IndexError as e:
                    obj_rank = 251

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
                     det_labels_rel[0],
                     pred_freq[gt_labels_rel],
                     rel_rank,
                     gt_labels_sbj,
                     det_labels_sbj[0],
                     obj_freq[gt_labels_sbj],
                     sbj_rank,
                     gt_labels_obj,
                     det_labels_obj[0],
                     obj_freq[gt_labels_obj],
                     obj_rank])


out_dir = '/home/x_abdelks/c2044/Large_Scale_VRD_pytorch/Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only/gvqa/Jan13-05-44-10_gpu208-10_step_with_prd_cls_v3/test/'
detections_file = out_dir + 'rel_detections_gt_boxes_prdcls.pkl'
csv_path = out_dir + 'rel_detections_gt_boxes_prdcls.csv'

detections = pickle.load(open(detections_file, 'rb'))

pred_freq_paths = '/ibex/scratch/x_abdelks/Large-Scale-VRD/datasets/large_scale_VRD/GVQA/freq_pred.npy'

obj_freq_paths = '/ibex/scratch/x_abdelks/Large-Scale-VRD/datasets/large_scale_VRD/GVQA/freq_obj.npy'

pred_freq = np.load(pred_freq_paths)
obj_freq = np.load(obj_freq_paths)


generate_csv_file_from_det_file(detections, pred_freq, obj_freq, csv_path)
