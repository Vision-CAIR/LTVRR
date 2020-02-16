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


def generate_csv_file_from_det_obj(detections, csv_path, obj_categores, prd_categories, obj_freq_dict, prd_freq_dict):
    with open(csv_path, 'w', newline='') as csvfile:
        total_test_iters = len(detections)
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

        for i in tqdm(range(0, total_test_iters)):
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

                sbj_freq = obj_freq_dict[det_labels_sbj]
                obj_freq = obj_freq_dict[det_labels_obj]
                rel_freq = prd_freq_dict[det_labels_rel]
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
                     rel_freq,
                     rel_rank,

                     gt_labels_sbj,
                     det_labels_sbj,
                     sbj_freq,
                     sbj_rank,

                     gt_labels_obj,
                     det_labels_obj,
                     obj_freq,
                     obj_rank])


def generate_csv_file_from_det_file(detections_file, csv_file, obj_categories, prd_categories, obj_freq_dict, prd_freq_dict):
    detections = pickle.load(open(detections_file, 'rb'))
    generate_csv_file_from_det_obj(detections, csv_file, obj_categories, prd_categories, obj_freq_dict, prd_freq_dict)



if __name__ == '__main__':
    #pred_freq_paths = '/ibex/scratch/x_abdelks/Large-Scale-VRD/datasets/large_scale_VRD/GVQA/freq_pred.npy'
    pred_freq_paths = '/ibex/scratch/x_abdelks/Large-Scale-VRD/datasets/large_scale_VRD/GVQA/reduced_data/10k/seed0/freq_prd.npy'
    #pred_freq_paths = '/ibex/scratch/x_abdelks/Large-Scale-VRD/datasets/large_scale_VRD/Visual_Genome/freq_pred.npy'

    #obj_freq_paths = '/ibex/scratch/x_abdelks/Large-Scale-VRD/datasets/large_scale_VRD/GVQA/freq_obj.npy'
    obj_freq_paths = '/ibex/scratch/x_abdelks/Large-Scale-VRD/datasets/large_scale_VRD/GVQA/reduced_data/10k/seed0/freq_obj.npy'
    #obj_freq_paths = '/ibex/scratch/x_abdelks/Large-Scale-VRD/datasets/large_scale_VRD/Visual_Genome/freq_obj.npy'

    pred_freq = np.load(pred_freq_paths)
    obj_freq = np.load(obj_freq_paths)

    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_baseline/gvqa/Jan28-08-41-52_gpu211-06_step_with_prd_cls_v3/test/'
    out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa10k_y_loss_only_baseline/gvqa10k/Feb11-08-27-46_gpu211-06_step_with_prd_cls_v3/test/'
    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_baseline/gvqa/Jan13-05-44-10_gpu208-10_step_with_prd_cls_v3/test/'
    detections_file = out_dir + 'rel_detections_gt_boxes_prdcls.pkl'
    csv_path = out_dir + 'rel_detections_gt_boxes_prdcls.csv'

    generate_csv_file_from_det_file(detections_file, pred_freq, obj_freq, csv_path)
    print('Wrote csv detections to:', csv_path)

    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_hubness/gvqa/Jan28-20-32-18_gpu210-10_step_with_prd_cls_v3/test/'
    out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa10k_y_loss_only_hubness/gvqa10k/Feb11-08-28-32_gpu210-10_step_with_prd_cls_v3/test/'
    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_hubness/gvqa/Jan13-05-44-14_gpu214-02_step_with_prd_cls_v3/test/'
    detections_file = out_dir + 'rel_detections_gt_boxes_prdcls.pkl'
    csv_path = out_dir + 'rel_detections_gt_boxes_prdcls.csv'

    generate_csv_file_from_det_file(detections_file, pred_freq, obj_freq, csv_path)
    print('Wrote csv detections to:', csv_path)

    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_hubness10k/gvqa/Jan25-07-12-43_gpu211-02_step_with_prd_cls_v3/test/'
    out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa10k_y_loss_only_hubness10k/gvqa10k/Feb11-10-07-52_gpu208-18_step_with_prd_cls_v3/test/'
    detections_file = out_dir + 'rel_detections_gt_boxes_prdcls.pkl'
    csv_path = out_dir + 'rel_detections_gt_boxes_prdcls.csv'

    generate_csv_file_from_det_file(detections_file, pred_freq, obj_freq, csv_path)
    print('Wrote csv detections to:', csv_path)

    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_focal_025/gvqa/Jan25-02-57-31_gpu208-14_step_with_prd_cls_v3/test/'
    out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa10k_y_loss_only_focal_025/gvqa10k/Feb11-18-15-44_gpu210-10_step_with_prd_cls_v3/test/'
    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_focal_025/gvqa/Jan13-05-44-39_gpu211-06_step_with_prd_cls_v3/test/'
    detections_file = out_dir + 'rel_detections_gt_boxes_prdcls.pkl'
    csv_path = out_dir + 'rel_detections_gt_boxes_prdcls.csv'

    generate_csv_file_from_det_file(detections_file, pred_freq, obj_freq, csv_path)
    print('Wrote csv detections to:', csv_path)

    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_focal_1/gvqa/Jan26-11-03-52_gpu210-10_step_with_prd_cls_v3/test/'
    out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa10k_y_loss_only_focal_1/gvqa10k/Feb11-19-56-53_gpu208-18_step_with_prd_cls_v3/test/'
    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_focal_1/gvqa/Jan14-06-21-51_gpu208-10_step_with_prd_cls_v3/test/'
    detections_file = out_dir + 'rel_detections_gt_boxes_prdcls.pkl'
    csv_path = out_dir + 'rel_detections_gt_boxes_prdcls.csv'

    generate_csv_file_from_det_file(detections_file, pred_freq, obj_freq, csv_path)
    print('Wrote csv detections to:', csv_path)

    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_focal_2/gvqa/Jan25-02-57-08_gpu214-10_step_with_prd_cls_v3/test/'
    out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa10k_y_loss_only_focal_2/gvqa10k/Feb11-23-16-27_gpu211-06_step_with_prd_cls_v3/test/'
    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_focal_2/gvqa/Jan13-05-42-54_gpu208-18_step_with_prd_cls_v3/test/'
    detections_file = out_dir + 'rel_detections_gt_boxes_prdcls.pkl'
    csv_path = out_dir + 'rel_detections_gt_boxes_prdcls.csv'

    generate_csv_file_from_det_file(detections_file, pred_freq, obj_freq, csv_path)
    print('Wrote csv detections to:', csv_path)

    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_focal_5/gvqa/Jan26-16-58-50_gpu214-14_step_with_prd_cls_v3/test/'
    out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa10k_y_loss_only_focal_5/gvqa10k/Feb12-04-01-25_gpu210-14_step_with_prd_cls_v3/test/'
    #out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_focal_5/gvqa/Jan14-06-25-03_gpu211-06_step_with_prd_cls_v3/test/'
    detections_file = out_dir + 'rel_detections_gt_boxes_prdcls.pkl'
    csv_path = out_dir + 'rel_detections_gt_boxes_prdcls.csv'

    generate_csv_file_from_det_file(detections_file, pred_freq, obj_freq, csv_path)
    print('Wrote csv detections to:', csv_path)


    out_dir = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa10k_y_loss_only_focal_10/gvqa10k/Feb12-05-45-22_gpu208-18_step_with_prd_cls_v3/test/'
    detections_file = out_dir + 'rel_detections_gt_boxes_prdcls.pkl'
    csv_path = out_dir + 'rel_detections_gt_boxes_prdcls.csv'

    generate_csv_file_from_det_file(detections_file, pred_freq, obj_freq, csv_path)
    print('Wrote csv detections to:', csv_path)
