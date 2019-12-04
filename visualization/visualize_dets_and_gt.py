# coding: utf-8

# In[1]:

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.lines as lines
import numpy as np
from six.moves import cPickle as pickle
import json
from tqdm import tqdm


dataset = 'gvqa'
model = '_hubness10000'
split = 'test'
if model == '_baseline':
    model_path = ''
else:
    model_path = model
if dataset == 'gvqa':
    dataset_folder = 'GVQA'
elif dataset == 'vg_wiki_and_relco':
    dataset_folder = 'Visual_Genome'
else:
    raise NotImplementedError

base_project_path = '/ibex/scratch/x_abdelks/Large-Scale-VRD/'

base_data_path = base_project_path + 'datasets/large_scale_VRD/'
current_data_folder = base_data_path + dataset_folder + '/'

base_checkpoints_folder = base_project_path + 'checkpoints/'
current_checkpoints_folder = base_checkpoints_folder + dataset + '/'

rel_path = current_data_folder + 'relationships_clean_spo_joined_and_merged.json'
relationships = json.load(open(rel_path))
# VG
dir_path = current_data_folder + 'vis_output/' + model + '/'

topk_dets_file = current_checkpoints_folder + \
                 'VGG16_reldn_fast_rcnn_conv4_spo_for_p/embd_fusion_w_relu_yall/8gpus_vgg16_softmaxed_triplet_no_last_l2norm_trainval_w_cluster_2_lan_layers{}/{}/reldn_detections.pkl'.format(model_path, split)

print('Loading detections pickle..')
with open(topk_dets_file, 'rb') as f:
    topk_dets = pickle.load(f, encoding='latin1')
print('Done')
print('topk_dets_orig', len(topk_dets))
# topk_dets = get_topk_dets(dets)


# print(len(topk_dets))
# print(topk_dets.keys())
# i = 0
# print(topk_dets['boxes_obj'][i].shape)
# print(topk_dets['gt_boxes_obj'][i].shape)
# print(np.sum(topk_dets['boxes_obj'][i] - topk_dets['gt_boxes_obj'][i]))
# # print(topk_dets['gt_boxes_obj'][i])
# print(topk_dets['scores_obj'][i].shape)
# print(topk_dets['labels_obj'][i].shape)
# print(topk_dets['scores_obj'][i].shape)
# print(topk_dets['labels_obj'][i].shape)
# print(topk_dets['scores_rel'][i].shape)
# print(topk_dets['labels_rel'][i].shape)
# min_len = 100
# for det in topk_dets['scores_obj']:
#     # print det.shape[0]
#     if min_len > det.shape[0]:
#         min_len = det.shape[0]
# print('min_len = ', min_len)
#
# max_gt_num = 0
# num_gt_50 = 0
# num_gt_40 = 0
# num_gt_30 = 0
# num_gt_20 = 0
# num_gt_10 = 0
#
# for gt_labels_prd in topk_dets['scores_rel']:
#     # gt_labels_prd = det['gt_labels_prd']
#     if gt_labels_prd.shape[0] > 50:
#         num_gt_50 += 1
#     if gt_labels_prd.shape[0] > 40:
#         num_gt_40 += 1
#     if gt_labels_prd.shape[0] > 30:
#         num_gt_30 += 1
#     if gt_labels_prd.shape[0] > 20:
#         num_gt_20 += 1
#     if gt_labels_prd.shape[0] > 10:
#         num_gt_10 += 1
#     if max_gt_num < gt_labels_prd.shape[0]:
#         max_gt_num = gt_labels_prd.shape[0]
# print('num_gt_50: ', num_gt_50)
# print('num_gt_40: ', num_gt_40)
# print('num_gt_30: ', num_gt_30)
# print('num_gt_20: ', num_gt_20)
# print('num_gt_10: ', num_gt_10)
# print('max_gt_num: ', max_gt_num)

# VG

rels_joined_merged = json.load(open(
    current_data_folder + 'relationships_clean_spo_joined_and_merged.json'))
rels_joined_merged_test_idx = json.load(
    open(current_data_folder + 'test_clean.json'))
rels_joined_merged_train_idx = json.load(
    open(current_data_folder + 'train_clean.json'))
rels_joined_merged_val_idx = json.load(
    open(current_data_folder + 'val_clean.json'))

all_rels_map = {}
for cnt, rel in enumerate(rels_joined_merged):
    all_rels_map[rel['image_id']] = cnt

img_path = current_data_folder + 'images/'
obj_cats = []
with open(current_data_folder + 'object_categories_spo_joined_and_merged.txt', 'r') as myfile:
    obj_cats = myfile.readlines()

prd_cats = []
with open(current_data_folder + 'predicate_categories_spo_joined_and_merged.txt', 'r') as myfile:
    prd_cats = myfile.readlines()

print(len(obj_cats))
print(len(prd_cats))


# with open('Large-Scale-VRD.pytorch/data/vg/objects.json') as f:
#     obj_cats = json.load(f)
# with open('Large-Scale-VRD.pytorch/data/vg/predicates.json') as f:
#     prd_cats = json.load(f)

def box_overlap(box1, box2):
    overlap = 0.0
    box_area = (
            (box2[2] - box2[0] + 1) *
            (box2[3] - box2[1] + 1)
    )
    iw = (
            min(box1[2], box2[2]) -
            max(box1[0], box2[0]) + 1
    )
    if iw > 0:
        ih = (
                min(box1[3], box2[3]) -
                max(box1[1], box2[1]) + 1
        )
        if ih > 0:
            ua = float(
                (box1[2] - box1[0] + 1) *
                (box1[3] - box1[1] + 1) +
                box_area - iw * ih
            )
            overlap = iw * ih / ua
    return overlap


# box1 and box2 are in [x1, y1. w. h] format
def box_union(box1, box2):
    xmin = min(box1[0], box2[0])
    ymin = min(box1[1], box2[1])
    xmax = max(box1[0] + box1[2] - 1, box2[0] + box2[2] - 1)
    ymax = max(box1[1] + box1[3] - 1, box2[1] + box2[3] - 1)
    return [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]


def box2rect(img, box):
    x = box[0] + edge_width / 2
    y = box[1] + edge_width / 2
    w = box[2] - box[0] - edge_width
    h = box[3] - box[1] - edge_width
    return x, y, w, h


import re

pattern = '(?!person)'
s = 'peson'
print(re.match(pattern, s))
if not (re.match(pattern, s)):
    print('skip')
else:
    print('dont skip ')

# pattern = '^(?!person)'
# if re.match(pattern,s) :
#     print('matched')
# else :


# In[16]:

edge_width = 3
font_size = 18


# pattern = '^(?!person)'
# if re.match(pattern,s) :
#     print('matched')
# else :
#     print('not matched')

def Visualize_Detections(ind, topk_dets, rels_joined_merged_test_idx, sbj_q, prd_q, obj_q, sbj_p='*', prd_p='*',
                         obj_p='*'):
    #print(str(topk_dets['image_idx'][ind]))

    #print('ind: ', ind)
    save_output = True

    # [u'boxes_obj', u'gt_boxes_obj', u'gt_labels_sbj', u'gt_labels_rel', u'labels_obj', u'labels_rel', u'scores_obj', u'gt_boxes_rel', u'image_id', u'scores_rel', u'scores_sbj', u'boxes_rel', u'labels_sbj', u'image_idx', u'boxes_sbj', u'gt_boxes_sbj', u'gt_labels_obj']

    gt_boxes_sbj = topk_dets['gt_boxes_sbj'][ind]
    gt_boxes_obj = topk_dets['gt_boxes_obj'][ind]
    gt_labels_sbj = topk_dets['gt_labels_sbj'][ind]
    gt_labels_prd = topk_dets['gt_labels_rel'][ind]
    gt_labels_obj = topk_dets['gt_labels_obj'][ind]

    topk = gt_labels_prd.shape[0]
    # rels_joined_merged_test_idx
    sbj_boxes = topk_dets['boxes_sbj'][ind]
    obj_boxes = topk_dets['boxes_obj'][ind]
    sbj_labels = topk_dets['labels_sbj'][ind]
    obj_labels = topk_dets['labels_obj'][ind]
    prd_labels = topk_dets['labels_rel'][ind]
    det_scores = topk_dets['scores_obj'][ind] + topk_dets['scores_rel'][ind] + topk_dets['scores_sbj'][ind]

    image_real_id = rels_joined_merged_test_idx[topk_dets['image_idx'][ind]]
    img_name = str(image_real_id) + '.jpg'

    n_q = 0
    sel_gt = [False] * min(topk, gt_labels_sbj.shape[0])
    for j in range(min(topk, gt_labels_sbj.shape[0])):
        # gt
        gt_sbj_label = gt_labels_sbj[j]
        gt_prd_label = gt_labels_prd[j]
        gt_obj_label = gt_labels_obj[j]
        s_name = obj_cats[gt_sbj_label].strip().lower()
        p_name = prd_cats[gt_prd_label].strip().lower()
        o_name = obj_cats[gt_obj_label].strip().lower()
        if sbj_q != '*' and (s_name != sbj_q or not (re.match(sbj_q, s_name))):
            continue

        if obj_q != '*' and (o_name != obj_q or not (re.match(obj_q, o_name))):
            continue

        if prd_q != '*' and (p_name != prd_q or not (re.match(prd_q, p_name))):
            #             import pdb
            #             pdb.set_trace()
            continue

        sbj_label = sbj_labels[j][0]
        prd_label = prd_labels[j][0]
        obj_label = obj_labels[j][0]
        s_name = obj_cats[sbj_label].strip().lower()
        p_name = prd_cats[prd_label].strip().lower()
        o_name = obj_cats[obj_label].strip().lower()
        if sbj_p != '*' and (s_name != sbj_p or not (re.match(sbj_p, s_name))):
            import pdb
            pdb.set_trace()
            continue

        if obj_p != '*' and (o_name != obj_p or not (re.match(obj_p, o_name))):
            continue

        if prd_p != '*' and (p_name != prd_p or not (re.match(prd_p, p_name))):
            #             import pdb
            #             pdb.set_trace()
            continue

        sel_gt[j] = True
        n_q = n_q + 1

    if (n_q == 0):
        return n_q
    #print('image: ', img_name)
    #print('num_gt: ', topk)
    img = mpimg.imread(img_path + img_name)

    assert gt_boxes_sbj.shape[0] == gt_boxes_obj.shape[0]

    fig = plt.figure(figsize=(24, 16))

    plt.subplot(122)
    ax = plt.gca()
    plt.imshow(img)
    plt.axis('off')
    gt_title = plt.title('gt')
    plt.setp(gt_title, color='b')

    for j in range(min(topk, gt_labels_sbj.shape[0])):
        # gt
        if (sel_gt[j] == False):
            continue

        gt_sbj_label = gt_labels_sbj[j]
        gt_prd_label = gt_labels_prd[j]
        gt_obj_label = gt_labels_obj[j]
        gt_sbj_box = gt_boxes_sbj[j]
        gt_obj_box = gt_boxes_obj[j]
        s_name = obj_cats[gt_sbj_label].strip()
        p_name = prd_cats[gt_prd_label].strip()
        o_name = obj_cats[gt_obj_label].strip()

        s_x, s_y, s_w, s_h = box2rect(img, gt_sbj_box)
        s_cx = s_x + s_w // 2
        s_cy = s_y + s_h // 2
        ax.text(s_cx, s_cy - 2,
                s_name,
                fontsize=font_size,
                color='white',
                bbox=dict(facecolor='orange', alpha=0.5, pad=0, edgecolor='none'))
        o_x, o_y, o_w, o_h = box2rect(img, gt_obj_box)
        o_cx = o_x + o_w // 2
        o_cy = o_y + o_h // 2
        ax.text(o_cx, o_cy - 2,
                o_name,
                fontsize=font_size,
                color='white',
                bbox=dict(facecolor='blue', alpha=0.5, pad=0, edgecolor='none'))
        # rel line
        rel_line = lines.Line2D([s_cx, o_cx], [s_cy, o_cy], linewidth=edge_width, color='purple')
        ax.add_line(rel_line)
        mid_x = (s_cx + o_cx) // 2
        mid_y = (s_cy + o_cy) // 2
        ax.text(mid_x, mid_y,
                #                 p_name + ' ' + str(j),
                p_name,
                fontsize=font_size,
                color='white',
                bbox=dict(facecolor='purple', alpha=0.5, pad=0, edgecolor='none'))

    # print(sel_gt)
    plt.subplot(121)
    ax = plt.gca()
    plt.imshow(img)
    plt.axis('off')
    det_title = plt.title('det')
    plt.setp(det_title, color='b')
    for j in range(min(topk, sbj_labels.shape[0])):
        # det
        if (sel_gt[j] == False):
            continue
        det_score = det_scores[j][0]
        sbj_label = sbj_labels[j][0]
        prd_label = prd_labels[j][0]
        obj_label = obj_labels[j][0]
        sbj_box = sbj_boxes[j]
        obj_box = obj_boxes[j]
        s_name = obj_cats[sbj_label]
        p_name = prd_cats[prd_label]
        o_name = obj_cats[obj_label]
        # if(s_name)
        s_x, s_y, s_w, s_h = box2rect(img, sbj_box)
        s_cx = s_x + s_w // 2
        s_cy = s_y + s_h // 2
        ax.text(s_cx, s_cy - 2,
                s_name,
                fontsize=font_size,
                color='white',
                bbox=dict(facecolor='orange', alpha=0.5, pad=0, edgecolor='none'))
        o_x, o_y, o_w, o_h = box2rect(img, obj_box)
        o_cx = o_x + o_w // 2
        o_cy = o_y + o_h // 2
        ax.text(o_cx, o_cy - 2,
                o_name,
                fontsize=font_size,
                color='white',
                bbox=dict(facecolor='blue', alpha=0.5, pad=0, edgecolor='none'))
        # rel line
        rel_line = lines.Line2D([s_cx, o_cx], [s_cy, o_cy], linewidth=edge_width, color='purple')
        ax.add_line(rel_line)
        mid_x = (s_cx + o_cx) // 2
        mid_y = (s_cy + o_cy) // 2
        ax.text(mid_x, mid_y,
                #                 p_name + ' ' + str(j),
                p_name,
                fontsize=font_size,
                color='white',
                bbox=dict(facecolor='purple', alpha=0.5, pad=0, edgecolor='none'))

        # for kk in range(5):
        #     det_score = det_scores[j][kk]
        #     sbj_label = sbj_labels[j][kk]
        #     prd_label = prd_labels[j][kk]
        #     obj_label = obj_labels[j][kk]
        #     s_name = obj_cats[sbj_label]
        #     p_name = prd_cats[prd_label]
        #     o_name = obj_cats[obj_label]
        #     print('Top {} : {} {} {}'.format(kk, s_name, p_name, o_name))
        #     print('\ttotal score:\t {:.6f}'.format(det_score))
        # input('Press enter to continue: ')
        # plt.show()

    if save_output:
        output_dir = os.path.join(dir_path, str(image_real_id))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'all_true_pos.jpg'), bbox_inches='tight')
    #plt.show()
    plt.close(fig)

    # print SPO names and scores

    #print("list of top detection of highest scores ....")
    #for j in range(min(topk, sbj_labels.shape[0])):
    #    # det
    #    if (sel_gt[j] == False):
    #        continue
    #    det_score = det_scores[j][0]
    #    sbj_label = sbj_labels[j][0]
    #    prd_label = prd_labels[j][0]
    #    obj_label = obj_labels[j][0]
    #    s_name = obj_cats[sbj_label]
    #    p_name = prd_cats[prd_label]
    #    o_name = obj_cats[obj_label]
    #    print('{} {} {}'.format(s_name, p_name, o_name))
    #    print('\ttotal score:\t {:.6f}'.format(det_score))

    return n_q


# In[ ]:

def Visualize_GT_byIndex(ind, all_rels, sbj_q, prd_q, obj_q):
    # print('ind: ', ind)
    save_output = True

    # [u'boxes_obj', u'gt_boxes_obj', u'gt_labels_sbj', u'gt_labels_rel', u'labels_obj', u'labels_rel', u'scores_obj', u'gt_boxes_rel', u'image_id', u'scores_rel', u'scores_sbj', u'boxes_rel', u'labels_sbj', u'image_idx', u'boxes_sbj', u'gt_boxes_sbj', u'gt_labels_obj']

    topk = gt_labels_prd.shape[0]

    image_real_id = all_rels[ind]['image_id']

    relationship_ann_ind = all_rels[ind]['relationships']

    # print('Number of Rels = ',len(relationship_ann_ind))

    # img_name = str(topk_dets['image_idx'][ind])+'.jpg'
    # print(img_name)

    # rels_joined_merged_test_idx

    img_name = str(image_real_id) + '.jpg'

    n_q = 0
    sel_gt = [False] * len(relationship_ann_ind)
    for j in range(len(relationship_ann_ind)):
        # gt

        #         print((rels_joined_merged[0]['relationships'][0]['object']))
        #         print((rels_joined_merged[0]['relationships'][0]['subject']))
        #         print((rels_joined_merged[0]['relationships'][0]['predicate']))
        subject_ann = relationship_ann_ind[j]['subject']
        object_ann = relationship_ann_ind[j]['object']

        s_name = subject_ann['name'].strip()
        p_name = relationship_ann_ind[j]['predicate'].strip()
        o_name = object_ann['name'].strip()

        if sbj_q != '*' and s_name != sbj_q:
            continue

        if obj_q != '*' and o_name != obj_q:
            continue

        if prd_q != '*' and p_name != prd_q:
            #             import pdb
            #             pdb.set_trace()
            continue
        sel_gt[j] = True
        n_q = n_q + 1

    if (n_q == 0):
        return n_q
    print('image: ', img_name)
    print('num_gt: ', topk)
    img = mpimg.imread(img_path + img_name)

    fig = plt.figure(figsize=(24, 16))

    plt.subplot(122)
    ax = plt.gca()
    plt.imshow(img)
    plt.axis('off')
    gt_title = plt.title('gt')
    plt.setp(gt_title, color='b')

    for j in range(len(relationship_ann_ind)):
        # gt
        if (sel_gt[j] == False):
            continue

        subject_ann = relationship_ann_ind[j]['subject']
        object_ann = relationship_ann_ind[j]['object']

        gt_sbj_box = [subject_ann['x'], subject_ann['y'], subject_ann['x'] + subject_ann['w'],
                      subject_ann['y'] + subject_ann['h']]
        gt_obj_box = [object_ann['x'], object_ann['y'], object_ann['x'] + object_ann['w'],
                      object_ann['y'] + object_ann['h']]

        #         gt_sbj_box = [subject_ann['y'],subject_ann['x'], subject_ann['y']+subject_ann['h'], subject_ann['x']+subject_ann['w']]
        #         gt_obj_box = [object_ann['y'],object_ann['x'], object_ann['y']+object_ann['h'], object_ann['x']+object_ann['w']]

        s_name = subject_ann['name'].strip()
        p_name = relationship_ann_ind[j]['predicate'].strip()
        o_name = object_ann['name'].strip()

        s_x, s_y, s_w, s_h = box2rect(img, gt_sbj_box)
        s_cx = s_x + s_w // 2
        s_cy = s_y + s_h // 2
        ax.text(s_cx, s_cy - 2,
                s_name,
                fontsize=font_size,
                color='white',
                bbox=dict(facecolor='orange', alpha=0.5, pad=0, edgecolor='none'))
        o_x, o_y, o_w, o_h = box2rect(img, gt_obj_box)
        o_cx = o_x + o_w // 2
        o_cy = o_y + o_h // 2
        ax.text(o_cx, o_cy - 2,
                o_name,
                fontsize=font_size,
                color='white',
                bbox=dict(facecolor='blue', alpha=0.5, pad=0, edgecolor='none'))
        # rel line
        rel_line = lines.Line2D([s_cx, o_cx], [s_cy, o_cy], linewidth=edge_width, color='purple')
        ax.add_line(rel_line)
        mid_x = (s_cx + o_cx) // 2
        mid_y = (s_cy + o_cy) // 2
        ax.text(mid_x, mid_y,
                #                 p_name + ' ' + str(j),
                p_name,
                fontsize=font_size,
                color='white',
                bbox=dict(facecolor='purple', alpha=0.5, pad=0, edgecolor='none'))

    if save_output:
        output_dir = os.path.join(dir_path, str(image_real_id))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'all_true_pos.jpg'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return n_q


# In[19]:


# p = re.compile()
s = 'OK'

# print(rels_joined_merged[0])

# In[ ]:

# rels_joined_merged = json.load(open(base_data_path + 'Visual_Genome/relationships_clean_spo_joined_and_merged.json'))
# rels_joined_merged_test_idx = json.load(open(base_data_path + 'Visual_Genome/test_clean.json'))
# rels_joined_merged_train_idx = json.load(open(base_data_path + 'Visual_Genome/train_clean.json'))
# rels_joined_merged_val_idx = json.load(open(base_data_path + 'Visual_Genome/val_clean.json'))
# all_rels_map = {}
# for cnt, rel in enumerate(rels_joined_merged):
# 		all_rels_map[rel['image_id']] = cnt


# ind = np.random.randint(0, len(rels_joined_merged_train_idx))
ind = 20
# ImageId = rels_joined_merged_train_idx[ind]
# # print(rels_joined_merged[0]['image_id'])
# print((rels_joined_merged[0]['relationships'][0]['object']))
# print((rels_joined_merged[0]['relationships'][0]['subject']))
# print((rels_joined_merged[0]['relationships'][0]['predicate']))

# Visualize_GT_byIndex(all_rels_map[ImageId], rels_joined_merged, '*', '*', '*')


rels = rels_joined_merged_train_idx
rels = rels_joined_merged_val_idx
rels = rels_joined_merged_test_idx
rels = rels_joined_merged_train_idx
# for ind in range(len(rels)):
#     ImageId = rels[ind]
#     n_ind= Visualize_GT_byIndex(all_rels_map[ImageId], rels_joined_merged, '*', 'riding', '*')
#     if n_ind>0:
#         input('Press enter to continue: ')


# topk_dets_file = current_checkpoints_folder + 'VGG16_reldn_fast_rcnn_conv4_spo_for_p/embd_fusion_w_relu_yall/1gpus_vgg16_softmaxed_triplet_no_last_l2norm_trainval_w_cluster_2_lan_layers/{}/reldn_detections.pkl'.format(split)
#
# with open(topk_dets_file, 'rb') as f:
#     topk_dets_val = pickle.load(f, encoding='latin1')

# sum_all=0
# print(len(topk_dets))
# print(len(rels_joined_merged_val_idx))
# for ind in range(len(rels_joined_merged_val_idx)):
#     n_ind= Visualize_Detections(ind,topk_dets_val, rels_joined_merged_val_idx, 'person','*','*', '(?!person)', '*', '*')
#     sum_all = sum_all+n_ind
#     if n_ind>0:
#         input('Press enter to continue: ')
#         x=1

# print('sum_all = ', sum_all)
data_len = len(topk_dets['gt_labels_sbj'])
print('topk_dets:', data_len)
for ind in tqdm(range(data_len)):
    n_ind = Visualize_Detections(ind, topk_dets, rels_joined_merged_test_idx, '*', '*', '*')
    #if n_ind > 0:
    #    input('Press enter to continue: ')

# relationships_train  = relationships[indices_train]'

_data_path = base_data_path + 'Visual_Genome/'
_object_classes = []
with open(_data_path + '/object_categories_spo_joined_and_merged.txt') as obj_classes:
    for line in obj_classes:
        _object_classes.append(line[:-1])
_num_object_classes = len(_object_classes)
_object_class_to_ind = dict(zip(_object_classes, range(_num_object_classes)))

_predicate_classes = ['__background__']
with open(_data_path + '/predicate_categories_spo_joined_and_merged.txt') as prd_classes:
    for line in prd_classes:
        _predicate_classes.append(line[:-1])
_num_predicate_classes = len(_predicate_classes)
_predicate_class_to_ind = dict(zip(_predicate_classes, range(_num_predicate_classes)))

print(_num_object_classes)
print(_num_predicate_classes)
# print(rels_joined_merged[0]['relationships'])

for i, rel in enumerate(rels_joined_merged[0]['relationships']):
    print(i, rel)
    break

# In[81]:



