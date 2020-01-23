
# coding: utf-8

# ## Analyze the behavior of various networks predicting S,O,P triplets; while paying attention to their comparative perfomance at the long-tail of the training distribution.

# In[5]:

import numpy as np
import pandas as pd
import os.path as osp
# import seaborn as sns # not critical.
import matplotlib.pylab as plt


# In[9]:

import os
import re
def files_in_subdirs(top_dir, search_pattern):  # TODO-Sherif: organize project as proper
    join = os.path.join                         # python module (e.g. see https://docs.python-guide.org/writing/structure/) then move this function
    regex = re.compile(search_pattern)          # e.g. in the helper.py
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = join(path, name)
            if regex.search(full_name):
                yield full_name


# In[10]:

top_data_dir = '/home/x_abdelks/c2044/Large_Scale_VRD_pytorch/Outputs/'

dataset = 'gvqa'
# # dataset = 'vg_wiki_and_relco'
#
# split = 'test'
# top_data_dir = osp.join(top_data_dir, 'ltvrd/reports/{}/{}'.format(dataset, split))


# In[11]:

# if split == 'val' and dataset == 'gvqa':
#     top_data_dir = osp.join(top_data_dir, '_baseline_hubness_hubness10k_hubness50k_focal_loss_g025_focal_loss_g1_focal_loss_g2_focal_loss_g5_focal_loss_g10_focal_loss_g50')


# In[12]:

all_csvs = np.array([f for f in files_in_subdirs(top_data_dir, '.csv$')])
method_names = np.array([osp.basename(f)[1:-len('.csv')] for f in all_csvs])

sids = np.argsort(method_names)
all_csvs = all_csvs[sids]
method_names = method_names[sids]

print ('Found {} methods.'.format(len(method_names)))


# In[13]:

apply_mask = True

methods_to_keep = ['baseline', 'focal_loss_g025', 'focal_loss_g1', 'focal_loss_g10',
                   'focal_loss_g2', 'focal_loss_g5', 'focal_loss_g50', 'hubness',
                   'hubness10k', 'hubness50k']

if apply_mask:
    keep_mask = np.zeros(len(method_names), dtype=np.bool)
    for i, m in enumerate(method_names):
        if m in methods_to_keep:
            keep_mask[i] = True
    all_csvs = all_csvs[keep_mask]
    method_names = method_names[keep_mask]

    print ('Kept methods', len(method_names))


# In[14]:

# Load meta-information (integer to "human" readable names)
if dataset == 'gvqa':
    meta_file_name = 'GVQA'
else:
    meta_file_name = 'Visual_Genome'
    
obj_names = '/ibex/scratch/x_abdelks/Large-Scale-VRD/datasets/large_scale_VRD/{}/object_categories_spo_joined_and_merged.txt'.format(meta_file_name)
rel_names = '/ibex/scratch/x_abdelks/Large-Scale-VRD/datasets/large_scale_VRD/{}/predicate_categories_spo_joined_and_merged.txt'.format(meta_file_name)

with open(obj_names) as fin:
    ids_to_obj_names = fin.readlines()
    ids_to_obj_names = {i:idx.rstrip() for i, idx in enumerate(ids_to_obj_names)}

with open(rel_names) as fin:
    ids_to_rel_names = fin.readlines()
    ids_to_rel_names = {i:idx.rstrip() for i, idx in enumerate(ids_to_rel_names)}    


# In[15]:

## Expected columns names of .csv
relation_prefix = 'rel'
object_prefix = 'obj'
subject_prefix = 'sbj'
gt_prefix = 'gt'
all_prediction_types  = [relation_prefix, object_prefix, subject_prefix]
raw_metrics = ['top1', 'top5', 'top10']


# In[16]:

def keep_only_heavy_tail_observations(dataframe, prediction_type, threshold_of_tail):
    df = dataframe.copy()
    freqs = df[[gt_prefix + '_' + prediction_type, prediction_type + '_freq_gt']]
    unique_freqs = freqs.groupby(gt_prefix + '_' + prediction_type).mean() # assumes same
    unique_freqs = unique_freqs.sort_values(prediction_type + '_freq_gt', ascending=False)
    n_total_occurences = unique_freqs.sum()
    unique_freqs[prediction_type + '_freq_gt'] /= float(n_total_occurences)
    valid = unique_freqs[unique_freqs.cumsum()[prediction_type + '_freq_gt'] > threshold_of_tail].index
    df = df[df[gt_prefix + '_' + prediction_type].isin(valid)]
    return df


# In[17]:

# Print basic statistics (ignore tail behavior, just overall average.)
verbose = True
collected_simple_means = dict()
collected_per_class_means = dict()
drop_left_right = False

for i, m in enumerate(method_names):
    if verbose:
        print('=============================')
        print(m)
    df = pd.read_csv(all_csvs[i])
    
    if drop_left_right and dataset == 'gvqa':
        print ('dropping left/right')
        df = df[(df.gt_rel != 286) & (df.gt_rel != 35)]
        
    for metric_type in raw_metrics:
        if verbose:
            print('------', metric_type, '------')
        for prediction_type in all_prediction_types:
            mu = df[prediction_type + '_' + metric_type].mean() * 100
            
            if verbose:
                print ('simple-average', prediction_type, '{:2.2f}'.format(mu))
            
            collected_simple_means[(m, prediction_type, metric_type)] = mu
            
            mu = df.groupby(gt_prefix + '_' + prediction_type)[prediction_type + '_' + metric_type].mean().mean() * 100
            
            if verbose:
                print ('per-class-average', prediction_type, '{:2.2f}'.format(mu))
                
            collected_per_class_means[(m, prediction_type, metric_type)] = mu                        


# In[10]:

# make latex with simple and per-class means.
for means, out_name in zip([collected_simple_means, collected_per_class_means], ['simple', 'per_class']):
    fout = open('{}_{}_mean_analysis.tex'.format(dataset, out_name), 'w')    
    ndf = pd.Series(means).reset_index()
    ndf.columns = ['method', 'prediction-type', 'metric', 'accuracy']
    for metric in raw_metrics:
        df = ndf[ndf['metric'] == metric]
        for prediction in all_prediction_types:
            tdf = df[df['prediction-type'] == prediction]            
            tdf.to_latex(buf=fout,
                         index=False,
                         columns=['method', 'prediction-type', 'metric', 'accuracy'], 
                         float_format="{:0.2f}".format)
    fout.close()

# exit()
# # In[18]:
#
# ## Auxiliary to work with the soft-metrics (as given by Mohamed)##
# # # and to link them to make plots that show long-tail responses.
# ## UNCOMMENT, and run once to create/save the xxx_train_freq.csv
# # df = pd.read_csv(all_csvs[0])
#
# # ndf = df[['gt_rel', 'rel_freq_gt']].groupby('gt_rel').mean().reset_index()
# # ndf['gt_rel'] = ndf['gt_rel'].apply(lambda x: ids_to_rel_names[x])
# # ndf = ndf.sort_values('rel_freq_gt', ascending=False)
# # ndf.to_csv('/home/optas/DATA/OUT/ltvrd/gvqa_relation_to_train_freq.csv', index=False)
#
# # ndf = df[['gt_sbj', 'sbj_freq_gt']].groupby('gt_sbj').mean().reset_index()
# # ndf['gt_sbj'] = ndf['gt_sbj'].apply(lambda x: ids_to_obj_names[x])
# # ndf = ndf.sort_values('sbj_freq_gt', ascending=False)
# # ndf.to_csv('/home/optas/DATA/OUT/ltvrd/gvqa_subject_to_train_freq.csv', index=False)
#
#
# # In[19]:
#
# # Just load the csv of one method, to use the frequency characteristics of the
# # the training data to plot your distributions.
# cum_dists = []
# all_figs = []
#
# for prediction_type in all_prediction_types:
#     df = pd.read_csv(all_csvs[0])
#     gb = df.groupby([gt_prefix + '_' +prediction_type]).groups
#
#     if prediction_type == 'rel':
#         id_to_name = ids_to_rel_names
#     else:
#         id_to_name = ids_to_obj_names
#
#     stats = []  # <name, train_freq - times it is found in the test-data>
#     for key, val in gb.iteritems():
#         gt_freq = np.unique(df[prediction_type + '_freq_gt'].loc[val])
#         assert len(gt_freq) == 1 # we grouped by the gt-type,
#                                  # hence the frequencies must be the same for all rows.
#         stats.append([id_to_name[key], gt_freq[0], len(val)])
#     sl = sorted(stats, key=lambda x: x[1], reverse=True)
#     print('{}: Most heavy items:'.format(prediction_type),  sl[:10])
#
#     freqs = np.array([x[1] for x in stats], dtype=np.float64)
#     freqs = sorted(freqs, reverse=True)
#     freqs /= np.sum(freqs)
#     cum_dist = np.cumsum(freqs)
#     cum_dists.append(cum_dist)
#
#     vals = []
#     for i, v in enumerate(sorted([x[1] for x in stats], reverse=True)):
#         vals.extend([i] * v )
#
#     fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
#     plt.suptitle(prediction_type)
#     sns.distplot(vals, ax=ax[0])
#     sns.distplot(vals, hist_kws=dict(cumulative=True), ax=ax[1])
#     all_figs.append(fig)
#
# for fig in all_figs:
#     fig
#
#
# # In[100]:
#
# # # Compare the accuracy of two methods:
# ## Input on soft-metrics (computed separately)
#
# method_a = 'hubness'
# method_b = 'baseline'
# sim = 'word2vec_visualVG'
#
# # in_f1 = '/home/optas/DATA/OUT/ltvrd/sorted_via_soft_scores/subject/_hubness10k_vgqa_sorted_mean_acc_per_class_on_subject_with_word2vec_visualVG.csv'
# # in_f2 = '/home/optas/DATA/OUT/ltvrd/sorted_via_soft_scores/subject/_baseline_vgqa_sorted_mean_acc_per_class_on_subject_with_word2vec_visualVG.csv'
#
# in_f1 = '/home/optas/DATA/OUT/ltvrd/sorted_via_soft_scores/relation/_baseline_vgqa_sorted_mean_acc_per_class_on_relation_with_word2vec_visualVG.csv'
# in_f2 = '/home/optas/DATA/OUT/ltvrd/sorted_via_soft_scores/relation/_hubness_vgqa_sorted_mean_acc_per_class_on_relation_with_word2vec_visualVG.csv'
#
# s0=pd.read_csv(in_f1)[sim]
# s1=pd.read_csv(in_f2)[sim]
#
# relative_score =  s0 - s1
# relative_score = relative_score.values
# n_obs = len(relative_score)
#
# fig, ax = plt.subplots(1, 1, figsize = (12, 5), dpi=300)
# # plt.title('{}--{}'.format(dataset, prediction_type))
#
# x0 = np.arange(n_obs)[relative_score > 0]
# x1 = np.arange(n_obs)[relative_score < 0]
#
# # sns.scatterplot(x0, np.ones(n_obs)[relative_score > 0], color='green')
# # sns.scatterplot(x1, -np.ones(n_obs)[relative_score < 0], color='red')
#
# # sns.scatterplot(x0, relative_score[relative_score > 0], color='green')
# # sns.scatterplot(x1, relative_score[relative_score < 0], color='red')
#
# sns.regplot(x0, relative_score[relative_score > 0], color='green')#, lowess=True)
# sns.regplot(x1, relative_score[relative_score < 0], color='red')#, lowess=True)
#
# plt.title('Hubness vs. Baseline: relation prediction, entire class distribution.')
# plt.legend([method_a, method_b], loc='upper left', fontsize=14)
# plt.xlabel('Class ID (decreasing training frequency)', fontsize=15)
# plt.grid()
# plt.ylabel('Relative Performance', fontsize=15)
#
# ### Some other statistics on the "relative" curve:
# # from scipy.stats import pearsonr
# # x = np.arange(n_obs)[relative_score < 0]
# # y = relative_score[relative_score < 0]
# # rho_neg = pearsonr(x, y)[0]
# # x = np.arange(n_obs)[relative_score > 0]
# # y = relative_score[relative_score > 0]
# # rho_pos = pearsonr(x, y)[0]
# # print rho_neg, rho_pos, rho_pos + rho_neg
# # print np.mean(relative_score > 0)
#
#
# # In[76]:
#
# threshold_of_tail = 0.
# metric = 'top1'
#
# for i, m in enumerate(method_names):
#     print(m)
#     df = pd.read_csv(all_csvs[i])
#     for prediction_type in all_prediction_types:
#         ndf = keep_only_heavy_tail_observations(df, prediction_type, threshold_of_tail=threshold_of_tail)
#         print prediction_type, ndf[prediction_type + '_' + metric].mean()
#     print
#
#
# # In[ ]:
#
# # Below is now dirty.
#
#
# # In[527]:
#
# fig, ax = plt.subplots(1, 1, figsize = (10, 10), dpi=200)
# plt.title('{}--{}'.format(dataset, prediction_type))
#
# rel = g2[prediction_type+ '_top1'] - g1[prediction_type + '_top1']
# # sns.scatterplot(np.arange(len(g1))[rel > 0], rel[rel>0], color='green', )
#
# plt.bar(np.arange(len(g1))[rel > 0], rel[rel>0], color='green')
# # sns.barplot(np.arange(len(g1))[rel > 0], rel[rel>0], color='green')
# # sns.regplot(np.arange(len(g1))[rel > 0], rel[rel>0], color='green')
# # plt.scatter(np.arange(len(g1))[rel < 0], rel[rel<0], color='red')
# # sns.barplot(np.arange(len(g1))[rel < 0], rel[rel<0], color='red')
# plt.bar(np.arange(len(g1))[rel < 0], rel[rel<0], color='red')
#
# # sns.regplot(np.arange(len(g1))[rel < 0], rel[rel<0], color='red')
# plt.legend(['improvement', 'worsening'])
# plt.xlabel('Class id (decreasing training frequency)', fontsize=15)
# plt.ylabel('Relative probility of hubness vs. baseline', fontsize=15)
#
#
# # In[ ]:
#
# # fig, ax = plt.subplots(1, 1, figsize = (10, 15), dpi=300)
#
# # s = sns.barplot(x=range(1, 2*len(g1), 2), y=prediction_type+'_top1', data=g1, facecolor='red')
# # s.set(xticklabels=[])
#
# # s = sns.barplot(x=range(2, 2*len(g2)+1, 2), y=prediction_type+'_top1', data=g2, facecolor='green')
# # s.set(xticklabels=[])
# # ax.set_xlabel('')
#
#
# # In[30]:
#
# # sns.lineplot(x=range(len(g1)), y=prediction_type+'_top1', data=g1, color='red')
#
#
# # In[350]:
#
# plt.figure(figsize=(20, 10))
# sns.scatterplot(x=range(1, 5*len(g1), 10), y=prediction_type+'_top1', data=g1, marker='.', color='red', s=60)
# sns.scatterplot(x=range(4, 5*len(g2) +1, 5), y=prediction_type+'_top1', data=g2, marker='.', color='green', s=60)
#
#
# # In[351]:
#
# plt.figure(figsize=(20, 10))
# sns.scatterplot(x=range(4, 5*len(g2) +1, 5), y=prediction_type+'_top1', data=g2, marker='.', color='green', s=60)
# sns.scatterplot(x=range(1, 5*len(g1), 5), y=prediction_type+'_top1', data=g1, marker='.', color='red', s=60)
#
#
# # In[349]:
#
# plt.figure(figsize=(20, 10))
# sns.scatterplot(x=range(1, 2*len(g1), 2), y=prediction_type+'_top1', data=g1, marker='.', color='green', s=60)
# sns.scatterplot(x=range(2, 2*len(g2) +1, 2), y=prediction_type+'_top1', data=g2, marker='.', color='red', s=60)
#
#
# # In[31]:
#
# if prediction_type == 'rel':
#     manual_bins = np.array([100000, 25000, 10000, 1000, 500, 100, 10, 1])
#
# if prediction_type == 'obj' or prediction_type == 'sbj':
#     manual_bins = np.array([200000, 100000, 50000, 25000, 10000, 5000, 1000, 500, 100, 25, 1])
#
#
# # In[39]:
#
# method_names
#
#
# # In[40]:
#
# distance = 'top1' # or top5, or #top10
# collected_results = []
# use_only=['baseline', 'focal_loss_g025', 'hubness10k']
#
# for i in range(len(method_names)):
#     if method_names[i] not in use_only:
#         continue
#     df = pd.read_csv(all_csvs[i])
#     print (method_names[i], 'n_lines=', len(df))
#     metric = prediction_type + '_' + distance
#     bin_values = np.digitize(df[prediction_type + '_freq_gt'], manual_bins)
#     df['bins'] = bin_values
# #     print df.groupby(['bins'])[metric].mean()[1:]
# #     collected_results.append(np.array(df.groupby(['bins'])[metric].mean()[1:]))
#     collected_results.append(np.array(df.groupby(['bins'])[metric].mean()))
#
#
# # In[50]:
#
# plt.figure(figsize=(9, 6))
# plt.grid()
# plt.xlabel('Frequency-based (sorted) logarithimic bins.', fontsize=18)
# plt.ylabel('Top-1 Accuracy.', fontsize=18)
# # f, ax = plt.subplots(1, 1)
# for i, experiment in enumerate(collected_results):
#     if i == 0:
#         plt.plot(np.arange(len(experiment)), experiment, '--')
#     else:
#         plt.plot(np.arange(len(experiment)), experiment, marker='.')
#
# #     sns.lineplot(x=np.arange(len(experiment)), y=experiment, ax=ax , markers='*')
# plt.legend(use_only, fontsize=18)

