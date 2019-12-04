'''
# How can we do the long tail with the soft? 
# we pick the winner from our table (most likely a hubness) and the baseline.
# I want a csv that tells me for every predition (rel. or sbj.)
# a) its class, (optionally the class-frequency), and the top-1 or top-5 [choose]
# score per a soft matrix.
'''

import pandas as pd
import os.path  as osp

#method_name = '_hubness10k'
method_name = '_hubness'
#method_name = '_baseline'
workon = 'relation' # or 'relations'
#in_csv_file = '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_focal_loss_g025/test/subjects_data_GQA.csv'  # method
in_csv_file = '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/{}/test/predicates_data_GQA.csv'.format(method_name)  # method
#similarities_used = ['word2vec_GNews'] # pick one
similarities_used = ['word2vec_visualVG'] # pick one
top_aux_data_dir = './' # where are the freq. dictionaries
print(method_name, similarities_used, workon)
df = pd.read_csv(in_csv_file)
## Clip similarities to [-1, 1] range.
for sim in similarities_used:
    too_large = df[sim] > 1    
    df.loc[too_large, sim] = 1
    too_small = df[sim] < -1
    df.loc[too_small, sim] = -1

freq_info = pd.read_csv(osp.join(top_aux_data_dir, 'gvqa_{}_to_train_freq.csv'.format(workon)))
freq_info_dict = dict()

if workon == 'subject':
    x, y = freq_info.gt_sbj, freq_info.sbj_freq_gt
elif workon == 'relation':
    x, y = freq_info.gt_rel, freq_info.rel_freq_gt

for k, v in zip(x, y):
    freq_info_dict[k] = v

ndf = df[df['i'] == 0][[sim, 'gold']]  # top-1
g = ndf.groupby('gold')
average_per_class = g[sim].mean().reset_index()
average_per_class['frequency'] = average_per_class['gold'].apply(lambda x: freq_info_dict[x])
average_per_class = average_per_class.sort_values('frequency', ascending=False)
average_per_class.to_csv('{}_vgqa_sorted_mean_acc_per_class_on_{}_with_{}.csv'.format(method_name, workon, sim))
