import pandas as pd
import numpy as np
from collections import defaultdict


# in_csv_file = '_baseline/test/subjects_data_GQA.csv'
#in_csv_file = '_focal_loss_g025/test/subjects_data_GQA.csv'
#in_csv_file = '_focal_loss_g1/test/subjects_data_GQA.csv'
# in_csv_file = '_focal_loss_g2/test/subjects_data_GQA.csv'
#in_csv_file = '_hubness/test/subjects_data_GQA.csv'
#in_csv_file = '_hubness10k/test/subjects_data_GQA.csv'
word2vec_files = ['/ibex/scratch/x_abdelks/MetricAnalysis/VQA_backup/_baseline/test/subjects_data_GQA.csv', 
                # '_focal_loss_g025/test/subjects_data_GQA.csv', 
                # '_focal_loss_g1/test/subjects_data_GQA.csv', 
                '/ibex/scratch/x_abdelks/MetricAnalysis/VQA_backup/_focal_loss_g2/test/subjects_data_GQA.csv', 
                '/ibex/scratch/x_abdelks/MetricAnalysis/VQA_backup/_hubness/test/subjects_data_GQA.csv',
                '/ibex/scratch/x_abdelks/MetricAnalysis/VQA_backup/_hubness10k/test/subjects_data_GQA.csv']

jcn_files = ['/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_baseline/test/subjects_data_GQA.csv', 
            '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_focal_loss_g2/test/subjects_data_GQA.csv', 
            '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_hubness/test/subjects_data_GQA.csv',
            '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_hubness10k/test/subjects_data_GQA.csv']
both_files = ['/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_focal_loss_g50/test/subjects_data_GQA.csv']
 	#'/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_focal_loss_g025/test/subjects_data_GQA.csv', 
               #'/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_focal_loss_g1/test/subjects_data_GQA.csv', 
               #'/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_hubness50k/test/subjects_data_GQA.csv']

all_files = [
#'/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_baseline/test/predicates_data_GQA.csv',
            '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_focal_loss_g025/test/predicates_data_GQA.csv',
            '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_focal_loss_g1/test/predicates_data_GQA.csv',
            '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_focal_loss_g2/test/predicates_data_GQA.csv',
            '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_focal_loss_g5/test/predicates_data_GQA.csv',
            '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_focal_loss_g10/test/predicates_data_GQA.csv',
            '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_focal_loss_g50/test/predicates_data_GQA.csv',
            '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_hubness/test/predicates_data_GQA.csv',
            '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_hubness10k/test/predicates_data_GQA.csv',
            '/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_hubness50k/test/predicates_data_GQA.csv']
#both_files = ['/ibex/scratch/x_abdelks/MetricAnalysis/VQA/_hubness10k/test/predicates_data_GQA.csv']
#in_csv_files = word2vec_files
in_csv_files = both_files
#in_csv_files = both_files
## TODO Add the last (fourth similarity)
similarities_used = ['jcn_similarity', 'word2vec_GNews', 'word2vec_visualVG']
#similarities_used = ['word2vec_GNews', 'word2vec_visualVG']
# similarities_used = ['jcn_similarity']

for in_csv_file in in_csv_files:
    print('Reading csv file:', in_csv_file)
    df = pd.read_csv(in_csv_file)
    print('Success!')

   ## Clip similarities to [-1, 1] range.
    for sim in similarities_used:
        too_large = df[sim] > 1    
        df.loc[too_large, sim] = 1
        too_small = df[sim] < -1
        df.loc[too_small, sim] = -1

    print('Average Rank (ignoring misses >250, with pandas)', df['i'][df['exact_match'] == 1].mean())


    ### Ugly (but well-understood) numpy code.
    hits = []
    hits_per_class = defaultdict(list)
    totally_missed = 0
    miss_penalty = 250 # penalty if you didn't find it in the 250-cases

    for k in range(0, len(df)-250, 250):
        
        i_th_slice = df.loc[k: k+250-1]
        assert i_th_slice['i'].min() == 0
        assert i_th_slice['i'].max() == 249    
        
        hit = np.where(i_th_slice['exact_match'] == 1.0)[0]    
        assert len(hit) in [0, 1]
        
        i_th_class = i_th_slice['gold'].unique()
        assert len(i_th_class) == 1
        i_th_class = i_th_class[0]
        
        if len(hit) == 1:
            hit = hit[0]
            hits.append(hit)
            hits_per_class[i_th_class].append(hit)
        else:
            totally_missed += 1
            if miss_penalty > 0:
                hits_per_class[i_th_class].append(miss_penalty)
            
    print('Average Rank (ignoring misses > 250):  {:.3f}'.format(np.mean(hits)))
    print('Total observations:', len(hits))
    print('Missed in top 250:', totally_missed)
    penalty_mu = np.mean(hits + [miss_penalty] * totally_missed)
    print('Average Rank (with miss penalty {}): {:.3f}'.format(miss_penalty, penalty_mu))

    p_class_mu = []
    for h in hits_per_class:
        p_class_mu.append(np.mean(hits_per_class[h]))    
    print('Per Class Average Rank (with miss penalty {})  {:.3f}'.format(miss_penalty, np.mean(p_class_mu)))
    ### <end> Ugly numpy code.


    g = df.groupby(['gold'])
    means_of_groups = []
    for k, v in g.groups.items():
        group_content = df.loc[v]
        group_i = group_content[group_content['exact_match'] == 1.0]['i']
        if len(group_i) > 0:
            group_mean = group_i.mean()
            means_of_groups.append(group_mean)
        else:
            pass # Missed element, ignore.
            
    print ('Per-class Average Rank:', np.mean(means_of_groups))


    for sim in similarities_used:
        mu = df[df['i'] == 0][sim].mean()
        print ('Average similarity per metric at Top-0', sim, '{:.3f}'.format(mu))
               
    for sim in similarities_used:    
        for k in [1]:
            ndf = df[df['i'] < k]
            max_per_image_gold = ndf.groupby(['image_id', 'gold'])[sim].max() # Missed Corner-case (bag/bag in same image.)
            mu = max_per_image_gold.groupby('gold').mean().mean()
            print ('Per-class similarity per metric at Top-{}'.format(k), sim, '{:.3f}'.format(mu))
