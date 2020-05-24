# Written by Sherif Abdelkarim on Jan 2020

import numpy as np
import pandas as pd
import os.path as osp
# import seaborn as sns # not critical.
import matplotlib.pylab as plt


# In[9]:

import os
import re


def files_in_subdirs(top_dir, search_pattern):  # TODO: organize project as proper
    join = os.path.join                         # python module (e.g. see https://docs.python-guide.org/writing/structure/) then move this function
    regex = re.compile(search_pattern)          # e.g. in the helper.py
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = join(path, name)
            if regex.search(full_name):
                yield full_name


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

def get_many_medium_few_scores(csv_path, cutoffs, syn):
    df = pd.read_csv(csv_path)
    # df = df.groupby(['gt_sbj', 'gt_rel', 'gt_obj']).mea)
    df['box_id'] = df.groupby('image_id').cumcount()
    metric_type = 'top1'
    all_prediction_types = ['rel', 'obj', 'sbj']
    if syn:
        syn_obj = pd.read_csv('./data/gvqa/objects_synsets.csv')
        syn_obj = syn_obj[['object_name', 'synset']]
        syn_obj.set_index('object_name', inplace=True)

        syn_prd = pd.read_csv('./data/gvqa/predicates_synsets.csv')
        syn_prd = syn_prd[['predicate_name', 'synset']]
        syn_prd.set_index('predicate_name', inplace=True)

    for prediction_type in all_prediction_types:
        df[prediction_type + '_' + metric_type] = df[prediction_type + '_rank'] < int(metric_type[3:])

    if syn:
        for prediction_type in ['sbj', 'obj']:
            df['gt_' + prediction_type + '_syn'] = syn_obj.loc[df['gt_' + prediction_type], 'synset'].to_list()
            df['det_' + prediction_type + '_syn'] = syn_obj.loc[df['det_' + prediction_type], 'synset'].to_list()
            df[prediction_type + '_top1_syn'] = df['gt_' + prediction_type + '_syn'] == df['det_' + prediction_type + '_syn']

        for prediction_type in ['rel']:
            df['gt_' + prediction_type + '_syn'] = syn_prd.loc[df['gt_' + prediction_type], 'synset'].to_list()
            df['det_' + prediction_type + '_syn'] = syn_prd.loc[df['det_' + prediction_type], 'synset'].to_list()
            df[prediction_type + '_top1_syn'] = df['gt_' + prediction_type + '_syn'] == df['det_' + prediction_type + '_syn']

    df['triplet_top1'] = df['rel_top1'] & df['sbj_top1'] & df['obj_top1']
    if syn:
        df['triplet_top1_syn'] = df['rel_top1_syn'] & df['sbj_top1_syn'] & df['obj_top1_syn']

    cutoff, cutoff_medium = cutoffs

    a = df.groupby('gt_rel').mean()
    classes_rel = (list(a.sort_values('rel_freq_gt').index))
    freqs_rel = (list(a.sort_values('rel_freq_gt')['rel_freq_gt']))
    classes_rel_few = classes_rel[:int(len(classes_rel)*cutoff)]
    classes_rel_medium = classes_rel[int(len(classes_rel)*cutoff):int(len(classes_rel)*cutoff_medium)]
    classes_rel_many = classes_rel[int(len(classes_rel)*cutoff_medium):]
    # freqs_rel = freqs_rel[:int(len(classes_rel)*cutoff)]

    a = df.groupby('gt_sbj').mean()
    classes_sbj = (list(a.sort_values('sbj_freq_gt').index))
    freqs_sbj = (list(a.sort_values('sbj_freq_gt')['sbj_freq_gt']))
    classes_sbj_few = classes_sbj[:int(len(classes_sbj)*cutoff)]
    classes_sbj_medium = classes_sbj[int(len(classes_sbj)*cutoff):int(len(classes_sbj)*cutoff_medium)]
    classes_sbj_many = classes_sbj[int(len(classes_sbj)*cutoff_medium):]
    # freqs_sbj = freqs_sbj[:int(len(classes_sbj)*cutoff)]

    a = df.groupby('gt_obj').mean()
    classes_obj = (list(a.sort_values('obj_freq_gt').index))
    freqs_obj = (list(a.sort_values('obj_freq_gt')['obj_freq_gt']))
    classes_obj_few = classes_obj[:int(len(classes_obj)*cutoff)]
    classes_obj_medium = classes_obj[int(len(classes_obj)*cutoff):int(len(classes_obj)*cutoff_medium)]
    classes_obj_many = classes_obj[int(len(classes_obj)*cutoff_medium):]
    # freqs_obj = freqs_obj[:int(len(classes_obj)*cutoff)]

    df_few_rel = df[df['gt_rel'].isin(classes_rel_few)]
    df_medium_rel = df[df['gt_rel'].isin(classes_rel_medium)]
    df_many_rel = df[df['gt_rel'].isin(classes_rel_many)]

    df_few_sbj = df[df['gt_sbj'].isin(classes_sbj_few)]
    df_medium_sbj = df[df['gt_sbj'].isin(classes_sbj_medium)]
    df_many_sbj = df[df['gt_sbj'].isin(classes_sbj_many)]

    df_few_obj = df[df['gt_obj'].isin(classes_obj_few)]
    df_medium_obj = df[df['gt_obj'].isin(classes_obj_medium)]
    df_many_obj = df[df['gt_obj'].isin(classes_obj_many)]


    # print('sbj_overall_top1', num(df_['sbj_top1'].mean() * 100.))
    # print('obj_overall_top1', num(df['obj_top1'].mean() * 100.))
    # print('rel few:', len(df_few_rel))
    # print('rel medium:',len(df_medium_rel))
    # print('rel many:', len(df_many_rel))
    #
    # print('sbj few:', len(df_few_sbj))
    # print('sbj medium:',len(df_medium_sbj))
    # print('sbj many:', len(df_many_sbj))
    #
    # print('obj few:', len(df_few_obj))
    # print('obj medium:',len(df_medium_obj))
    # print('obj many:', len(df_many_obj))
    # print('all:', len(df))
    # print()
    print('Many, Medium, Few accuracy scores using exact matching:')

    print('rel many:', df_many_rel['rel_top1'].mean() * 100.)
    print('rel med:', df_medium_rel['rel_top1'].mean() * 100.)
    print('rel few:', df_few_rel['rel_top1'].mean() * 100.)
    print('rel all:', df['rel_top1'].mean() * 100.)
    print()
    print('sbj many:', df_many_sbj['sbj_top1'].mean() * 100.)
    print('sbj med:', df_medium_sbj['sbj_top1'].mean() * 100.)
    print('sbj few:', df_few_sbj['sbj_top1'].mean() * 100.)
    print('sbj all:', df['sbj_top1'].mean() * 100.)
    print()
    print('obj man:', df_many_obj['obj_top1'].mean() * 100.)
    print('obj med:', df_medium_obj['obj_top1'].mean() * 100.)
    print('obj few:', df_few_obj['obj_top1'].mean() * 100.)
    print('obj all:', df['obj_top1'].mean() * 100.)
    print()
    # print('triplet accuracy few:', df_few_rel['triplet_top1'].mean() * 100.)
    # print('triplet accuracy med:', df_medium_rel['triplet_top1'].mean() * 100.)
    # print('triplet accuracy man:', df_many_rel['triplet_top1'].mean() * 100.)
    # print('triplet accuracy all:', df['triplet_top1'].mean() * 100.)
    # print('=========================================================')
    if syn:
        print('Many, Medium, Few accuracy scores using synset matching:')
        print('rel syn many:', df_many_rel['rel_top1_syn'].mean() * 100.)
        print('rel syn med:', df_medium_rel['rel_top1_syn'].mean() * 100.)
        print('rel syn few:', df_few_rel['rel_top1_syn'].mean() * 100.)
        print('rel syn all:', df['rel_top1_syn'].mean() * 100.)
        print()
        print('sbj syn many:', df_many_sbj['sbj_top1_syn'].mean() * 100.)
        print('sbj syn med:', df_medium_sbj['sbj_top1_syn'].mean() * 100.)
        print('sbj syn few:', df_few_sbj['sbj_top1_syn'].mean() * 100.)
        print('sbj syn all:', df['sbj_top1_syn'].mean() * 100.)
        print()
        print('obj syn many:', df_many_obj['obj_top1_syn'].mean() * 100.)
        print('obj syn med:', df_medium_obj['obj_top1_syn'].mean() * 100.)
        print('obj syn few:', df_few_obj['obj_top1_syn'].mean() * 100.)
        print('obj syn all:', df['obj_top1_syn'].mean() * 100.)
        print()

    # print('triplet accuracy few:', df_few_rel['triplet_top1_syn'].mean() * 100.)
    # print('triplet accuracy med:', df_medium_rel['triplet_top1_syn'].mean() * 100.)
    # print('triplet accuracy man:', df_many_rel['triplet_top1_syn'].mean() * 100.)
    # print('triplet accuracy all:', df['triplet_top1_syn'].mean() * 100.)
    # print('=========================================================')

def get_wordsim_metrics_from_csv(csv_file):
    verbose = True
    collected_simple_means = dict()
    collected_per_class_means = dict()
    print('Reading csv file')
    df = pd.read_csv(csv_file)
    print('Done')
    # wordnet_metrics = ['lch', 'wup', 'res', 'jcn', 'lin', 'path']
    wordnet_metrics = ['lch', 'wup', 'lin', 'path']
    word2vec_metrics = ['w2v_gn']
    gt_prefix = 'gt'

    for prediction_type in ['sbj']:
        for metric_type in wordnet_metrics + word2vec_metrics:
            mu = df[prediction_type + '_' + metric_type].mean()

            if verbose:
                print('overall', prediction_type, metric_type, '{:2.2f}'.format(mu))

            collected_simple_means[(csv_file, prediction_type, metric_type)] = mu

    for prediction_type in ['rel']:
        for metric_type in  word2vec_metrics:
            mu = df[prediction_type + '_' + metric_type].mean()

            if verbose:
                print('overall', prediction_type, metric_type, '{:2.2f}'.format(mu))

            collected_simple_means[(csv_file, prediction_type, metric_type)] = mu

    for prediction_type in ['sbj', 'obj']:
        for metric_type in wordnet_metrics + word2vec_metrics:
            mu = df.groupby(gt_prefix + '_' + prediction_type)[prediction_type + '_' + metric_type].mean().mean()

            if verbose:
                print('per-class', prediction_type, metric_type, '{:2.2f}'.format(mu))

            collected_per_class_means[(csv_file, prediction_type, metric_type)] = mu

    for prediction_type in ['rel']:
        for metric_type in word2vec_metrics:
            mu = df.groupby(gt_prefix + '_' + prediction_type)[prediction_type + '_' + metric_type].mean().mean()

            if verbose:
                print('per-class', prediction_type, metric_type, '{:2.2f}'.format(mu))

            collected_per_class_means[(csv_file, prediction_type, metric_type)] = mu
    return collected_simple_means, collected_per_class_means


def get_metrics_from_csv(csv_file, get_mr=False):
    verbose = True
    collected_simple_means = dict()
    collected_per_class_means = dict()
    print('Reading csv file')
    df = pd.read_csv(csv_file)
    print('Done')
    # df['rel_top1'] = df['rel_rank'] < 1
    metric_type = 'top1'
    all_prediction_types = ['rel', 'obj', 'sbj']
    gt_prefix = 'gt'
    for prediction_type in all_prediction_types:
        df[prediction_type + '_' + metric_type] = df[prediction_type + '_rank'] < int(metric_type[3:])

    df['triplet_top1'] = df['rel_top1'] & df['sbj_top1'] & df['obj_top1']

    if verbose:
        print('------', metric_type, '------')

    # Overall Accuracy
    for prediction_type in all_prediction_types:
        mu = (len(df[df[prediction_type + '_rank'] < int(metric_type[3:])]) / len(df)) * 100.0
        # mu = df[prediction_type + '_' + metric_type].mean() * 100

        if verbose:
            print('simple-average', prediction_type, '{:2.2f}'.format(mu))

        collected_simple_means[(csv_file, prediction_type, metric_type)] = mu
    print()
    if get_mr:
        # Overall Mean Rank
        for prediction_type in all_prediction_types:
            mu = df[prediction_type + '_rank'].mean() * 100.0 / 250.0
            # mu = df.groupby(gt_prefix + '_' + prediction_type)[prediction_type + '_rank'].mean()
            # print(mu)

            if verbose:
                print('overall-mr', prediction_type, '{:2.2f}'.format(mu))

            # collected_per_class_means[(csv_file, prediction_type, metric_type)] = mu
    print()
    # Per-class Accuracy
    for prediction_type in all_prediction_types:
        mu = df.groupby(gt_prefix + '_' + prediction_type)[prediction_type + '_' + metric_type].mean().mean() * 100
        # mu = df.groupby(gt_prefix + '_' + prediction_type)[prediction_type + '_rank'].mean()
        # print(mu)

        if verbose:
            print('per-class-average', prediction_type, '{:2.2f}'.format(mu))

        collected_per_class_means[(csv_file, prediction_type, metric_type)] = mu
    print()
    if get_mr:
        # Per-class Mean Rank
        for prediction_type in all_prediction_types:
            mu = df.groupby(gt_prefix + '_' + prediction_type)[prediction_type + '_rank'].mean().mean() * 100.0 / 250.0
            # mu = df.groupby(gt_prefix + '_' + prediction_type)[prediction_type + '_rank'].mean()
            # print(mu)

            if verbose:
                print('per-class-mr', prediction_type, '{:2.2f}'.format(mu))

            # collected_per_class_means[(csv_file, prediction_type, metric_type)] = mu
    print()

    mu = df['triplet_top1'].mean() * 100.0
    if verbose:
        print('simple-average', 'triplet', '{:2.2f}'.format(mu))

    for prediction_type in all_prediction_types:
        mu = df.groupby(gt_prefix + '_' + prediction_type)['triplet_top1'].mean().mean() * 100
        if verbose:
            print('per-class-average', 'triplet_' + prediction_type, '{:2.2f}'.format(mu))

    print()

    return collected_simple_means, collected_per_class_means


