# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'
ANN_FN2 = 'annotation_file2'
ANN_FN3 = 'predicate_file'
# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'


print('datasets catalogue', 'cfg.RNG_SEED', cfg.RNG_SEED)
# Available datasets
DATASETS = {
    # VG dataset
    'vg_train': {
        IM_DIR:
            _DATA_DIR + '/vg/VG_100K',
        ANN_FN:
            _DATA_DIR + '/vg/detections_train.json',
        ANN_FN2:
            _DATA_DIR + '/vg/rel_annotations_train.json',
        ANN_FN3:
            _DATA_DIR + '/vg/predicates.json',
    },
    # for now vg_val is identical to vg_test
    'vg_val': {
        IM_DIR:
            _DATA_DIR + '/vg/VG_100K',
        ANN_FN:
            _DATA_DIR + '/vg/detections_val.json',
        ANN_FN2:
            _DATA_DIR + '/vg/rel_annotations_val.json',
        ANN_FN3:
            _DATA_DIR + '/vg/predicates.json',
    },
    'vg80k_train': {
        IM_DIR:
            _DATA_DIR + '/vg80k/VG_100K',
        ANN_FN:
            _DATA_DIR + '/vg80k/seed{}/detections_train.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/vg80k/seed{}/rel_annotations_train.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/vg80k/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'vg80k_val': {
        IM_DIR:
            _DATA_DIR + '/vg80k/VG_100K',
        ANN_FN:
            _DATA_DIR + '/vg80k/seed{}/detections_val.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/vg80k/seed{}/rel_annotations_val.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/vg80k/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'vg80k_test': {
        IM_DIR:
            _DATA_DIR + '/vg80k/VG_100K',
        ANN_FN:
            _DATA_DIR + '/vg80k/seed{}/detections_test.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/vg80k/seed{}/rel_annotations_test.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/vg80k/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },

    'vg8k_train': {
        IM_DIR:
            _DATA_DIR + '/vg8k/VG_100K',
        ANN_FN:
            _DATA_DIR + '/vg8k/seed{}/detections_train.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/vg8k/seed{}/rel_annotations_train.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/vg8k/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'vg8k_val': {
        IM_DIR:
            _DATA_DIR + '/vg8k/VG_100K',
        ANN_FN:
            _DATA_DIR + '/vg8k/seed{}/detections_val.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/vg8k/seed{}/rel_annotations_val.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/vg8k/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'vg8k_test': {
        IM_DIR:
            _DATA_DIR + '/vg8k/VG_100K',
        ANN_FN:
            _DATA_DIR + '/vg8k/seed{}/detections_test.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/vg8k/seed{}/rel_annotations_test.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/vg8k/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'gvqa_train': {
        IM_DIR:
            _DATA_DIR + '/gvqa/images',
        ANN_FN:
            _DATA_DIR + '/gvqa/seed{}/detections_train.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/gvqa/seed{}/rel_annotations_train.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/gvqa/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'gvqa_val': {
        IM_DIR:
            _DATA_DIR + '/gvqa/images',
        ANN_FN:
            _DATA_DIR + '/gvqa/seed{}/detections_val.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/gvqa/seed{}/rel_annotations_val.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/gvqa/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'gvqa_test': {
        IM_DIR:
            _DATA_DIR + '/gvqa/images',
        ANN_FN:
            _DATA_DIR + '/gvqa/seed{}/detections_test.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/gvqa/seed{}/rel_annotations_test.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/gvqa/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'gvqa20k_train': {
        IM_DIR:
            _DATA_DIR + '/gvqa20k/images',
        ANN_FN:
            _DATA_DIR + '/gvqa20k/seed{}/detections_train.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/gvqa20k/seed{}/rel_annotations_train.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/gvqa20k/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'gvqa20k_val': {
        IM_DIR:
            _DATA_DIR + '/gvqa20k/images',
        ANN_FN:
            _DATA_DIR + '/gvqa20k/seed{}/detections_val.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/gvqa20k/seed{}/rel_annotations_val.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/gvqa20k/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'gvqa20k_test': {
        IM_DIR:
            _DATA_DIR + '/gvqa20k/images',
        ANN_FN:
            _DATA_DIR + '/gvqa20k/seed{}/detections_test.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/gvqa20k/seed{}/rel_annotations_test.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/gvqa20k/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'gvqa10k_train': {
        IM_DIR:
            _DATA_DIR + '/gvqa10k/images',
        ANN_FN:
            _DATA_DIR + '/gvqa10k/seed{}/detections_train.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/gvqa10k/seed{}/rel_annotations_train.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/gvqa10k/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'gvqa10k_val': {
        IM_DIR:
            _DATA_DIR + '/gvqa10k/images',
        ANN_FN:
            _DATA_DIR + '/gvqa10k/seed{}/detections_val.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/gvqa10k/seed{}/rel_annotations_val.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/gvqa10k/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    'gvqa10k_test': {
        IM_DIR:
            _DATA_DIR + '/gvqa10k/images',
        ANN_FN:
            _DATA_DIR + '/gvqa10k/seed{}/detections_test.json'.format(cfg.RNG_SEED),
        ANN_FN2:
            _DATA_DIR + '/gvqa10k/seed{}/rel_annotations_test.json'.format(cfg.RNG_SEED),
        ANN_FN3:
            _DATA_DIR + '/gvqa10k/seed{}/predicates.json'.format(cfg.RNG_SEED),
    },
    # VRD dataset
    'vrd_train': {
        IM_DIR:
            _DATA_DIR + '/vrd/train_images',
        ANN_FN:
            _DATA_DIR + '/vrd/detections_train.json',
        ANN_FN2:
            _DATA_DIR + '/vrd/new_annotations_train.json',
        ANN_FN3:
            _DATA_DIR + '/vrd/predicates.json',
    },
    'vrd_val': {
        IM_DIR:
            _DATA_DIR + '/vrd/val_images',
        ANN_FN:
            _DATA_DIR + '/vrd/detections_val.json',
        ANN_FN2:
            _DATA_DIR + '/vrd/new_annotations_val.json',
        ANN_FN3:
            _DATA_DIR + '/vrd/predicates.json',
    },
}
