"""
Dataset Specifics
Extended from ADNet code by Hansen et al.
"""

import torch
import random


def get_label_names(dataset):
    label_names = {}
    if dataset == 'AROIDuke':
        label_names[0] = 'BG'
        label_names[1] = 'tissue'
    elif dataset == 'AROIDuke_ONE_block_paste':
        label_names[0] = 'BG'
        label_names[1] = 'tissue'
        label_names[2] = 'tool'
        label_names[3] = 'artifact'
    elif dataset == 'AROIDuke_RPE_paste':
        label_names[0] = 'BG'
        label_names[1] = 'ILM'
        label_names[1] = 'RPE'
        label_names[2] = 'tool'
        label_names[3] = 'artifact'
    elif dataset == 'horizontal':
        label_names[0] = 'BG'
        label_names[1] = 'tissue'
        label_names[2] = 'tool'
        label_names[3] = 'artifact'
    elif dataset == 'vertical':
        label_names[0] = 'BG'
        label_names[1] = 'tissue'
        label_names[2] = 'tool'
    elif dataset == 'macular':
        label_names[0] = 'BG'
        label_names[1] = 'ILM-RPE'
        label_names[2] = 'RPE'
        label_names[3] = 'tool'
        label_names[4] = 'fragments'
    elif dataset == 'erm':
        label_names[0] = 'BG'
        label_names[1] = 'tissue'
        label_names[2] = 'tool'
        
    return label_names
