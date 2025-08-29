"""
Experiment configuration file
Extended from config file from original PANet Repository
"""
import glob
import itertools
import os
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from util.utils import *
from yacs.config import CfgNode as CN

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment("CDFS")
ex.captured_out_filter = apply_backspaces_and_linefeeds

###### Set up source folder ######
source_folders = ['.', './dataloaders', './models', './utils']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)


@ex.config
def cfg():
    """Default configurations"""
    seed = 2021
    gpu_id = 0
    num_workers = 0  # 0 for debugging.
    mode = 'train'

    ## dataset
    dataset = 'AROIDuke_ONE_block_paste'  # i.e. abdominal MRI - 'CHAOST2'; cardiac MRI - CMR
    eval_fold = 0   # (0-4) for 5-fold cross-validation
    test_label = [1, 4]  # for evaluation
    img_size = 256
    video_path = None
    video_h = 295
    video_w = 615
    measure = 'mm'

    ## training
    n_steps = 1000
    batch_size = 1
    n_shot = 1
    n_query = 1
    lr_step_gamma = 0.95
    bg_wt = 0.1
    t_loss_scaler = 0.0
    ignore_label = 255
    print_interval = 100  # raw=100
    save_snapshot_every = 3000
    max_iters_per_load = 1000  # epoch size, interval for reloading the dataset

    # Network
    model_name = "famnet"
    reload_model_path = None

    optim_type = 'sgd'
    optim = {
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0.0005,  # 0.0005
    }

    exp_str = '_'.join(
        [mode]
        + [dataset, ]
        + [f'cv{eval_fold}'])

    path = {
        'log_dir': './runs',
        'data_root': './data/OCT/',
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
