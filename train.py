#!/usr/bin/env python
"""
For Evaluation
Extended from ADNet code by Hansen et al.
"""
import shutil
import os

import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config import ex
from dataloaders.datasets import TrainDataset as TrainDataset
from models.CDFSMIS import FewShotSeg
from util.utils import *
from util.losses import CombinedLoss

def pixel_accuracy(pred, label):
    pred_flatten = pred.flatten()
    label_flatten = label.flatten()
    accuracy = accuracy_score(label_flatten, pred_flatten)
    return accuracy


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        # Set up source folder
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproduciablity.
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model = FewShotSeg()
    model = model.cuda()
    model.train()

    _log.info(f'Set optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    lr_milestones = [(ii + 1) * _config['max_iters_per_load'] for ii in
                     range(_config['n_steps'] // _config['max_iters_per_load'] - 1)]
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=_config['lr_step_gamma'])
    criterion = CombinedLoss()

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path']['data_root'] + _config['dataset'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'max_iter': _config['max_iters_per_load'],
        'img_size': _config['img_size'],
    }
    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(train_dataset,
                              batch_size=_config['batch_size'],
                              shuffle=True,
                              num_workers=_config['num_workers'],
                              pin_memory=True,
                              drop_last=True)

    n_sub_epochs = _config['n_steps'] // _config['max_iters_per_load']  # number of times for reloading
    log_loss = {'total_loss': 0, 'query_loss': 0, 'align_loss': 0, 'thresh_loss': 0}

    loss_values = []
    i_iter = 0
    _log.info(f'Start training...')
    for sub_epoch in range(n_sub_epochs):
        _log.info(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')
        scores = Scores()
        for _, sample in enumerate(train_loader):

            # print(sample['support_images'].shape, sample['support_fg_labels'].shape, sample['query_images'].shape, sample['query_labels'].shape)
            # torch.Size([16, 4, 3, 256, 256]) torch.Size([16, 4, 256, 256]) torch.Size([16, 3, 256, 256]) torch.Size([16, 256, 256])
            # BS * n_shot * 3 * 256 * 256 => n_shot * (BS * C * H * W)
            assert sample['support_images'].shape[1] == sample['support_fg_labels'].shape[1] == 1
            support_images = sample['support_images'].squeeze(1).float().cuda()
            support_fg_mask = sample['support_fg_labels'].squeeze(1).float().cuda()
            query_images = sample['query_images'].float().cuda()
            query_labels = sample['query_labels'].long().cuda()

            # Compute outputs and losses.
            optimizer.zero_grad()
            query_pred, proto_loss = model(support_images, support_fg_mask, query_images, query_labels, opt=optimizer, train=True)
            query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                         1 - torch.finfo(torch.float32).eps)), query_labels)

            # compute metrics
            query_pred = query_pred.argmax(dim=1) # torch.Size([1, 2, 256, 256])
            for i in range(len(query_pred)):
                scores.record(query_pred[i], query_labels[i])
            dice = torch.tensor(scores.patient_dice[:-len(query_pred)]).mean().item()
            miou = torch.tensor(scores.patient_iou[:-len(query_pred)]).mean().item()

            loss = query_loss + proto_loss

            # Compute gradient and do SGD step.
            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()

            loss_values.append(query_loss)

            _run.log_scalar('total_loss', loss.item())
            _run.log_scalar('query_loss', query_loss)

            log_loss['total_loss'] += loss.item()
            log_loss['query_loss'] += query_loss

            # Print loss and take snapshots.
            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                query_loss = log_loss['query_loss'] / _config['print_interval']

                log_loss['total_loss'] = 0
                log_loss['query_loss'] = 0

                _log.info('step: %d, total_loss: %.4f, query_loss: %.4f, dice: %.4f, miou: %.4f' % (
                    i_iter + 1, total_loss, query_loss, dice, miou))
                # f' align_loss: {align_loss}')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save(model.state_dict(),
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

            i_iter += 1

    loss_values = np.array(loss_values)
    # loss_values = loss_values.detach().cpu().numpy()
    np.savetxt(os.path.join(f'{_run.observers[0].dir}', 'loss_values.txt'), loss_values)

    _log.info('End of training.')
    return 1
