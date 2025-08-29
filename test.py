#!/usr/bin/env python
"""
For evaluation
Extended from ADNet code by Hansen et al.
"""
import os
import shutil
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models.CDFSMIS import FewShotSeg
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import *
from util.utils import *
from config import ex
import cv2

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
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
    model.cuda()
    model.load_state_dict(torch.load(_config['reload_model_path'], map_location='cpu', weights_only=True), strict=False)

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path']['data_root'] + _config['dataset'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'img_size': _config['img_size'],
    }
    test_dataset = TestDataset(data_config)
    test_loader = DataLoader(test_dataset,
                             batch_size=_config['batch_size'],
                             shuffle=False,
                             num_workers=_config['num_workers'],
                             pin_memory=True,
                             drop_last=True)

    # Get unique labels (classes).
    labels = get_label_names(_config['dataset'])
    n_classes = len(labels)

    # Loop over classes.
    class_dice = {}
    class_iou = {}
    image_dict ={}
    pred_dict = {}
    gt_dict = {}

    _log.info(f'Starting validation...')
    for label_val, label_name in labels.items():

        # Skip BG class.
        if label_name == 'BG':
            continue
        elif np.intersect1d([label_val], _config['test_label']).size == 0:
            continue

        _log.info(f'Test Class: {label_name}')

        # Get support sample + mask for current class.
        test_dataset.label = label_val
        test_dataset.n_classes = n_classes

        support_sample = test_dataset.getSupport(label=label_val)
        support_image = [shot.float().cuda().unsqueeze(0) for shot in support_sample['image']]
        support_fg_mask = [shot.float().cuda().unsqueeze(0) for shot in support_sample['label']]

        # Test.
        with torch.no_grad():
            model.eval()

            # Loop through query volumes.
            scores = Scores()
            for i, sample in enumerate(test_loader):

                query_image = sample['image'].float().cuda()
                query_label = sample['label'].long()
                query_id = sample['id'][0].split("/")[-1].split(".")[0]

                # Compute output.
                query_pred = model.predict_mask_nshot(support_image, support_fg_mask, query_image, None)
                query_pred = query_pred.cpu()

                # Save results in a dict
                if pred_dict.get(query_id) is not None:
                    pred_dict.update({query_id: torch.cat([pred_dict[query_id], query_pred], dim=0)})
                    gt_dict.update({query_id: torch.cat([gt_dict[query_id], query_label], dim=0)})
                else:
                    pred_dict.update({query_id: query_pred})
                    gt_dict.update({query_id: query_label})
                    image_dict.update({query_id: sample['image'].squeeze(1)})

                # Record scores.
                scores.record(query_pred, query_label.squeeze(0))

                # Log.
                _log.info(f'Tested query image: {query_id}. Dice score: {round(scores.patient_dice[-1].item(), 4)}, IoU score: {round(scores.patient_iou[-1].item(), 4)}')

                # break

            # Log class-wise results
            class_dice[label_name] = round(torch.tensor(scores.patient_dice).mean().item(), 4)
            class_iou[label_name] = round(torch.tensor(scores.patient_iou).mean().item(), 4)
            _log.info(f'Test Class: {label_name}')
            _log.info(f'Mean class IoU: {class_iou[label_name]}')
            _log.info(f'Mean class Dice: {class_dice[label_name]}')


    def dict_Avg(Dict):
        L = len(Dict)  # 取字典中键值对的个数
        S = sum(Dict.values())  # 取字典中键对应值的总和
        A = S / L
        return A

    _log.info(f'Starting distance estimation...')

    tool_id = list(labels.keys())[list(labels.values()).index('tool')]
    tissue_id = list(labels.keys())[list(labels.values()).index('tissue')]

    scores = Scores()
    for name, pred in pred_dict.items():
        # multi-class prediction
        pred_mask = torch.argmax(pred, dim=0) + 1
        pred_mask *= pred.sum(dim=0).bool()
        pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)

        # multi-class ground truth
        gt = gt_dict[name]
        gt_mask = torch.argmax(gt, dim=0) + 1
        gt_mask *= gt.sum(dim=0).bool()
        gt_mask_np = gt_mask.cpu().numpy().astype(np.uint8)

        # raw image
        img = image_dict[name]
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
        img_np = (img_np * 255).astype(np.uint8)
        img_np = np.ascontiguousarray(img_np, dtype=np.uint8)

        # distance estimate
        pred_est = distance_estimate(pred_mask_np, tool_id, tissue_id, _config)
        gt_est = distance_estimate(gt_mask_np, tool_id, tissue_id, _config)

        # distance error
        if pred_est[0] and pred_est[1]:
            # pred_dis = (pred_est[1][1] - pred_est[0][1])
            # gt_dis = (gt_est[1][1] - gt_est[0][1])
            # error = abs((pred_dis - gt_dis) / gt_dis)
            error = abs(pred_est[2] - gt_est[2])
            scores.distance(error)
        else:
            error = 1.0
            scores.distance(error)

        # Log.
        _log.info(f'Tested query image: {name}. Distance error: {round(error, 4)}.')

        # # visualization
        # save_path = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'{name}.png')
        # visualize(img_np, pred_mask_np, gt_mask_np, labels, pred_est, gt_est, _config, save_path)

        # # overlay
        # label_colors = {
        #     0: (0, 0, 0, 0),
        #     1: (220, 50, 47, 180),
        #     2: (60, 201, 230, 180),
        #     3: (239, 247, 5, 180),
        # }
        # color_mask = np.zeros((pred_mask_np.shape[0], pred_mask_np.shape[1], 3), dtype=np.uint8)
        # for label, color in label_colors.items():
        #     color_mask[pred_mask_np == label] = color[:3][::-1]
        # alpha = 0.5
        # pred_mask = ((1 - alpha) * img_np + alpha * color_mask).astype(np.uint8)
        # dataset_name = _config['dataset']
        # save_path = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'Our_{dataset_name}_{name}.png')
        # cv2.imwrite(save_path, pred_mask)

    _log.info(f'Final results...')
    _log.info(f'IoU: {class_iou}. Whole mean IoU: {dict_Avg(class_iou)}')
    _log.info(f'Dice: {class_dice}. Whole mean Dice: {dict_Avg(class_dice)}')

    mDisError = round(torch.tensor(scores.distance_error).mean().item(), 4)
    _log.info(f'Mean distance error: {mDisError}')

    _log.info(f'End of validation.')
    return 1
