#!/usr/bin/env python
"""
For evaluation
Extended from ADNet code by Hansen et al.
"""
import shutil
import torch.backends.cudnn as cudnn
from models.CDFSMIS import FewShotSeg
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import *
from utils import *
from config import ex
import PIL.Image as Image

import os
import cv2
from torchvision import transforms


def fill_in_box_horizontal(frame):
    horizontal_ioct = frame[58:353, 653:1268, :]
    horizontal_ioct[8:10, 8:60, :] = horizontal_ioct[6:8, 8:60, :] # up
    horizontal_ioct[58:60, 8:60, :] = horizontal_ioct[56:58, 8:60, :] # down
    horizontal_ioct[8:60, 8:10, :] = horizontal_ioct[8:60, 6:8, :] # left
    horizontal_ioct[8:60, 58:60, :] = horizontal_ioct[8:60, 56:58, :] # right
    horizontal_ioct[94:199, 605:609, :] = horizontal_ioct[94:199, 609:613, :] # right
    horizontal_ioct[32:36, 8:60, :] = horizontal_ioct[36:40, 8:60, :] # blue
    horizontal_ioct[29:38, 51:56, :] = horizontal_ioct[29:38, 46:51, :]

    horizontal_ioct[151:156, 583:600, :] = horizontal_ioct[151:156, 566:583, :] # up
    horizontal_ioct[135:144, 591:593, :] = horizontal_ioct[135:144, 593:595, :] # up
    horizontal_ioct[136:139, 589:591, :] = horizontal_ioct[136:139, 587:589, :] # up

    return horizontal_ioct


def fill_in_box_vertical(frame):
    vertical_ioct = frame[365:660, 653:1268, :]
    vertical_ioct[8:10, 8:60, :] = vertical_ioct[6:8, 8:60, :] # up
    vertical_ioct[58:60, 8:60, :] = vertical_ioct[56:58, 8:60, :] # down
    vertical_ioct[8:60, 8:10, :] = vertical_ioct[8:60, 6:8, :] # left
    vertical_ioct[8:60, 58:60, :] = vertical_ioct[8:60, 56:58, :] # right
    vertical_ioct[94:199, 605:609, :] = vertical_ioct[94:199, 609:613, :] # right
    vertical_ioct[8:60, 32:35, :] = vertical_ioct[8:60, 35:38, :] # purple
    vertical_ioct[11:17, 31:37, :] = vertical_ioct[5:11, 31:37, :] # purple

    vertical_ioct[151:156, 583:600, :] = vertical_ioct[151:156, 566:583, :] # up
    vertical_ioct[135:144, 591:593, :] = vertical_ioct[135:144, 593:595, :] # up
    vertical_ioct[136:139, 589:591, :] = vertical_ioct[136:139, 587:589, :] # up

    return vertical_ioct


@ex.automain
def main(_run, _config, _log):
    # ------------ log dir ------------
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

    # ------------ device ------------
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

    # ------------ load model ------------
    _log.info(f'Create model...')
    model = FewShotSeg()
    model.cuda()
    model.load_state_dict(torch.load(_config['reload_model_path'], map_location='cpu', weights_only=True), strict=False)

    # ------------ load image and video ------------
    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path']['data_root'] + _config['dataset'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'img_size': _config['img_size'],
    }
    test_dataset = TestDataset(data_config)

    h, w = _config['video_h'], _config['video_w']
    preprocess = test_dataset.transform
    postprocess = transforms.Compose([
        transforms.Resize((h, w), antialias=True)
        ])

    video_dir = _config['video_path']
    video = cv2.VideoCapture(video_dir)
    frame_id = 0
    frame_dist = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # ------------ video writer ------------
    output_dir = f'./{_run.observers[0].dir}/video_preds'
    os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_dir + "/output.mp4"), fourcc, 30.0, (w, h))

    # Get unique labels (classes).
    labels = get_label_names(_config['dataset'])

    tool_id = list(labels.keys())[list(labels.values()).index('tool')]
    tissue_id = list(labels.keys())[list(labels.values()).index('tissue')]

    _log.info(f'Starting validation...')

    # Test.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        
        # frame inference 
        start.record()
        query_pred = torch.zeros((1, h, w))
        for label_val, label_name in labels.items():

            # Skip BG class.
            if label_name == 'BG':
                continue
            elif np.intersect1d([label_val], _config['test_label']).size == 0:
                continue

            # Get support sample for each class
            test_dataset.label = label_val
            test_dataset.n_classes = len(labels)

            support_sample = test_dataset.getSupport(label=label_val)
            support_image = [shot.float().cuda().unsqueeze(0) for shot in support_sample['image']]
            support_fg_mask = [shot.float().cuda().unsqueeze(0) for shot in support_sample['label']]

            with torch.no_grad():
                model.eval()
        
                # cut iOCT
                if _config['dataset'] == 'horizontal':
                    ioct = fill_in_box_horizontal(frame)
                elif _config['dataset'] == 'vertical':
                    ioct = fill_in_box_vertical(frame)
                ioct = cv2.cvtColor(ioct, cv2.COLOR_BGR2GRAY)

                # pre-processing
                query_image = Image.fromarray(ioct)
                query_image = preprocess(query_image)
                query_image = (query_image - query_image.mean()) / query_image.std()
                query_image = torch.stack(3 * [query_image], axis=1).float().cuda()
                query_id = frame_id

                # prediction
                pred = model.predict_mask_nshot(support_image, support_fg_mask, query_image, None)

                # post-processing
                pred = postprocess(pred).cpu()
                query_pred[pred != 0] = label_val

        end.record()
        torch.cuda.synchronize()

        tool_id = list(labels.keys())[list(labels.values()).index('tool')]
        tissue_id = list(labels.keys())[list(labels.values()).index('tissue')]

        # multi-class prediction
        query_pred = query_pred.squeeze().cpu().numpy().astype(np.uint8)

        # raw image
        query_image = np.stack([ioct]*3, axis=-1).astype(np.uint8)

        # distance estimate
        p_tool, p_tissue, dist, side = distance_estimate(query_pred, tool_id, tissue_id, _config)

        # visualization
        # frame_dist = distance_record(frame_dist, dist)
        frame_dist.append(dist)
        pred_mask = draw_video(query_image, query_pred,
                               p_tool, p_tissue, dist, frame_dist, side, 
                               total_frames, _config)

        # Save video
        file_name = os.path.join(f'{_run.observers[0].dir}/interm_preds',
                                f'{query_id}_pred.png')
        cv2.imwrite(file_name, cv2.cvtColor(pred_mask, cv2.COLOR_RGB2BGR))
        out.write(cv2.cvtColor(pred_mask, cv2.COLOR_RGB2BGR))
        _log.info(f'{query_id} has been saved with {round(start.elapsed_time(end) / 1000, 3)} s. ')

        frame_id += 1

    out.release()
    _log.info(f'End of validation.')
    return 1
