"""
Utils for Dataset
Extended from ADNet code by Hansen et al.
"""
import random
import torch
import numpy as np
import logging

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distance_estimate(pred, tool_id, tissue_id, config):
    """
    Measure the distance from tool to the tissue surface
    pred: segmentation prediction, np.array [H * W]
    tool_id: id representing tool
    tissue_id: id representing tissue
    """

    tool_mask = (pred == tool_id)
    tissue_mask = (pred == tissue_id)
    tool_region = np.stack(np.nonzero(tool_mask), axis=-1)
    tissue_region = np.stack(np.nonzero(tissue_mask), axis=-1)

    if len(tool_region) == 0 or len(tissue_region) == 0:
        return None, None, None, None
    else:
        tissue_column = tissue_mask.any(axis=0)
        shadow_column = np.where(~tissue_column)[0]

        if len(shadow_column) == 0:
            # lowest tool point
            lowest_tool_idx = np.argmax(tool_region[:, 0])  # max y
            tool_point = tool_region[lowest_tool_idx]  # [y, x]
            tool_point = [int(tool_point[1]), int(tool_point[0])]

            # highest tissue point
            tool_column = np.where(tool_mask.any(axis=0))[0] # x idx with tools
            tissue_under_tool = tissue_mask[:, tool_column] 
            tissue_ys, tissue_xs_idx = np.nonzero(tissue_under_tool)
            highest_idx = tissue_xs_idx[np.argmin(tissue_ys)]
            tissue_point = [int(tool_column[highest_idx]), int(min(tissue_ys))]

            side = 'left' if tissue_point[0] < tool_point[0] else 'right'
        
            distance = None
            if tool_point and tissue_point: 
                unit = config['measure']
                distance = abs(tissue_point[1] - tool_point[1]) if unit=='px' \
                    else round(abs(tissue_point[1] - tool_point[1]) * 6 / config['video_w'], 3)

            return tool_point, tissue_point, distance, side


        tool_mask_in_shadow = tool_mask[:, shadow_column]
        tool_ys, tool_xs_idx = np.nonzero(tool_mask_in_shadow)

        tool_point, tissue_point, side = None, None, None

        if len(tool_ys) > 0:
            # lowest tool point
            tool_xs = shadow_column[tool_xs_idx]
            lowest_idx = np.argmax(tool_ys)
            tool_point = [int(tool_xs[lowest_idx]), int(tool_ys[lowest_idx])]

            # leftmost and rightmost tool point
            tool_coords = np.stack([tool_xs, tool_ys], axis=-1)
            left_idx = np.argmin(tool_coords[:, 0])
            right_idx = np.argmax(tool_coords[:, 0])

            left_point = tool_coords[left_idx].tolist()
            right_point = tool_coords[right_idx].tolist()
            side = 'left' if left_point[1] > right_point[1] else 'right'

            # find tissue points 
            left_tissue = tissue_region[tissue_region[:, 1] < left_point[0]]
            right_tissue = tissue_region[tissue_region[:, 1] > right_point[0]]
            if (side == 'left') and (len(left_tissue) > 0):
                rightmost_left_cols = np.sort(np.unique(left_tissue[:, 1]))[-10:]
                left_tissue = left_tissue[np.isin(left_tissue[:, 1], rightmost_left_cols)]
                high_idx = np.argmin(left_tissue[:, 0])
                tissue_point = left_tissue[high_idx]
                tissue_point = [tissue_point[1], tissue_point[0]]

            elif (side == 'right') and (len(right_tissue) > 0):
                leftmost_right_cols = np.sort(np.unique(right_tissue[:, 1]))[:10]
                right_tissue = right_tissue[np.isin(right_tissue[:, 1], leftmost_right_cols)]
                high_idx = np.argmin(right_tissue[:, 0])
                tissue_point = right_tissue[high_idx]
                tissue_point = [tissue_point[1], tissue_point[0]]

            else:
                if len(left_tissue) > 0:
                    rightmost_left_cols = np.sort(np.unique(left_tissue[:, 1]))[-10:]
                    left_tissue = left_tissue[np.isin(left_tissue[:, 1], rightmost_left_cols)]
                    high_idx = np.argmin(left_tissue[:, 0])
                    tissue_point = left_tissue[high_idx]
                    tissue_point = [tissue_point[1], tissue_point[0]]
                elif len(right_tissue) > 0:
                    leftmost_right_cols = np.sort(np.unique(right_tissue[:, 1]))[:10]
                    right_tissue = right_tissue[np.isin(right_tissue[:, 1], leftmost_right_cols)]
                    high_idx = np.argmin(right_tissue[:, 0])
                    tissue_point = right_tissue[high_idx]
                    tissue_point = [tissue_point[1], tissue_point[0]]
        
        distance = None
        if tool_point and tissue_point: 
            unit = config['measure']
            distance = abs(tissue_point[1] - tool_point[1]) if unit=='px' \
                else round(abs(tissue_point[1] - tool_point[1]) * 6 / config['video_w'], 3)

        return tool_point, tissue_point, distance, side


def color_map(img, label_mask):
    label_colors = {
        1: (220, 50, 47, 180),
        2: (60, 201, 230, 180),
        3: (239, 247, 5, 180),
    }
    colored_mask = np.zeros((*label_mask.shape, 3), dtype=np.float32)
    for label_val, color in label_colors.items():
        colored_mask[label_mask == label_val] = color[:3]
    alpha = 0.5
    mask_color = ((1 - alpha) * img + alpha * colored_mask).astype(np.uint8)

    return mask_color
    

def draw_distance(mask_color, tool_point, tissue_point, dist, side, config):
    if tool_point and tissue_point: 
        # Draw lines
        x2 = int(min(tool_point[0], tissue_point[0])) - 20 if side == 'left' \
            else int(max(tool_point[0], tissue_point[0])) + 20
        cv2.line(mask_color, tool_point, (x2, tool_point[1]), (255, 255, 255), 1)
        cv2.line(mask_color, tissue_point, (x2, tissue_point[1]), (255, 255, 255), 1)
        # Draw arrows
        arrow_x = int(min(tool_point[0], tissue_point[0])) - 16 if side == 'left' \
            else int(max(tool_point[0], tissue_point[0])) + 16
        top_y, bottom_y = min(tool_point[1], tissue_point[1]), max(tool_point[1], tissue_point[1])
        cv2.arrowedLine(mask_color, (arrow_x, bottom_y), (arrow_x, top_y), (255, 255, 255), 1, tipLength=0.1)
        cv2.arrowedLine(mask_color, (arrow_x, top_y), (arrow_x, bottom_y), (255, 255, 255), 1, tipLength=0.1)
        # Draw circles at points
        cv2.circle(mask_color, tool_point, 4, (0, 0, 255), -1)
        cv2.circle(mask_color, tissue_point, 4, (255, 0, 0), -1)
        # Draw text label
        unit = config['measure']
        text_pos = (arrow_x + 10, (tool_point[1] + tissue_point[1]) // 2)
        cv2.putText(mask_color, f'{dist} {unit}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return mask_color


def draw_distance_plot(mask_color, frame_dist, total_frames, config):
    import cv2
    import numpy as np

    h, w = mask_color.shape[:2]
    margin = 20
    plot_w = 120
    plot_h = 60
    # x0 = w - plot_w - margin   # origin x (top-left of plot area)
    # y0 = margin + plot_h       # origin y (bottom-left of plot area)
    # x0, y0 = margin, h - margin # bottom-right
    x0, y0 = margin, margin + plot_h # top-left

    unit = config['measure']

    # Fill None with previous value
    vals = []
    last = 0.0
    for v in frame_dist:
        if v is None:
            vals.append(last)
        else:
            last = float(v)
            vals.append(last)
    if len(vals) == 0:
        return mask_color

    # --- Ensure Y axis scale is >= 1.2 mm so 1 mm tick is below arrow ---
    max_val = max(vals) if max(vals) > 0 else 1.0
    if unit == 'mm':
        min_axis_val = 1.2  # give extra headroom above 1 mm
        max_val = max(max_val, min_axis_val)

    tip_px = 5
    tl_x = tip_px / max(1, plot_w)
    tl_y = tip_px / max(1, plot_h)
    # Draw Y axis
    cv2.arrowedLine(mask_color, (x0, y0), (x0, y0 - plot_h),
                    (255, 255, 255), 1, cv2.LINE_AA, 0, tipLength=tl_y)
    cv2.putText(mask_color, f"Distance ({unit})", (x0 - 5, y0 - plot_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw X axis
    cv2.arrowedLine(mask_color, (x0, y0), (x0 + plot_w, y0),
                    (255, 255, 255), 1, cv2.LINE_AA, 0, tipLength=tl_x)
    cv2.putText(mask_color, "Time (t)", (x0 + plot_w + 5, y0 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Origin label
    cv2.putText(mask_color, "0", (x0 - 12, y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # --- 1 mm tick on Y axis ---
    if unit == 'mm':
        y_1mm = y0 - int((1.0 / max_val) * plot_h)
        cv2.line(mask_color, (x0 - 2, y_1mm), (x0 + 2, y_1mm), (255, 255, 255), 1)
        cv2.putText(mask_color, "1", (x0 - 15, y_1mm + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # --- total_frames tick on X axis ---
    axis_frames = int(total_frames * 1.1)  # +10% space
    px_end_tick = x0 + int((total_frames / axis_frames) * plot_w)
    cv2.line(mask_color, (px_end_tick, y0 - 2), (px_end_tick, y0 + 2), (255, 255, 255), 1)
    cv2.putText(mask_color, str(total_frames),
                (px_end_tick - 10, y0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Scale & plot distances
    pts = []
    for i, d in enumerate(vals):
        px = x0 + int((i / axis_frames) * plot_w) if axis_frames > 1 else x0
        py = y0 - int((d / max_val) * plot_h)
        pts.append([px, py])

    if len(pts) >= 2:
        cv2.polylines(mask_color, [np.array(pts, dtype=np.int32)],
                      False, (0, 255, 0), 1)

    # Mark last point
    last_pt = tuple(pts[-1])
    cv2.circle(mask_color, last_pt, 3, (0, 0, 255), -1)

    return mask_color

def draw_video(img, pred, pred_tool, pred_tissue, pred_dist, frame_dist, pred_side, total_frames, config):
    pred_mask_color = color_map(img, pred)
    pred_mask_color = draw_distance(pred_mask_color, pred_tool, pred_tissue, pred_dist, pred_side, config)
    pred_mask_color = draw_distance_plot(pred_mask_color, frame_dist, total_frames, config)

    return pred_mask_color


def visualize(img, pred, gt, labels, pred_est, gt_est, config, save_path):
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(labels))]
    custom_cmap = ListedColormap(colors)

    pred_mask_color = color_map(img, pred)
    gt_mask_color = color_map(img, gt)

    pred_tool, pred_tissue, pred_distance, pred_side = pred_est
    gt_tool, gt_tissue, gt_distance, gt_side = gt_est
    pred_mask_color = draw_distance(pred_mask_color, pred_tool, pred_tissue, pred_distance, pred_side, config)
    gt_mask_color = draw_distance(gt_mask_color, gt_tool, gt_tissue, gt_distance, gt_side, config)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(img)
    axs[0, 0].set_title("Query Image")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(pred, cmap=custom_cmap, vmin=0, vmax=(len(labels)-1), interpolation='none')
    axs[0, 1].set_title("Prediction")
    axs[0, 1].axis('off')
    axs[1, 0].imshow(pred_mask_color)
    axs[1, 0].set_title("Distance Prediction")
    axs[1, 0].axis('off')
    axs[1, 1].imshow(gt_mask_color)
    axs[1, 1].set_title("Distance Ground Truth")
    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class Scores():

    def __init__(self):
        self.patient_dice = []
        self.patient_iou = []
        self.distance_error = []

    def record(self, preds, label):
        assert len(torch.unique(preds)) < 3

        tp = torch.sum((label == 1) * (preds == 1))
        tn = torch.sum((label == 0) * (preds == 0))
        fp = torch.sum((label == 0) * (preds == 1))
        fn = torch.sum((label == 1) * (preds == 0))

        if (tp + fp + fn) == 0:
            self.patient_dice.append(torch.tensor(1.0))
            self.patient_iou.append(torch.tensor(1.0))
        else:
            self.patient_dice.append(2 * tp / (2 * tp + fp + fn + 1e-5))
            self.patient_iou.append(tp / (tp + fp + fn + 1e-5))

    def distance(self, error):
        self.distance_error.append(error)


def set_logger(path):
    logger = logging.getLogger()
    logger.handlers = []
    formatter = logging.Formatter('[%(levelname)] - %(name)s - %(message)s')
    logger.setLevel("INFO")

    # log to .txt
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
