"""
Dataset for Training and Test
Extended from ADNet code by Hansen et al.
"""
import torch
import cv2
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as deftfx
from torchvision.transforms import InterpolationMode
import glob
import os
import random
import numpy as np
import PIL.Image as Image
from . import image_transforms as myit
from .dataset_specifics import *


class TestDataset(Dataset):

    def __init__(self, args):

        self.n_shot = args['n_shot']
        self.img_size = args['img_size']

        # reading the paths
        self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'test/images/*'))
        self.label_dirs = glob.glob(os.path.join(args['data_dir'], 'test/masks/*'))

        self.sup_img_dirs = glob.glob(os.path.join(args['data_dir'], 'sup/images/*'))
        self.sup_lbl_dirs = glob.glob(os.path.join(args['data_dir'], 'sup/masks/*'))

        self.transform = deftfx.Compose([deftfx.Resize(size=(self.img_size, self.img_size), interpolation=InterpolationMode.NEAREST),
                                         deftfx.ToTensor(),
                                         ])

        # check the order
        for image_dir, label_dir in zip(self.image_dirs, self.label_dirs):
            image_filename = image_dir.split("/")[-1].split(".")[0]
            label_filename = label_dir.split("/")[-1].split(".")[0]
            assert image_filename == label_filename

        for image_dir, label_dir in zip(self.sup_img_dirs, self.sup_lbl_dirs):
            image_filename = image_dir.split("/")[-1].split(".")[0]
            label_filename = label_dir.split("/")[-1].split(".")[0]
            assert image_filename == label_filename

        self.label = None
        self.n_classes = None

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):

        img_path = self.image_dirs[idx]
        img = self.transform(Image.open(img_path).convert('L')).squeeze().numpy()
        img = (img - img.mean()) / img.std()
        img = np.stack(3 * [img], axis=0)

        lbl_path = self.label_dirs[idx]
        lbl = self.transform(Image.open(lbl_path).convert('L')).squeeze().numpy() * 255
        lbl = lbl.astype(np.uint8)
        lbl = 1 * (lbl == self.label)

        sample = {'id': img_path}

        # Evaluation protocol.
        sample['image'] = torch.from_numpy(img)
        sample['label'] = torch.from_numpy(lbl)

        return sample

    def get_support_index(self, n_shot, C):
        """
        Selecting intervals according to Ouyang et al.
        """
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None):
        if label is None:
            raise ValueError('Need to specify label class!')
        
        imgs = [Image.open(im).convert('L') for im in self.sup_img_dirs][:self.n_shot]
        imgs = np.stack([self.transform(im).squeeze().numpy() for im in imgs], axis=0)
        imgs = (imgs - imgs.mean()) / imgs.std()
        imgs = np.stack(3 * [imgs], axis=1)

        lbls = [Image.open(lbl).convert('L') for lbl in self.sup_lbl_dirs][:self.n_shot]
        lbls = np.stack([self.transform(lbl).squeeze().numpy() * 255 for lbl in lbls], axis=0)
        lbls = lbls.astype(np.uint8)
        lbls = 1 * (lbls == label)

        sample = {}
        sample['image'] = torch.from_numpy(imgs)
        sample['label'] = torch.from_numpy(lbls)

        return sample


class TrainDataset(Dataset):

    def __init__(self, args):
        self.n_shot = args['n_shot']
        self.max_iter = args['max_iter']
        self.img_size = args['img_size']

        self.transform = deftfx.Compose([deftfx.Resize(size=(self.img_size, self.img_size), interpolation=InterpolationMode.NEAREST),
                                         deftfx.ToTensor(),
                                         ])

        self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'trn/images/*'))
        self.label_dirs = glob.glob(os.path.join(args['data_dir'], 'trn/masks/*'))

        # read images
        self.images = {}
        self.labels = {}
        for image_dir, label_dir in zip(self.image_dirs, self.label_dirs):
            image_filename = image_dir.split("/")[-1].split(".")[0]
            label_filename = label_dir.split("/")[-1].split(".")[0]
            assert image_filename == label_filename
            self.images[image_dir] = Image.open(image_dir).convert('L')
            self.labels[label_dir] = Image.open(label_dir).convert('L')
        
        print(f"Succeed in loading {len(self.image_dirs)} images!")

    def __len__(self):
        return self.max_iter

    def gamma_tansform(self, img):
        gamma_range = (0.5, 1.5)
        gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = (img.max() - cmin + 1e-5)

        img = img - cmin + 1e-5
        img = irange * np.power(img * 1.0 / irange, gamma)
        img = img + cmin

        return img

    def geom_transform(self, img, mask):

        affine = {'rotate': 5, 'shift': (5, 5), 'shear': 5, 'scale': (0.9, 1.2)}
        alpha = 10
        sigma = 5
        order = 3

        tfx = []
        tfx.append(myit.RandomAffine(affine.get('rotate'),
                                     affine.get('shift'),
                                     affine.get('shear'),
                                     affine.get('scale'),
                                     affine.get('scale_iso', True),
                                     order=order))
        tfx.append(myit.ElasticTransform(alpha, sigma))
        transform = deftfx.Compose(tfx)

        if len(img.shape) == 4:
            for shot in range(img.shape[0]):
                cat = np.concatenate((img[shot], mask[shot][None])).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[shot] = cat[:3, :, :]
                mask[shot] = np.rint(cat[3:, :, :].squeeze())

        else:
            cat = np.concatenate((img, mask[None])).transpose(1, 2, 0)
            cat = transform(cat).transpose(2, 0, 1)
            img = cat[:3, :, :]
            mask = np.rint(cat[3:, :, :].squeeze())

        return img, mask

    def sample_episode(self, idx):
        query_idx = random.choice(range(len(self.image_dirs)))

        support_idxs = []
        while True:  # keep sampling support set if query == support
            support_idx = random.choice(range(len(self.image_dirs)))
            if support_idx != query_idx: support_idxs.append(support_idx)
            if len(support_idxs) == self.n_shot: break

        return support_idxs, query_idx

    def __getitem__(self, idx):

        # sample image idx
        support_idxs, query_idx = self.sample_episode(idx)

        # get support images from dictionary
        support_keys = [self.image_dirs[idx] for idx in support_idxs]
        sup_img = np.stack([self.transform(self.images[key]).squeeze().numpy() for key in support_keys], axis=0)
        support_keys = [self.label_dirs[idx] for idx in support_idxs]
        sup_lbl = np.stack([self.transform(self.labels[key]).squeeze().numpy() for key in support_keys], axis=0)

        # get query image from dictionary
        qry_img = self.transform(self.images[self.image_dirs[query_idx]]).squeeze().numpy()
        qry_lbl = self.transform(self.labels[self.label_dirs[query_idx]]).squeeze().numpy()

        # get 3 channels
        sup_img = np.stack((sup_img, sup_img, sup_img), axis=1)
        qry_img = np.stack((qry_img, qry_img, qry_img), axis=0)

        # normalize
        sup_img = (sup_img - sup_img.mean()) / sup_img.std()
        qry_img = (qry_img - qry_img.mean()) / qry_img.std()

        # sample class(es) (gt/supervoxel)
        unique = list(set(np.unique(sup_lbl)) & set(np.unique(qry_lbl)))
        unique.remove(0)

        cls_idx = random.choice(unique)

        sup_lbl = (sup_lbl == cls_idx)
        qry_lbl = (qry_lbl == cls_idx)

        # gamma transform
        if np.random.random(1) > 0.5:
            qry_img = self.gamma_tansform(qry_img)
        else:
            sup_img = self.gamma_tansform(sup_img)

        # geom transform
        if np.random.random(1) > 0.5:
            qry_img, qry_lbl = self.geom_transform(qry_img, qry_lbl)
        else:
            sup_img, sup_lbl = self.geom_transform(sup_img, sup_lbl)

        sample = {'support_images': sup_img,
                  'support_fg_labels': sup_lbl,
                  'query_images': qry_img,
                  'query_labels': qry_lbl,
                  'selected_class': cls_idx,
                  'support_id': self.image_dirs[support_idxs[0]],
                  'query_id': self.image_dirs[query_idx]}

        return sample

