import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder import Res50Encoder

from .mamba_block import CrossBlock as Cross_Block
from .cross_mamba_simple import Mamba as Cross_Mamba
from .head import SegmentationHead
from . import initialization as init
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from functools import partial


class AttentionMacthcing(nn.Module):
    def __init__(self, feature_dim=512, seq_len=5000):
        super(AttentionMacthcing, self).__init__()
        self.fc_spt = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        self.fc_qry = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(seq_len * 2, seq_len // 5),

            nn.ReLU(),
            nn.Linear(seq_len // 5, 2 * seq_len),
        )
        self.sigmoid = nn.Sigmoid()


    def correlation_matrix(self, spt_fg_fts, qry_fg_fts):
        """
        Calculates the correlation matrix between the spatial foreground features and query foreground features.

        Args:
            spt_fg_fts (torch.Tensor): The spatial foreground features. 
            qry_fg_fts (torch.Tensor): The query foreground features. 

        Returns:
            torch.Tensor: The cosine similarity matrix. Shape: [1, 1, N].
        """

        spt_fg_fts = F.normalize(spt_fg_fts, p=2, dim=1)  # shape [1, 512, 900]
        qry_fg_fts = F.normalize(qry_fg_fts, p=2, dim=1)  # shape [1, 512, 900]

        cosine_similarity = torch.sum(spt_fg_fts * qry_fg_fts, dim=1, keepdim=True)  # shape: [1, 1, N]

        return cosine_similarity

    def forward(self, spt_fg_fts, qry_fg_fts, band):
        """
        Args:
            spt_fg_fts (torch.Tensor): Spatial foreground features. 
            qry_fg_fts (torch.Tensor): Query foreground features. 
            band (str): Band type, either 'low', 'high', or other.

        Returns:
            torch.Tensor: Fused tensor. Shape: [1, 512, 5000].
        """

        spt_proj = F.relu(self.fc_spt(spt_fg_fts))  # shape: [1, 512, 900]
        qry_proj = F.relu(self.fc_qry(qry_fg_fts))  # shape: [1, 512, 900]

        similarity_matrix = self.sigmoid(self.correlation_matrix(spt_fg_fts, qry_fg_fts))
        
        if band == 'low' or band == 'high':
            weighted_spt = (1 - similarity_matrix) * spt_proj  # shape: [1, 512, 900]
            weighted_qry = (1 - similarity_matrix) * qry_proj  # shape: [1, 512, 900]
        else:
            weighted_spt = similarity_matrix * spt_proj  # shape: [1, 512, 900]
            weighted_qry = similarity_matrix * qry_proj  # shape: [1, 512, 900]

        combined = torch.cat((weighted_spt, weighted_qry), dim=2)  # shape: [1, 1024, 900]
        fused_tensor = F.relu(self.fc_fusion(combined))  # shape: [1, 512, 900]

        return fused_tensor

class FAM(nn.Module):
    def __init__(self, feature_dim=512, N=900):
        super(FAM, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.attention_matching = AttentionMacthcing(feature_dim, N)
        self.adapt_pooling = nn.AdaptiveAvgPool1d(N)

    def forward(self, spt_fg_fts, qry_fg_fts):
        """
        Forward pass of the FAM module.

        Args:
            spt_fg_fts (list): List of spatial foreground features, shot x [bs x 512 x N]. 
            qry_fg_fts (tensor): query foreground features, bs x 512 x N.
            

        Returns:
            tuple: A tuple containing the fused low, mid, and high frequency features.
        """
        if qry_fg_fts.shape[-1] == 0:
            qry_fg_fts = F.pad(qry_fg_fts, (0, 1))

        spt_fg_fts = self.adapt_pooling(spt_fg_fts)
        qry_fg_fts = self.adapt_pooling(qry_fg_fts)

        spt_fg_fts_low, spt_fg_fts_mid, spt_fg_fts_high = self.filter_frequency_bands(spt_fg_fts, cutoff=0.30)
        qry_fg_fts_low, qry_fg_fts_mid, qry_fg_fts_high = self.filter_frequency_bands(qry_fg_fts, cutoff=0.30)

        fused_fts_low = self.attention_matching(spt_fg_fts_low, qry_fg_fts_low, 'low')
        fused_fts_mid = self.attention_matching(spt_fg_fts_mid, qry_fg_fts_mid, 'mid')
        fused_fts_high = self.attention_matching(spt_fg_fts_high, qry_fg_fts_high, 'high')

        return fused_fts_low, fused_fts_mid, fused_fts_high
    


    def reshape_to_square(self, tensor):
        """
        Reshapes a tensor to a square shape.

        Args:
            tensor (torch.Tensor): The input tensor of shape (B, C, N), where B is the batch size,
                C is the number of channels, and N is the number of elements.

        Returns:
            tuple: A tuple containing:
                - square_tensor (torch.Tensor): The reshaped tensor of shape (B, C, side_length, side_length),
                  where side_length is the length of each side of the square tensor.
                - side_length (int): The length of each side of the square tensor.
                - side_length (int): The length of each side of the square tensor.
                - N (int): The original number of elements in the input tensor.
        """
        B, C, N = tensor.shape
        side_length = int(np.ceil(np.sqrt(N)))
        padded_length = side_length ** 2
        
        padded_tensor = torch.zeros((B, C, padded_length), device=tensor.device)
        padded_tensor[:, :, :N] = tensor

        square_tensor = padded_tensor.view(B, C, side_length, side_length)
        
        return square_tensor, side_length, side_length, N
    


    def filter_frequency_bands(self, tensor, cutoff=0.2):
            """
            Filters the input tensor into low, mid, and high frequency bands.

            Args:
                tensor (torch.Tensor): The input tensor to be filtered.
                cutoff (float, optional): The cutoff value for frequency band filtering.

            Returns:
                torch.Tensor: The low frequency band of the input tensor.
                torch.Tensor: The mid frequency band of the input tensor.
                torch.Tensor: The high frequency band of the input tensor.
            """

            tensor = tensor.float()
            tensor, H, W, N = self.reshape_to_square(tensor)
            B, C, _, _ = tensor.shape

            max_radius = np.sqrt((H // 2)**2 + (W // 2)**2)
            low_cutoff = max_radius * cutoff
            high_cutoff = max_radius * (1 - cutoff)

            fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1))

            def create_filter(shape, low_cutoff, high_cutoff, mode='band', device=self.device):
                rows, cols = shape
                center_row, center_col = rows // 2, cols // 2
                
                y, x = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device), indexing='ij')
                distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)
                
                mask = torch.zeros((rows, cols), dtype=torch.float32, device=device)
                
                if mode == 'low':
                    mask[distance <= low_cutoff] = 1
                elif mode == 'high':
                    mask[distance >= high_cutoff] = 1
                elif mode == 'band':
                    mask[(distance > low_cutoff) & (distance < high_cutoff)] = 1
                
                return mask

            low_pass_filter = create_filter((H, W), low_cutoff, None, mode='low')[None, None, :, :]
            high_pass_filter = create_filter((H, W), None, high_cutoff, mode='high')[None, None, :, :]
            mid_pass_filter = create_filter((H, W), low_cutoff, high_cutoff, mode='band')[None, None, :, :]

            low_freq_fft = fft_tensor * low_pass_filter
            high_freq_fft = fft_tensor * high_pass_filter
            mid_freq_fft = fft_tensor * mid_pass_filter

            low_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
            high_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
            mid_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(mid_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real

            low_freq_tensor = low_freq_tensor.view(B, C, H * W)[:, :, :N]
            high_freq_tensor = high_freq_tensor.view(B, C, H * W)[:, :, :N]
            mid_freq_tensor = mid_freq_tensor.view(B, C, H * W)[:, :, :N]

            return low_freq_tensor, mid_freq_tensor, high_freq_tensor
    

class CrossMamba(nn.Module):
    def __init__(self, d_model = 512, d_state = 16):
        super(CrossMamba, self).__init__()
        self.mamba = Cross_Block(
                    d_model,
                    mixer_cls=partial(Cross_Mamba, d_state=d_state, d_conv=4, expand=2),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
        
        self.mamba_bw = Cross_Block(
                    d_model,
                    mixer_cls=partial(Cross_Mamba, d_state=d_state, d_conv=4, expand=2),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
        
    def forward(self, _q, _v):
        for_residual = None
        forward_f, for_residual = self.mamba(_q, _v, for_residual, 
                                             inference_params=None)
        forward_f = (forward_f + for_residual) 
        
        back_residual = None
        backward_q = torch.flip(_q, [1])
        backward_v = torch.flip(_v, [1])
        backward_f, back_residual = self.mamba_bw(backward_q, backward_v, 
                                                  back_residual, inference_params=None)
        backward_f = (backward_f + back_residual) if back_residual is not None else backward_f

        backward_f = torch.flip(backward_f, [1])
        forward_f = forward_f + backward_f
        return forward_f


class CMFM(nn.Module): # Attention-based Feature Fusion Module
    def __init__(self, feature_dim):
        super(CMFM, self).__init__()
        self.CM1 = CrossMamba(d_model=feature_dim)
        self.CM2 = CrossMamba(d_model=feature_dim)
        self.relu = nn.ReLU()
    
    def forward(self, low, mid, high):
        low, mid, high = low.transpose(-2, -1), mid.transpose(-2, -1), high.transpose(-2, -1)
        low_new = self.CM1(mid, low)
        high_new = self.CM2(mid, high)
        fused_features = self.relu(low_new + mid + high_new)
        return fused_features.transpose(-2, -1)


class FewShotSeg(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = Res50Encoder(replace_stride_with_dilation=[True, True, False],
                                    pretrained_weights="COCO")  # or "ImageNet"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.scaler = 20.0
        self.criterion = nn.NLLLoss(ignore_index=255, weight=torch.FloatTensor([0.1, 1.0]).cuda())

        self.N = 900
        self.FAM = FAM(feature_dim=512, N=self.N)
        self.CMFM = CMFM(feature_dim=512)

        # Decoder
        self.head = SegmentationHead(
            in_channels=512,
            mid_channels=128,
            out_channels=2,
            activation="softmax2d",
            kernel_size=3,
            upsampling=2,
        )

        self.initialize()

    def initialize(self):
        init.initialize_fam(self.FAM)
        init.initialize_head(self.head)

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, opt, train=False):
        """
        Args:
            supp_imgs: support images
                shot x [B x 3 x H x W], tensor
            supp_mask: foreground masks for support images
                shot x [B x H x W], tensor
            qry_imgs: query images
                B x 3 x H x W, tensor
            qry_mask: label
                B x H x W, tensor
        """

        self.n_shots = len(supp_imgs)
        bs = supp_imgs.shape[0]
        img_size = supp_imgs.shape[-2:]
        assert bs == 1
        assert self.n_shots == 1

        ## Feature Extracting With ResNet Backbone
        # Extract features #
        imgs_concat = torch.cat([supp_imgs, qry_imgs], dim=0)
        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = img_fts[:bs]  # B x C x H' x W'
        qry_fts = img_fts[bs:]  # B x C x H' x W'
        
        # Get threshold #
        self.thresh_pred = tao[bs:]  # t for query features
        self.thresh_pred_ = tao[:bs]  # t for support features

        # prototype for each in the batch
        spt_fg_proto = self.getFeatures(supp_fts, supp_mask)
        
        # CPG module *******************
        qry_pred = self.getPred(qry_fts, spt_fg_proto, self.thresh_pred)  # Wa x N x H' x W'
        qry_pred_coarse = F.interpolate(qry_pred.unsqueeze(1), size=img_size, mode='bilinear', align_corners=True)

        if train:
            log_qry_pred_coarse = torch.cat([1 - qry_pred_coarse, qry_pred_coarse], dim=1).log()
            coarse_loss = self.criterion(log_qry_pred_coarse, qry_mask)
        else:
            coarse_loss = torch.zeros(1).to(self.device)

        # ************************************************
        spt_fg_fts = self.get_fg(supp_fts, supp_mask)  # (1, 512, N)
        qry_fg_fts = self.get_fg(qry_fts, qry_pred_coarse.squeeze(1))  # (1, 512, N)
        
        fused_fts_low, fused_fts_mid, fused_fts_high = self.FAM(spt_fg_fts, qry_fg_fts)
        fused_fg_fts = self.CMFM(fused_fts_low, fused_fts_mid, fused_fts_high)


        fg_proto = self.getPrototype(fused_fg_fts)
        pred = self.getPred(qry_fts, fg_proto, self.thresh_pred)  # N x Wa x H' x W'

        output_qry = self.head(pred * qry_fts)


        return output_qry, coarse_loss

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: B x C x H x W
            prototype: prototype of one semantic class
                expect shape: B x C
            thresh:
                expect shape: B x 1
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh[..., None]))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: B x C x H' x W'
            mask: binary mask, expect shape: B x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[:, None, ...], dim=(-2, -1)) \
                     / (mask[:, None, ...].sum(dim=(-2, -1)) + 1e-5)  # B x C

        return masked_fts

    def get_fg(self, fts, mask):

        """
        :param fts: (B, C, H', W')
        :param mask: (B, H, W)
        :return:
        :foreground features (1, C, N)
        """

        mask = torch.round(mask).unsqueeze(1).bool()
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        result_list = []

        for batch_id in range(fts.shape[0]):
            tmp_tensor = fts[batch_id]  
            tmp_mask = mask[batch_id]  

            foreground_features = tmp_tensor[:, tmp_mask.squeeze()]  # find the positive pixels, values

            if foreground_features.shape[1] == 1:  
                foreground_features = torch.cat((foreground_features, foreground_features), dim=1)

            result_list.append(foreground_features)  

        foreground_features = torch.stack(result_list)

        return foreground_features

    def getPrototype(self, fts):
        """
        Average the features to obtain the prototype
        :param fts:  (1, 512, N)
        :return: 1, 512, 1
        """
        N = fts.size(2)
        proto = torch.sum(fts, dim=2) / (N + 1e-5)

        return proto

    def predict_mask_nshot(self, supp_imgs, supp_mask, qry_imgs, qry_mask):

        # # Perform multiple prediction given (nshot) number of different support sets
        # nshot = len(supp_imgs)
        # logit_mask_agg = 0
        # for s_idx in range(nshot):
        #     logit_mask, _, = self(supp_imgs[s_idx], supp_mask[s_idx], qry_imgs, qry_mask, None, train=False)
        #     pred_mask = logit_mask.argmax(dim=1)
        #     logit_mask_agg += pred_mask
        #     if nshot == 1: return logit_mask_agg

        # pred_mask = logit_mask_agg.float() / nshot
        # pred_mask[pred_mask < 0.5] = 0
        # pred_mask[pred_mask >= 0.5] = 1

        # support_fg_mask = torch.cat(support_fg_mask, dim=0)

        nshot = len(supp_imgs)
        imgs_concat = torch.cat([*supp_imgs, qry_imgs], dim=0)
        img_fts, _ = self.encoder(imgs_concat)
        _, c, _, _ = img_fts.shape
        supp_fts = img_fts[:nshot].view(nshot, c, -1)
        supp_fts = torch.mean(supp_fts, dim=-1)
        qry_fts = img_fts[nshot:].view(1, c, -1)
        qry_fts = torch.mean(qry_fts, dim=-1)
        similarity = F.cosine_similarity(supp_fts, qry_fts.repeat(nshot, 1))
        max_sim = torch.argmax(similarity)

        logit_mask, _, = self(supp_imgs[max_sim], supp_mask[max_sim], qry_imgs, qry_mask, None, train=False)
        pred_mask = logit_mask.argmax(dim=1)

        return pred_mask
    
    def video_infer(self, supp_imgs, supp_mask, qry_imgs, qry_mask, supp_id=None):
        if supp_id:
            logit_mask, _, = self(supp_imgs[supp_id], supp_mask[supp_id], qry_imgs, qry_mask, None, train=False)
            pred_mask = logit_mask.argmax(dim=1)
        else:
            nshot = len(supp_imgs)
            imgs_concat = torch.cat([*supp_imgs, qry_imgs], dim=0)
            img_fts, _ = self.encoder(imgs_concat)
            _, c, _, _ = img_fts.shape
            supp_fts = img_fts[:nshot].view(nshot, c, -1)
            supp_fts = torch.mean(supp_fts, dim=-1)
            qry_fts = img_fts[nshot:].view(1, c, -1)
            qry_fts = torch.mean(qry_fts, dim=-1)
            similarity = F.cosine_similarity(supp_fts, qry_fts.repeat(nshot, 1))
            supp_id = torch.argmax(similarity)

            logit_mask, _, = self(supp_imgs[supp_id], supp_mask[supp_id], qry_imgs, qry_mask, None, train=False)
            pred_mask = logit_mask.argmax(dim=1)

        return supp_id, pred_mask

