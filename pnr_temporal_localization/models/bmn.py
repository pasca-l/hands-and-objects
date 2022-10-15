import math
import numpy as np
import torch, torch.fx
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim


class System():
    def __init__(self):
        self.model = BMNwithHead()
        self.loss = BMNLossFunc()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        self.label_transform = BMNLabelTransform()


class BMNwithHead(nn.Module):
    def __init__(self, frame_num=32):
        super().__init__()
        self.frame_num = frame_num

        resnet3d = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        resnet3d_modules = nn.ModuleList([*list(resnet3d.blocks.children())])
        self.backbone = nn.Sequential(*resnet3d_modules[:-1])
        self.model = BoundaryMatchingNetwork(self.frame_num)

    def forward(self, x):
        x.requires_grad = True
        f = self.backbone(x)
        f = f.permute(0,1,3,4,2).reshape(-1, 224*224*2, self.frame_num)
        conf_map, start, end = self.model(f)

        return [conf_map, start, end]


class BoundaryMatchingNetwork(nn.Module):
    def __init__(self, frame_num):
        super().__init__()
        self.tscale = frame_num
        self.prop_boundary_ratio = 0.5
        self.num_sample = 32
        self.num_sample_perbin = 3
        self.feat_dim = 224 * 224 * 2

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True)
        )

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        base_feature = self.x_1d_b(x)
        start = self.x_1d_s(base_feature).squeeze(1)
        end = self.x_1d_e(base_feature).squeeze(1)
        confidence_map = self.x_1d_p(base_feature)
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        confidence_map = self.x_2d_p(confidence_map)
        return confidence_map, start, end

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.tscale):
            mask_mat_vector = []
            for start_index in range(self.tscale):
                if start_index <= end_index:
                    p_xmin = start_index
                    p_xmax = end_index + 1
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)


class BMNLabelTransform():
    def __init__(self, frame_num=32):
        self.temporal_scale = frame_num
        self.temporal_gap = 1 / self.temporal_scale
        self.anchor_xmin = [self.temporal_gap * (i - 0.5) 
                            for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) 
                            for i in range(self.temporal_scale)]
        self.prec_frame_time = 0.0
        self.duration = 8

    def __call__(self, batch):
        # change the measurement from second to percentage
        info = batch[2]
        device = batch[0].device
        keyframe_time = (info["clip_pnr_frame"] - info["clip_start_frame"]) / 30
        tmp_start = torch.fmax(torch.fmin(torch.ones(len(batch[0])), torch.tensor((self.prec_frame_time / self.duration))), torch.tensor((0))).to(device)
        tmp_end = torch.fmax(torch.fmin(torch.tensor((1)), keyframe_time / self.duration), torch.tensor((0))).to(device)

        #generate R_s and R_e
        # gt_bbox = np.array([tmp_start.to('cpu').detach().numpy(), tmp_end.to('cpu').detach().numpy()])
        # gt_xmins = gt_bbox[:, 0]
        # gt_xmaxs = gt_bbox[:, 1]
        # gt_len_small = 3 * self.temporal_gap
        # gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        # gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        # gt_iou_map = np.zeros([self.temporal_scale, self.temporal_scale])
        # for i in range(self.temporal_scale):
        #     for j in range(i, self.temporal_scale):
        #         gt_iou_map[i, j] = np.max(
        #             self._iou_with_anchors(i * self.temporal_gap, (j + 1) * self.temporal_gap, gt_xmins, gt_xmaxs))
        # gt_iou_map = torch.Tensor(gt_iou_map).to(device)

        gt_len_small = 3 * self.temporal_gap
        gt_start_bboxs = torch.stack((tmp_start - gt_len_small / 2, tmp_start + gt_len_small / 2), axis=1)
        gt_end_bboxs = torch.stack((tmp_end - gt_len_small / 2, tmp_end + gt_len_small / 2), axis=1)

        gt_iou_map = torch.zeros(len(batch[0]), self.temporal_scale, self.temporal_scale)
        for i in range(self.temporal_scale):
            for j in range(i, self.temporal_scale):
                gt_iou_map[:, i, j] = self._iou_with_anchors(torch.as_tensor(i * self.temporal_gap), torch.as_tensor((j + 1) * self.temporal_gap), tmp_start, tmp_end)

        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_start.append(self._ioa_with_anchors(torch.as_tensor(self.anchor_xmin[jdx]), torch.as_tensor(self.anchor_xmax[jdx]), gt_start_bboxs[:, 0], gt_start_bboxs[:, 1]))
        match_score_end = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_end.append(self._ioa_with_anchors(torch.as_tensor(self.anchor_xmin[jdx]), torch.as_tensor(self.anchor_xmax[jdx]), gt_end_bboxs[:, 0], gt_end_bboxs[:, 1]))
        match_score_start = torch.cat(match_score_start).reshape(len(batch[0]), -1)
        match_score_end = torch.cat(match_score_end).reshape(len(batch[0]), -1)

        return [gt_iou_map, match_score_start, match_score_end]

    def _ioa_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        # calculate the overlap proportion between the anchor and all bbox for supervise signal,
        # the length of the anchor is 0.01
        len_anchors = anchors_max - anchors_min
        int_xmin = torch.maximum(anchors_min, box_min)
        int_xmax = torch.minimum(anchors_max, box_max)
        inter_len = torch.maximum(int_xmax - int_xmin, torch.zeros(int_xmin.shape).to(int_xmin.device))
        scores = torch.divide(inter_len, len_anchors)
        return scores

    def _iou_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        """Compute jaccard score between a box and the anchors.
        """
        len_anchors = anchors_max - anchors_min
        int_xmin = torch.maximum(anchors_min, box_min)
        int_xmax = torch.minimum(anchors_max, box_max)
        inter_len = torch.maximum(int_xmax - int_xmin, torch.zeros(int_xmin.shape).to(int_xmin.device))
        union_len = len_anchors - inter_len + box_max - box_min
        jaccard = torch.divide(inter_len, union_len)
        return jaccard


class BMNLossFunc(nn.Module):
    def __init__(self, frame_num=32):
        super().__init__()
        self.temporal_scale = frame_num

    def forward(self, output, target):
        conf_map, start, end = output
        label_conf, label_start, label_end = target

        loss = self.bmn_loss_func(conf_map, start, end, label_conf, label_start, label_end)

        return loss

    def bmn_loss_func(self, pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end):
        pred_bm_reg = pred_bm[:, 0].contiguous()
        pred_bm_cls = pred_bm[:, 1].contiguous()

        bm_mask = np.triu(np.ones((self.temporal_scale, self.temporal_scale)))
        bm_mask = torch.Tensor(bm_mask).to(pred_bm.device)
        gt_iou_map = gt_iou_map.to(pred_bm.device) * bm_mask

        pem_reg_loss = self.pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
        pem_cls_loss = self.pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)
        tem_loss = self.tem_loss_func(pred_start, pred_end, gt_start, gt_end)

        loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
        return loss

    def tem_loss_func(self, pred_start, pred_end, gt_start, gt_end):
        def bi_loss(pred_score, gt_label):
            gt_label = gt_label.view(-1)
            pred_score = pred_score.view(-1, gt_label.shape[0])
            pmask = (gt_label > 0.5).float()
            num_entries = len(pmask)
            num_positive = torch.sum(pmask)
            ratio = num_entries / num_positive
            coef_0 = 0.5 * ratio / (ratio - 1)
            coef_1 = 0.5 * ratio
            epsilon = 0.000001
            loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
            loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask)
            loss = -1 * torch.mean(loss_pos + loss_neg, dim=1)
            return loss

        loss_start = bi_loss(pred_start, gt_start)
        loss_end = bi_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss

    def pem_reg_loss_func(self, pred_score, gt_iou_map, mask):
        u_hmask = (gt_iou_map > 0.7).float()
        u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
        u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
        u_lmask = u_lmask * mask

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = num_h / (num_m + 1e-10)
        u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).to(u_mmask.device)
        u_smmask = u_mmask * u_smmask
        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = num_h / (num_l + 1e-10)
        u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).to(u_lmask.device)
        u_slmask = u_lmask * u_slmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        weights = u_hmask + u_smmask + u_slmask

        loss = nnf.mse_loss(pred_score * weights, gt_iou_map * weights)
        loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).to(loss.device)) / (torch.sum(weights)+1e-10)

        return loss

    def pem_cls_loss_func(self, pred_score, gt_iou_map, mask):
        pmask = (gt_iou_map > 0.9).float()
        nmask = (gt_iou_map <= 0.9).float()
        nmask = nmask * mask

        num_positive = torch.sum(pmask)
        num_entries = num_positive + torch.sum(nmask)
        ratio = num_entries / (num_positive+1e-10)
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
        loss = -1 * torch.sum(loss_pos + loss_neg) / (num_entries+1e-10)
        return loss
