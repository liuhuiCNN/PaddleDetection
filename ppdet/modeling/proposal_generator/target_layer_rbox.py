# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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

import paddle

from ppdet.core.workspace import register, serializable

from ppdet.modeling.utils import bbox_util
import numpy as np


from paddle.utils.cpp_extension import load

G_USE_CUSTOM_OP = True
if G_USE_CUSTOM_OP:
    custom_ops = load(
        name="custom_jit_ops",
        sources=["ppdet/ext_op/ext_op_new/rbox_iou_op.cc", "ppdet/ext_op/ext_op_new/rbox_iou_op.cu"])

@register
@serializable
class S2ANetAnchorAssigner(object):
    def __init__(self, batch_size_per_im=256,
                 pos_fraction=0.5,
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.4,
                 min_iou_thr=0.0,
                 ignore_iof_thr=-2,
                 use_random=True):
        super(S2ANetAnchorAssigner, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.pos_fraction = pos_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_iou_thr = min_iou_thr
        self.ignore_iof_thr = ignore_iof_thr
        self.use_random = use_random

    def anchor_valid(self, anchors):
        """

        Args:
            anchor: M x 4

        Returns:

        """
        if anchors.ndim == 3:
            anchors = anchors.reshape(-1, anchor.shape[-1])
        assert anchors.ndim == 2
        anchor_num = anchors.shape[0]
        anchor_valid = np.ones((anchor_num), np.uint8)
        anchor_inds = np.arange(anchor_num)
        return anchor_inds

    def assign_anchor(self,
                      anchors,
                      gt_bboxes,
                      gt_lables,
                      pos_iou_thr,
                      neg_iou_thr,
                      min_iou_thr=0.0,
                      ignore_iof_thr=-2):
        """

        Args:
            anchors:
            gt_bboxes:[M, 5] rc,yc,w,h,angle
            gt_lables:

        Returns:

        """
        assert anchors.shape[1] == 4 or anchors.shape[1] == 5
        assert gt_bboxes.shape[1] == 4 or gt_bboxes.shape[1] == 5
        anchors_xc_yc = anchors
        gt_bboxes_xc_yc = gt_bboxes

        # calc rbox iou
        anchors_xc_yc = anchors_xc_yc.astype(np.float32)
        anchors_xc_yc = paddle.to_tensor(anchors_xc_yc)
        gt_bboxes_xc_yc = paddle.to_tensor(gt_bboxes_xc_yc)

        # call custom_ops
        #from paddle.utils.cpp_extension import load
        #custom_ops = load(
        #name="custom_jit_ops",
        #sources=["ppdet/ext_op/ext_op_new/rbox_iou_op.cc", "ppdet/ext_op/ext_op_new/rbox_iou_op.cu"])
        iou = custom_ops.rbox_iou(anchors_xc_yc, gt_bboxes_xc_yc)
        iou = iou.numpy()

        # every gt's anchor's index
        gt_bbox_anchor_inds = iou.argmax(axis=0)
        gt_bbox_anchor_iou = iou[gt_bbox_anchor_inds, np.arange(iou.shape[1])]
        gt_bbox_anchor_iou_inds = np.where(iou == gt_bbox_anchor_iou)[0]

        # every anchor's gt bbox's index
        anchor_gt_bbox_inds = iou.argmax(axis=1)
        anchor_gt_bbox_iou = iou[np.arange(iou.shape[0]), anchor_gt_bbox_inds]

        # (1) set labels=-2 as default
        labels = np.ones((iou.shape[0],), dtype=np.int32) * ignore_iof_thr

        # (2) assign ignore
        labels[anchor_gt_bbox_iou < min_iou_thr] = ignore_iof_thr

        # (3) assign neg_ids -1
        assign_neg_ids1 = anchor_gt_bbox_iou >= min_iou_thr
        assign_neg_ids2 = anchor_gt_bbox_iou < neg_iou_thr
        assign_neg_ids = np.logical_and(assign_neg_ids1, assign_neg_ids2)
        labels[assign_neg_ids] = -1

        # anchor_gt_bbox_iou_inds
        # (4) assign max_iou as pos_ids >=0
        anchor_gt_bbox_iou_inds = anchor_gt_bbox_inds[gt_bbox_anchor_iou_inds]
        # gt_bbox_anchor_iou_inds = np.logical_and(gt_bbox_anchor_iou_inds, anchor_gt_bbox_iou >= min_iou_thr)
        labels[gt_bbox_anchor_iou_inds] = gt_lables[anchor_gt_bbox_iou_inds]

        # (5) assign >= pos_iou_thr as pos_ids
        iou_pos_iou_thr_ids = anchor_gt_bbox_iou >= pos_iou_thr
        iou_pos_iou_thr_ids_box_inds = anchor_gt_bbox_inds[iou_pos_iou_thr_ids]
        labels[iou_pos_iou_thr_ids] = gt_lables[iou_pos_iou_thr_ids_box_inds]

        return anchor_gt_bbox_inds, anchor_gt_bbox_iou, labels

    def __call__(self, anchors,
                       gt_bboxes,
                       gt_labels,
                       is_crowd,
                       im_scale):

        assert anchors.ndim == 2
        assert anchors.shape[1] == 5
        assert gt_bboxes.ndim == 2
        assert gt_bboxes.shape[1] == 5

        batch_size_per_im = self.batch_size_per_im
        pos_iou_thr = self.pos_iou_thr
        neg_iou_thr = self.neg_iou_thr
        min_iou_thr = self.min_iou_thr
        ignore_iof_thr = self.ignore_iof_thr
        pos_fraction = self.pos_fraction
        use_random = self.use_random

        anchor_num = anchors.shape[0]

        # TODO: support not square image
        im_scale = im_scale[0][0]
        anchors_inds = self.anchor_valid(anchors)
        anchors = anchors[anchors_inds]
        gt_bboxes = gt_bboxes * im_scale
        is_crowd_slice = is_crowd
        not_crowd_inds = np.where(is_crowd_slice == 0)

        # Step1: match anchor and gt_bbox
        anchor_gt_bbox_inds, anchor_gt_bbox_iou, labels = self.assign_anchor(anchors, gt_bboxes, gt_labels.reshape(-1),
                                                                        pos_iou_thr, neg_iou_thr, min_iou_thr,
                                                                        ignore_iof_thr)

        # Step2: sample anchor
        pos_inds = np.where(labels >= 0)[0]
        neg_inds = np.where(labels == -1)[0]

        # Step3: make output
        anchors_num = anchors.shape[0]
        bbox_targets = np.zeros_like(anchors)
        bbox_weights = np.zeros_like(anchors)
        pos_labels = np.ones(anchors_num, dtype=np.int32) * -1
        pos_labels_weights = np.zeros(anchors_num, dtype=np.float32)

        pos_sampled_anchors = anchors[pos_inds]
        #print('ancho target pos_inds', pos_inds, len(pos_inds))
        pos_sampled_gt_boxes = gt_bboxes[anchor_gt_bbox_inds[pos_inds]]
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_util.rbox2delta(pos_sampled_anchors, pos_sampled_gt_boxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            pos_labels[pos_inds] = labels[pos_inds]
            pos_labels_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            pos_labels_weights[neg_inds] = 1.0
        return (pos_labels, pos_labels_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)
