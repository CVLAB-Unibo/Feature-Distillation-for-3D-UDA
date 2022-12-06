import torch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hesiod import hcfg

class Paws_loss(nn.Module):
    """
    Make semi-supervised PAWS loss
    :param multicrop: number of small multi-crop views
    :param tau: cosine similarity temperature
    :param T: target sharpenning temperature
    :param me_max: whether to perform me-max regularization
    """
    def __init__(self, tau=0.1, T=0.25, me_max=False):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.tau = tau
        self.T = T
        self.me_max = me_max

    # def sharpen(self, p):
    #     sharp_p = p**(1./self.T)
    #     sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    #     return sharp_p

    def snn(self, query, supports, labels):
        """ Soft Nearest Neighbours similarity classifier """
        # Step 1: normalize embeddings
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

        # Step 3: compute similarlity between local embeddings
        return self.softmax(query @ supports.T / self.tau) @ labels

    def forward(self, 
                anchor_views, anchor_supports, anchor_support_labels
                # target_views, target_supports, target_support_labels
                ):
        
        # Step 1: compute anchor predictions
        probs = self.snn(anchor_views, anchor_supports, anchor_support_labels)
        probs = F.softmax(probs, dim=1)

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(anchor_support_labels**(-probs)), dim=1))

        # Step 4: compute me-max regularizer
        # rloss = 0.
        # if self.me_max:
        #     avg_probs = torch.mean(self.sharpen(probs), dim=0)
        #     rloss -= torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(len(avg_probs))

        return loss


class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.1, student_temp=0.5,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
    # self.center_momentum = center_momentum
        # self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        teacher_out = F.softmax((teacher_output/ self.teacher_temp), dim=-1)
        total_loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)

        # self.update_center(teacher_output)
        return total_loss.mean()

    # @torch.no_grad()
    # def update_center(self, teacher_output):
    #     """
    #     Update center used for teacher output.
    #     """
    #     batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
    #     batch_center = batch_center / (len(teacher_output))

    #     # ema update
    #     self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2

def get_loss_fn(weight=None):
    if hcfg("losses.loss_fn") == "crossentropy":
        return nn.CrossEntropyLoss(weight=weight)
    if hcfg("losses.loss_fn") == "chamfer":
        return ChamferLoss()
