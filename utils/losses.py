import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from hesiod import hcfg

class Distillation_Loss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.1, student_temp=0.5):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        teacher_out = F.softmax((teacher_output/ self.teacher_temp), dim=-1)
        total_loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)

        return total_loss.mean()

def get_loss_fn(weight=None):
    if hcfg("losses.loss_fn") == "crossentropy":
        return nn.CrossEntropyLoss(weight=weight)
