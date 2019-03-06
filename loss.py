import torch

class am_softmax_loss(torch.nn.Module):
    def __init__(self, ce_loss, class_num, scale=30, margin=0.35):
        super(am_softmax_loss, self).__init__()
        self.class_num = class_num
        self.ce_loss = ce_loss
        self.scale = scale
        self.margin = margin

    def forward(self, logits, one_hot_targets, targets):
        logits_ = one_hot_targets * (logits - self.margin) + (1. - one_hot_targets) * logits
        logits_ *= self.scale 
        return self.ce_loss(logits_,targets)
