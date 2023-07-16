from algorithms.base_framework import SingleModel
import torch.nn.functional as F
import torch

class LogitClipping(SingleModel):
    def train_batch(self, index, inputs, targets, epoch):
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        logits = self.net(inputs)

        delta = 1/self.args.temp
        norms = torch.norm(logits, p=self.args.lp, dim=-1, keepdim=True) + 1e-7
        logits_norm = torch.div(logits, norms) * delta
        clip = (norms > self.args.temp).expand(-1, logits.shape[-1])
        logits_final = torch.where(clip, logits_norm, logits)

        loss = self.loss_function(logits_final, targets)
        return loss



