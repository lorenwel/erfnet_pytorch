import numpy as np
import torch



class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)

class MSELossPosElements(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = torch.nn.MSELoss(False, False)

    def forward(self, output_prob, output_cost, targets):
        shape = output_prob.size()
        cur_loss = self.loss(output_cost.expand(shape), targets.expand(shape))
        # cur_loss = self.loss(output_prob.expand(shape), targets.expand(shape))
        weighted_loss = cur_loss * output_prob
        # only compute loss for places where label exists.
        masked_loss = weighted_loss.masked_select(torch.ne(targets, -1.0))
        return masked_loss.mean()

class L1LossClassProbMasked(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = torch.nn.L1Loss(False, False)

    def forward(self, output_prob, output_cost, targets):
        targets = targets.unsqueeze(1)
        shape = output_prob.size()
        cur_loss = self.loss(output_cost.expand(shape), targets.expand(shape))
        # cur_loss = self.loss(output_prob.expand(shape), targets.expand(shape))
        weighted_loss = cur_loss * output_prob
        # only compute loss for places where label exists.
        masked_loss = weighted_loss.sum(dim=1, keepdim=True).masked_select(torch.ne(targets, -1.0))
        return masked_loss.mean()

class LogLikelihoodLossClassProbMasked(torch.nn.Module):
    # This loss ignores the 1/2 factor of the "proper" likelihood function

    def __init__(self, opt_eps=0.0):
        super().__init__()

        self.loss = torch.nn.MSELoss(False, False)
        self.opt_eps = opt_eps

    def forward(self, output_prob, output_cost, output_var, targets):
        targets = targets.unsqueeze(1)
        shape = output_prob.size()
        # Add opt_eps to avoid instabilities with very small variance
        var_expanded = output_var.expand(shape)+self.opt_eps
        mse_loss = self.loss(output_cost.expand(shape), targets.expand(shape))
        cur_loss = (2*np.pi*var_expanded).log() + mse_loss/var_expanded
        # cur_loss = self.loss(output_prob.expand(shape), targets.expand(shape))
        weighted_loss = cur_loss * output_prob
        # only compute loss for places where label exists.
        masked_loss = weighted_loss.sum(dim=1, keepdim=True).masked_select(torch.ne(targets, -1.0))
        return masked_loss.mean()

class LogLikelihoodLossMasked(torch.nn.Module):
    # This loss ignores the 1/2 factor of the "proper" likelihood function

    def __init__(self, opt_eps=0.0):
        super().__init__()

        self.loss = torch.nn.MSELoss(False, False)
        self.opt_eps = opt_eps

    def forward(self, output_cost, output_var, targets):
        mse_loss = self.loss(output_cost, targets)
        output_var = output_var + self.opt_eps
        # Add opt_eps to avoid instabilities with very small variance
        cur_loss = (2*np.pi*output_var).log() + mse_loss/output_var
        # only compute loss for places where label exists.
        masked_loss = cur_loss.masked_select(torch.ne(targets, -1.0))
        return masked_loss.mean()

class L1LossMasked(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = torch.nn.L1Loss(False, False)

    def forward(self, outputs, targets):
        return self.loss(outputs.squeeze(1), targets).masked_select(torch.ne(targets, -1.0)).mean()

class L1LossTraversability(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.non_trav_weight = torch.autograd.Variable(torch.from_numpy(np.array([0.1], dtype="float32"))).cuda()

        self.loss = torch.nn.L1Loss(False, False)

    def forward(self, outputs, targets):
        targets = targets.unsqueeze(1)
        return (1-outputs[torch.ne(targets, -1.0)]).abs().mean() + (outputs[torch.eq(targets, -1.0)]).abs().mean()*self.non_trav_weight


class L1Loss(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = torch.nn.L1Loss(True, True)

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)

class MSELossWeighted(torch.nn.Module):

    def __init__(self, scale_factor):
        super().__init__()

        self.loss = torch.nn.MSELoss(False, False)
        self.scale_factor = scale_factor

    def forward(self, outputs, targets, weight):
        return (self.loss(outputs, targets) * self.scale_factor * weight).mean()

class ClassificationAccuracy():

    def __call__(self, prob, targets):
        max_class = prob.argmax(dim=1, keepdim=True).squeeze()
        n_correct = (max_class == targets).sum().float()
        n_total = (torch.ge(targets, 0.0)).sum().float()
        return n_correct / n_total

class MeanAccuracy():

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

    def __call__(self, prob, targets):
        max_class = prob.argmax(dim=1, keepdim=True).squeeze()
        correct_pred = max_class.masked_select(max_class == targets)
        mean_acc = 0
        num_classes = 0
        class_acc = []
        for cur_class in range(0, self.n_classes):
            n_correct = (correct_pred == cur_class).sum().float()
            n_total = (targets == cur_class).sum().float()
            cur_acc = n_correct/n_total
            class_acc.append(cur_acc)
            if n_total > 0:
                num_classes += 1
                mean_acc += cur_acc

        return mean_acc / num_classes, np.array(class_acc)
