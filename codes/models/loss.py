import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.VGG_tex_loss import VGG

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


class SumOfWeightsLoss(nn.Module):
    """ Encourages the kernel G is imitating to sum to 1 """

    def __init__(self):
        super(SumOfWeightsLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.one = torch.ones([]).float().cuda()

    def forward(self, kernel):
        return self.loss(self.one, torch.sum(kernel))


class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """

    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.2
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.abs(kernel)**self.power, torch.zeros_like(kernel))


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

def gram_matrix(features):
    N, C, H, W = features.size()
    feat_reshaped = features.view(N, C, -1)

    # Use torch.bmm for batch multiplication of matrices
    gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))

    return gram

class TextureLoss(nn.Module):
    """
    creates a criterion to compute weighted gram loss.
    """
    def __init__(self, use_weights=False):
        super(TextureLoss, self).__init__()
        self.use_weights = use_weights

        self.model = VGG(model_type='vgg19')
        self.register_buffer('a', torch.tensor(-20., requires_grad=False))
        self.register_buffer('b', torch.tensor(.65, requires_grad=False))

    def forward(self, x, maps, weights):
        input_size = x.shape[-1]
        x_feat = self.model(x, ['relu1_1', 'relu2_1', 'relu3_1'])
        layer_name = ['relu1_1', 'relu2_1', 'relu3_1']

        for i in range(3):
            weights_scaled = F.interpolate(weights, scale_factor=2**(2-i), mode='bicubic', align_corners=True)
            # compute coefficients
            coeff = weights_scaled * self.a.detach() + self.b.detach()
            coeff = torch.sigmoid(coeff)

            # weighting features and swapped maps
            maps[i] = maps[i] * coeff
            x_feat[layer_name[i]] = x_feat[layer_name[i]] * coeff

        # for large scale
        loss_relu1_1 = torch.norm(
            gram_matrix(x_feat['relu1_1']) - gram_matrix(maps[0]),
        ) / 4. / ((input_size * input_size * 1024) ** 2)

        # for medium scale
        loss_relu2_1 = torch.norm(
            gram_matrix(x_feat['relu2_1']) - gram_matrix(maps[1])
        ) / 4. / ((input_size * input_size * 512) ** 2)

        # for small scale
        loss_relu3_1 = torch.norm(
            gram_matrix(x_feat['relu3_1']) - gram_matrix(maps[2])
        ) / 4. / ((input_size * input_size * 256) ** 2)

        loss = (loss_relu1_1 + loss_relu2_1 + loss_relu3_1) / 3.

        return loss








