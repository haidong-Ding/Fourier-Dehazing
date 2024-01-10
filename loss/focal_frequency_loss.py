import torch
import torch.nn as nn

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft


class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, patch_factor=1, ave_spectrum=False):
        super(FocalFrequencyLoss, self).__init__()
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq):
        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        return torch.mean(freq_distance)

    def forward(self, pred, target, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq)
