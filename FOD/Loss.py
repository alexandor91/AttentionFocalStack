import torch
import torch.nn as nn

from torch.nn import functional as F


# ---------------------------------------------------------------------
# Utilities (same as your original)
# ---------------------------------------------------------------------
def compute_scale_and_shift(prediction, target, mask):
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    divisor = torch.sum(M)
    if divisor == 0:
        return 0
    return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    valid = M.nonzero()
    image_loss[valid] = image_loss[valid] / M[valid]
    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))
    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))

    diff = (prediction - target) * mask
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    grad_x = grad_x * mask_x

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    grad_y = grad_y * mask_y

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()
        self.__reduction = reduction_batch_based if reduction == 'batch-based' else reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()
        self.__reduction = reduction_batch_based if reduction == 'batch-based' else reduction_image_based
        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0
        for scale in range(self.__scales):
            step = pow(2, scale)
            total += gradient_loss(prediction[:, ::step, ::step],
                                   target[:, ::step, ::step],
                                   mask[:, ::step, ::step],
                                   reduction=self.__reduction)
        return total


# ---------------------------------------------------------------------
# Focal Stack Adapted Scale-and-Shift Invariant Loss
# ---------------------------------------------------------------------
class FocalStackDepthLoss(nn.Module):
    """
    Adapted loss for depth estimation from focal stack or transformed focus cues.
    Combines:
      1. Scale-and-shift invariant data loss (per-slice)
      2. Gradient regularization
      3. Optional focus consistency loss between adjacent focal slices
    """

    def __init__(self, alpha=0.5, beta_focus_consistency=0.1, scales=4, reduction='batch-based'):
        super().__init__()
        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha
        self.__beta = beta_focus_consistency
        self.__prediction_ssi = None

    def forward(self, prediction, target):
        """
        Args:
            prediction: (B, N, H, W) predicted depth maps for N focal slices or cues
            target: (B, H, W) ground-truth depth
        """
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(1)  # (B,1,H,W)

        B, N, H, W = prediction.shape
        mask = target > 0
        total_loss = 0.0
        pred_ssi_all = []

        # Compute SSI loss per focal slice
        for i in range(N):
            pred_slice = prediction[:, i, :, :]
            scale, shift = compute_scale_and_shift(pred_slice, target, mask)
            pred_ssi = scale.view(-1, 1, 1) * pred_slice + shift.view(-1, 1, 1)
            pred_ssi_all.append(pred_ssi)

            loss_data = self.__data_loss(pred_ssi, target, mask)
            loss_reg = self.__regularization_loss(pred_ssi, target, mask)
            total_loss += loss_data + self.__alpha * loss_reg

        # Average across focal slices
        total_loss = total_loss / N
        pred_ssi_all = torch.stack(pred_ssi_all, dim=1)

        # Optional: focus consistency (adjacent slices should have similar gradient structure)
        if self.__beta > 0 and N > 1:
            for i in range(N - 1):
                diff = torch.abs(pred_ssi_all[:, i + 1] - pred_ssi_all[:, i])
                total_loss += self.__beta * torch.mean(diff)

        self.__prediction_ssi = pred_ssi_all
        return total_loss

    @property
    def prediction_ssi(self):
        return self.__prediction_ssi
