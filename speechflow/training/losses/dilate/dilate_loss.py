import torch

from speechflow.training.losses.dilate import path_soft_dtw, soft_dtw


def dilate_loss(outputs, targets, alpha: float = 0.5, gamma: float = 0.001):
    device = outputs.device
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]

    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    D = torch.zeros((batch_size, N_output, N_output)).to(device)
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(
            targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1)
        )
        D[k : k + 1, :, :] = Dk

    loss_shape = softdtw_batch(D, gamma)

    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)
    omega = soft_dtw.pairwise_distances(torch.range(1, N_output).view(N_output, 1)).to(
        device
    )
    loss_temporal = torch.sum(path * omega) / (N_output * N_output)
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal
