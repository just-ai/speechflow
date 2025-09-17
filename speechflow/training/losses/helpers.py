from torch.nn import functional as F

__all__ = ["get_loss_from_name"]


def get_loss_from_name(name: str) -> callable:
    if name == "l1":
        return F.l1_loss
    elif name == "smooth_l1":
        return F.smooth_l1_loss
    elif name == "l2":
        return F.mse_loss
    elif name == "CE":
        return F.cross_entropy
    elif name == "BCEl":
        return F.binary_cross_entropy_with_logits
    else:
        raise NotImplementedError(f"[get_loss_from_name] {name} loss is not implemented")
