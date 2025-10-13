import torch

def mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error.
    """
    return torch.mean((preds - targets) ** 2)


def nse_loss(preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Nash-Sutcliffe Efficiency loss.
    Returns numerator/denominator so minimizing it maximizes NSE.
    """
    if preds.numel() == 0 or targets.numel() == 0:
        return torch.tensor(0.0, device=preds.device)

    mean_target = torch.mean(targets)
    numerator   = torch.sum((targets - preds) ** 2)
    denominator = torch.sum((targets - mean_target) ** 2)

    if denominator < 1e-6:
        return torch.tensor(0.0, device=preds.device)

    return numerator / (denominator + epsilon)


def get_loss_function(cfg):
    """
    Build a loss function based on config.yaml.
    Supports MSE, NSE, or weighted combos.
    """
    if isinstance(cfg.loss, str):
        if cfg.loss.upper() == "MSE":
            return mse_loss
        elif cfg.loss.upper() == "NSE":
            return nse_loss
        else:
            raise ValueError(f"Unknown loss: {cfg.loss}")

    elif isinstance(cfg.loss, (list, tuple)):
        weights = cfg.loss_weights
        funcs = []
        for l in cfg.loss:
            if l.upper() == "MSE":
                funcs.append(mse_loss)
            elif l.upper() == "NSE":
                funcs.append(nse_loss)
            else:
                raise ValueError(f"Unknown loss: {l}")

        def combined_loss(preds, targets):
            return sum(w * fn(preds, targets) for w, fn in zip(weights, funcs))

        return combined_loss

    else:
        raise ValueError("cfg.loss must be str or list")

def masked_mse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid (mask==1) elements only."""
    se = (preds - targets) ** 2
    se = se * mask
    denom = mask.sum().clamp_min(1.0)
    return se.sum() / denom


def masked_nse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Nash-Sutcliffe Efficiency loss over valid elements only, as a minimization objective."""
    valid = mask > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=preds.device)
    p = preds[valid]
    t = targets[valid]
    mean_t = t.mean()
    num = ((t - p) ** 2).sum()
    den = ((t - mean_t) ** 2).sum().clamp_min(eps)
    return num / den