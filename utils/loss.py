import torch

def mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error."""
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
    # Single loss
    if isinstance(cfg.loss, str):
        if cfg.loss.upper() == "MSE":
            return mse_loss
        elif cfg.loss.upper() == "NSE":
            return nse_loss
        else:
            raise ValueError(f"Unknown loss: {cfg.loss}")

    # Multiple losses with weights
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
