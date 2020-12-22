"""
https://github.com/ElementAI/N-BEATS/blob/master/common/torch/losses.py
"""
import torch

def divide_non_nan(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    """x/y where resulting NaN or Inf are replaced by 0. 
    https://github.com/ElementAI/N-BEATS/blob/04f56c4ca4c144071b94089f7195b1dd606072b0/common/torch/ops.py#L38

    Args:
        x (torch.Tensor): [description]
        y (torch.Tensor): [description]

    Returns:
        torch.Tensor: [description]
    """
    res = x / y
    res[torch.isnan(res)] = 0.0
    res[torch.isinf(res)] = 0.0
    return res

def mape_loss(forecast: torch.Tensor, target: torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
    """Measures the Mean Absolute Percentage Error.

    Args:
        forecast (torch.Tensor): The forecasted value(s)
        target (torch.Tensor): The target value(s)
        mask (torch.Tensor): The mask indicating potentially padded zeros in the forecast.

    Returns:
        torch.Tensor: The loss.
    """
    weights = divide_non_nan(mask, target)
    return torch.mean(torch.abs(forecast - target) * weights)
 
def smape_loss(forecast: torch.Tensor, target: torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
    """Measures the Symmetric Mean Absolute Percentage Error. https://robjhyndman.com/hyndsight/smape/

    Args:
        forecast (torch.Tensor): The forecasted value(s)
        target (torch.Tensor): The target value(s)
        mask (torch.Tensor): The mask indicating potentially padded zeros in the forecast.

    Returns:
        torch.Tensor: The loss.
    """
    return 200 * torch.mean(
        divide_non_nan(
            torch.abs(forecast - target), torch.abs(forecast) + torch.abs(target)
            ) * mask
        )
    

def mase_loss(forecast, target, sample, sample_mask, frequency):
    """The Mean Absolute Scaled Error.
    TODO: Fix the naive seasonal scaling for batches containing different frequencies.
    Args:
        forecast (torch.Tensor): The forecasted value(s)
        target (torch.Tensor): The target value(s)
        sample (torch.Tensor): The insample values used to calculate the scaling.
        sample_mask (torch.Tensor): The mask indicating potentially padded zeros in the forecast.
        frequency (int): The frequency of the data used to scale by the naive seasonal forecast.

    Returns:
        torch.Tensor: The loss.
    """
    scaling = []
    #np_in_sample =
    for i, row in enumerate(sample):
        scaling.append(torch.mean(torch.abs(row[frequency[i]:] - row[:-frequency[i]]), dim=0))
    scaling = torch.tensor(scaling)
    #scaling2 = torch.mean(torch.abs(sample[:, frequency:] - sample[:, :-frequency]), dim=1)
    #scaling2 = torch.mean(torch.abs(sample[:, 12:] - sample[:, :-12]), dim=1)
    #scaling = torch.mean(torch.abs(sample[:, 12:] - sample[:, :-12]), dim=1)
    #assert torch.sum(scaling - scaling2) == 0, "not 0 scaling and scaling2"
    inv_scaling_masked = divide_non_nan(sample_mask, scaling.unsqueeze(1))

    return torch.mean(torch.abs(target - forecast) * inv_scaling_masked)




def owa_loss(forecast, target, sample, sample_mask, frequency):
    """
    TODO: This is shady because we don't know the scores of the naive2 forecast of the
          test set meaning we introduce some bias. We could calculate the naive2 score
          for a validation set. We can maybe assume it would be similar.
    """
    naive2_smape = 13.564
    naive2_mase = 1.912

    # MASE
    scaling = []

    for i, row in enumerate(sample):
        scaling.append(torch.mean(torch.abs(row[frequency[i]:] - row[:-frequency[i]]), dim=0))
    scaling = torch.tensor(scaling)
    inv_scaling_masked = divide_non_nan(sample_mask, scaling.unsqueeze(1))
    torch.mean(torch.abs(target - forecast) * inv_scaling_masked)

    mase = torch.mean(torch.abs(target - forecast) * inv_scaling_masked)

    # SMAPE
    smape = 200 * torch.mean(divide_non_nan(torch.abs(forecast - target), torch.abs(forecast) + torch.abs(target)))

    owa = 0.5 * smape / naive2_smape + 0.5 * mase / naive2_mase

    return owa