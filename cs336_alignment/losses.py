from typing import Literal

import torch


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,                       # B, S
    mask: torch.Tensor,                                                 # B, S    
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    if loss_normalization != 'sequence':
        raise Exception('Not implemented yet')

    resp_len = mask.sum(dim=-1)
    masked_token_loss = per_token_policy_gradient_loss * mask.float()
    sequence_loss = masked_token_loss.sum(dim=-1) / resp_len
    loss = sequence_loss.mean()
    return loss
    