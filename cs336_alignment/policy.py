from typing import Literal

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    # Tokenize prompt and response separately, with no special tokens, then concatenate.
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    max_len = max(len(p) + len(o) for p, o in zip(prompt_ids, output_ids))
    
     # padding value is irrelevant (it's masked out), it just has to be a valid id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    batch = len(prompt_ids)
    # right-padded full sentence
    padded = torch.full((batch, max_len), pad_id, dtype=torch.long)   # type: ignore
    response_mask = torch.zeros((batch, max_len - 1), dtype=torch.bool)

    for jj, (p, o) in enumerate(zip(prompt_ids, output_ids)):
        padded[jj, : len(p) + len(o)] = torch.tensor(p + o, dtype=torch.long)
        # Response tokens are at positions [len(p), len(p)+len(o)) of the sequence. 
        # Labels areshifted left by one, so they occupy [len(p)-1, len(p)+len(o)-1).
        response_mask[jj, len(p) - 1 : len(p) + len(o) - 1] = True

    # Slice the padded *batch*, not each sequence: shorter rows keep their final real token in input_ids
    return {"input_ids": padded[:, :-1], "labels": padded[:, 1:], "response_mask": response_mask}

import torch
import torch.nn.functional as F


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    # Don't pass labels to the HF model b/c we do not need it to compute its own loss.
    model_output = model(input_ids=input_ids)
    logits = model_output.logits  # (B, S, V)
    logits = logits.float()  # fp32 for a stable log-softmax even if the model runs in bf16/fp16
    vocab = logits.shape[-1]

    # Per-token NLL via the fused cross-entropy; 
    nll = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), reduction="none")
    # negate to get log p(label_t | prefix).
    out = {"log_probs": (-nll).view_as(labels)}  # (B, S), keeps grad

    if return_token_entropy:
        with torch.no_grad(): 
            logp = torch.log_softmax(logits.detach(), dim=-1)
            out["token_entropy"] = -(logp.exp() * logp).sum(dim=-1)  # (B, S)
    return out


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.

    """
    # temp limitation is that we support  only on-policy gradient
    # also we ignore old_log_probs, cliprange and response mask in this initial version
    if importance_reweighting_method != 'none':
        raise NotImplementedError

    token_loss = - raw_rewards_or_advantages * policy_log_probs
    return token_loss, {}



