from collections.abc import Callable
from typing import Literal

import torch
from transformers import PreTrainedTokenizerBase

from cs336_alignment.losses import aggregate_loss_across_microbatch
from cs336_alignment.policy import compute_policy_gradient_loss, get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.rewards import compute_group_normalized_rewards, compute_rollout_rewards


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps microbatches.
    
        Args:
            model: PreTrainedModel
                HuggingFace model to train.
            tokenizer: PreTrainedTokenizer
                Tokenizer to use for tokenization.
            optimizer: Optimizer
                Optimizer for the model.
            gradient_accumulation_steps: int
                Number of microbatches per optimizer step.
            max_grad_norm: float | None
                If not None, clip the gradient norm to this value before calling
                optimizer.step().
            reward_fn: Callable[[str, str], dict[str, float]]
                Scores the rollout responses against the ground truths, producing
                a dict with keys "reward", "format_reward", and "answer_reward".
            repeated_prompts: list[str]
                The prompts for the examples. The length of this list is
                rollout_batch_size, because the prompt for each example is repeated
                group_size times.
            rollout_responses: list[str]
                Rollouts from the policy. The length of this list is
                rollout_batch_size = n_prompts_per_rollout_batch * group_size.
            repeated_ground_truths: list[str]
                The ground truths for the examples. The length of this list is
                rollout_batch_size, because the ground truth for each example is
                repeated group_size times.
            group_size: int
                Number of responses per question (group).
            baseline: Literal["mean", "none"]
                If mean, subtract the per-group mean reward; if none, do nothing.
            advantage_eps: float
                Small constant to avoid division by zero in normalization.
            advantage_normalizer: Literal["std", "none", "mean"]
                If std, divide by the per-group standard deviation; if none, do
                nothing; if mean, divide by the per-group mean reward.
            importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
                "none": no importance reweighting; "noclip": apply importance
                reweighting without clipping; "grpo": do PPO/GRPO-style token-level
                reweighting and clipping; "gspo": do GSPO-style sequence-level
                reweighting and clipping.
            old_log_probs: torch.Tensor | None
                Required unless importance_reweighting_method = "none"; shape
                (batch_size, sequence_length).
            cliprange: float | None = None
                Clip parameter epsilon, required when importance_reweighting_method
                is "grpo" or "gspo".
            loss_normalization: Literal["sequence", "constant"] = "sequence"
                "sequence": average loss over each sequence, then average over
                sequences; "constant": normalize total loss by a constant (fixed
                for all of training).
            normalization_constant: int | None = None
                The constant to divide total loss by; required if
                loss_normalization = "constant".
    
        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]].
                loss
                    scalar tensor. The batch loss, adjusted for gradient
                    accumulation. We return this so we can log it.
                metadata
                    Dict with metadata from the underlying loss call, gradient norm
                    before clipping, and any other statistics you might want to log.
        """

    # tokenize the prompts and responses, and get the response mask
    tok_dd = tokenize_prompt_and_output( repeated_prompts, rollout_responses, tokenizer)

    device = next(model.parameters()).device
    model.train()

#    rewards 
    raw_rewards, rr_metadata = compute_rollout_rewards( reward_fn, rollout_responses, repeated_ground_truths)
    gn_rewards, gn_metadata = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer  )

    n_rollouts = len(repeated_prompts)
    microbatch_size = n_rollouts // gradient_accumulation_steps 

    total_loss, entropy_sum, n_response_tokens = 0.0, 0.0, 0

    for start in range(0, n_rollouts, microbatch_size):
        slice_range = slice(start, start + microbatch_size)
        microbatch_inputs = tok_dd['input_ids'][slice_range].to(device)
        microbatch_labels = tok_dd['labels'][slice_range].to(device)
        microbatch_mask = tok_dd['response_mask'][slice_range].to(device)
        microbatch_rewards = gn_rewards[slice_range].to(device)
        microbatch_old_log_probs = old_log_probs[slice_range].to(device) if old_log_probs is not None else None

        # forward pass
        logprob_response = get_response_log_probs(model, microbatch_inputs, microbatch_labels, True)
        microbatch_logprob = logprob_response['log_probs']                  # MB, S
        per_token_entropy = logprob_response['token_entropy']    # MB, S

        per_token_loss, loss_meta = compute_policy_gradient_loss(
            microbatch_rewards.unsqueeze(-1), microbatch_logprob, importance_reweighting_method, microbatch_old_log_probs, cliprange, microbatch_mask
        )
        microbatch_loss = aggregate_loss_across_microbatch(per_token_loss, microbatch_mask, loss_normalization, normalization_constant)

        if loss_normalization == "sequence":
            # each microbatch loss is a mean over ITS sequences; weight by its share of the batch.
            # (With "constant", the fixed constant already accounts for batch size: no rescale.)
            microbatch_loss = microbatch_loss * (microbatch_inputs.shape[0] / n_rollouts)

        microbatch_loss.backward()

        total_loss += microbatch_loss.item()
        entropy_sum += (per_token_entropy * microbatch_mask).sum().item()
        n_response_tokens += microbatch_mask.sum().item()

    max_norm = max_grad_norm if max_grad_norm is not None else float("inf")
    grad_norm = torch.nn.utils.clip_grad_norm_( model.parameters(), max_norm )

    optimizer.step()
    optimizer.zero_grad()

    metadata  = {**rr_metadata, **gn_metadata}
    metadata['grad_norm'] = grad_norm # type: ignore
    metadata['loss'] = total_loss
    metadata['token_entropy'] = entropy_sum / n_response_tokens if n_response_tokens > 0 else 0
    metadata['raw_rewards'] = raw_rewards.mean().item() 
    metadata['gn_rewards'] = gn_rewards.mean().item()

    return torch.tensor(total_loss), metadata 
