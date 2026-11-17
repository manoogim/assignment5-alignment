from collections.abc import Callable
from typing import Literal

from numpy import mean
import torch

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """

    assert len(rollout_responses) == len(repeated_ground_truths), 'Length of rollout responses and ground thruths'
    # Each rollout is scored independently against its own (repeated) ground truth.
    scores = [reward_fn(resp, gt) for resp, gt in zip(rollout_responses, repeated_ground_truths)]

    raw_rewards = torch.tensor([s["reward"] for s in scores], dtype=torch.float32)
    format_rewards = torch.tensor([s["format_reward"] for s in scores], dtype=torch.float32)
    answer_rewards = torch.tensor([s["answer_reward"] for s in scores], dtype=torch.float32)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "format_reward_mean": format_rewards.mean().item(),
        "answer_reward_mean": answer_rewards.mean().item(),
    }
    return raw_rewards, metadata



def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    def one_advantage(rewards, baseline, eps, normalizer):
        mean = rewards.mean()
        centered = rewards - mean if baseline=='mean' else rewards
        if normalizer == 'std':
            scale = rewards.std() + eps
        elif normalizer == 'mean':
            scale = mean + eps
        else:
            scale = 1.

        a = centered / scale
        return a, mean, scale

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    
    if raw_rewards.ndim != 1:
        raise ValueError("raw_rewards must be a 1D tensor")

    if raw_rewards.numel() == 0:
        raise ValueError("raw_rewards cannot be empty")
    
    if baseline not in ("mean", "none"):
        raise ValueError(f"unknown baseline {baseline!r}")
    
    if advantage_normalizer not in ("std", "none", "mean"):
        raise ValueError(f"unknown advantage_normalizer {advantage_normalizer!r}")
    
    if raw_rewards.numel() % group_size != 0:
        raise ValueError("rollout batch size must be divisible by group_size")
    
    nn = len(raw_rewards) // group_size

    advantages =[]
    n_uniform = 0
    for jj in range(nn):
        rewards = raw_rewards[jj*group_size: (jj+1)*group_size]
        a, mu, std = one_advantage(rewards, baseline, advantage_eps, advantage_normalizer)
        advantages.append(a)

        if rewards.max() == rewards.min():
            n_uniform += 1  # Count groups with identical rewards.
  
    metadata = {
        "n_uniform": n_uniform, 
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
    }
    return torch.cat(advantages), metadata