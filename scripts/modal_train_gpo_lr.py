from __future__ import annotations

from cs336_alignment.modal_utils import app, submit_commands
steps = 200
seed = 7
lr_arr = [3e-6, 2e-5, 5e-5]
wandb_project = "cs336-alignment_lr_abla"
def make_cmd(seed: int, steps: int, lr: float) -> list[str]:
    default_args =  [
    "--backend", "vllm",
    "--device", "cuda:0",
    "--model-id", "allenai/OLMo-2-0425-1B",
    "--wandb-mode", "online",
    "--run-name", f"grpo_{steps}step-lr{lr}_seed{seed}",
    "--n-steps", str(steps),
    "--eval-every", "10",
    "--dtype", "bfloat16",
    "--seed", str(seed),
    "--lr", str(lr),
    "--wandb-project", wandb_project
    ]
    cmd = ["python", "-u", "scripts/train_grpo.py", *default_args]
    return cmd

# DEFAULT_ARGS = [
#     "--backend", "vllm",
#     "--device", "cuda:0",
#     "--model-id", "allenai/OLMo-2-0425-1B",
#     "--wandb-mode", "online",
#     "--run-name", f"grpo_{steps}step_seed{seed}",
#     "--n-steps", str(steps),
#     "--eval-every", "10",
#     "--dtype", "bfloat16",
#     "--seed", str(seed),
# ]

@app.local_entrypoint()
def main(*argv: str) -> None:
    # Since argparse uses "last flag wins", any flags passed via CLI
    # automatically override the matching flags in DEFAULT_ARGS.

    cmd_arr = [make_cmd(seed, steps, lr) for lr in lr_arr]
    submit_commands(cmd_arr)

