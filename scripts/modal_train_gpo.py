from __future__ import annotations

from cs336_alignment.modal_utils import app, submit_commands
steps = 200
seeds = [11, 17, 23, 29]
def make_cmd(seed: int, steps: int) -> list[str]:
    default_args =  [
    "--backend", "vllm",
    "--device", "cuda:0",
    "--model-id", "allenai/OLMo-2-0425-1B",
    "--wandb-mode", "online",
    "--run-name", f"grpo_{steps}step_seed{seed}",
    "--n-steps", str(steps),
    "--eval-every", "10",
    "--dtype", "bfloat16",
    "--seed", str(seed),
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

    cmd_arr = [make_cmd(seed, steps) for seed in seeds]
    submit_commands(cmd_arr)

