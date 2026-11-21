"""
GRPO training loop (assignment section 4.3), on-policy. SKETCH, untested.

Real run (2 GPUs: policy on cuda:0, vLLM server on GPU 1):
    python scripts/train_grpo.py --backend vllm --seed 0 --run-name grpo_baseline

Laptop debug run (CPU, HF generation sharing weights with the policy, tiny sizes):
    python scripts/train_grpo.py --backend hf --device cpu --model-id <small causal LM> \
        --wandb-mode disabled --n-steps 3 --eval-every 1 --rollout-batch-size 8 --group-size 4 \
        --grad-accum-steps 2 --n-train-examples 16 --n-val-examples 8 --max-tokens 64 --gen-batch-size 4
"""
import argparse
from asyncio import run
import json
import random
import time
from pathlib import Path

import torch
import wandb

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import grpo_train_step  # adjust to wherever yours lives
from cs336_alignment.hf_generation import HFGenerator
from cs336_alignment.vllm_utils import VLLMServer

PROMPT_FILE = "cs336_alignment/prompts/r1_zero.prompt"
   
def load_gsm8k(path: str) -> list[dict]:
    rows = [json.loads(line) for line in open(path)]
    return [{"question": r["question"], "ground_truth": r["answer"].split("####")[-1].strip()} for r in rows]


def make_sampling_params(seed: int, n: int, max_tokens: int) -> dict:
    # vllm_utils requires "n" and "seed"; stop at the closing answer tag as the assignment specifies.
    return {
        "temperature": 1.0,
        "max_tokens": max_tokens,
        "n": n,
        "seed": seed,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }

def init_models(args):
    policy, tokenizer = get_model_and_tokenizer(args.model_id, args.device, args.dtype)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    generator = make_generator(args, policy, tokenizer)
    return policy, tokenizer, optimizer, generator

def make_generator(args, policy, tokenizer):
    """Both return objects expose start/stop/init_weight_sync/sync_policy_weights/generate_completions."""
    if args.backend == "vllm":
        server = VLLMServer(model_id=args.model_id, gpu=1, seed=args.seed)
        server.start()
        server.init_weight_sync(args.device)
        return server
    else:
        # HF backend: generate with the policy model itself, so weights are always current (sync is a no-op).
        return HFGenerator(model=policy, tokenizer=tokenizer)


def evaluate(generator, template, examples, seed, max_tokens, batch_size) -> tuple[float, dict[str, float]]:
    start_time = time.time()
    prompts = [template.format(question=ex["question"]) for ex in examples]
    params = make_sampling_params(seed, n=1, max_tokens=max_tokens)
    comps = generator.generate_completions(prompts, params, batch_size=batch_size)
    scores = [r1_zero_reward_fn(c.text, ex["ground_truth"]) for c, ex in zip(comps, examples)]
    n = len(scores)
    eval_elapsed = time.time() - start_time
    return (eval_elapsed, {
        "val/reward": sum(s["reward"] for s in scores) / n,
        "val/format_reward": sum(s["format_reward"] for s in scores) / n,
        "val/avg_response_len": sum(len(c.token_ids) for c in comps) / n,  # in tokens
    })


def is_cadence_hit (step, interval):
    """
    check if it is time to log, save checkpoint or calc validation loss
    """
    result = ( step % interval == 0)    
    return result

def upload_artifact(wr):
    import wandb

    # Use your existing initialized run.
    artifact = wandb.Artifact(name="grpo-checkpoint",type="model",)

    artifact.add_dir(
    local_path="./checkpoints/step_100",
    name="checkpoint",  # Optional folder name inside the artifact
)

    wr.log_artifact(artifact)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--device", default="cuda:0", help="policy device; use 'cpu' for laptop debugging")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="grpo")
    ap.add_argument("--wandb-project", default="cs336-a5-grpo")
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--n-train-examples", type=int, default=6400)
    ap.add_argument("--n-val-examples", type=int, default=1024)
    ap.add_argument("--rollout-batch-size", type=int, default=256)  # responses, not prompts
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--grad-accum-steps", type=int, default=32)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gen-batch-size", type=int, default=256, help="prompts per generation call; use ~4 for HF on CPU")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--log-rollouts-every", type=int, default=40)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16", help="policy dtype; vLLM is always bfloat16")
    args = ap.parse_args()
    return args

def load_data(args):  
    train = load_gsm8k("data/gsm8k/train.jsonl")
    random.Random(args.seed).shuffle(train)
    train = train[: args.n_train_examples]
    val = load_gsm8k("data/gsm8k/test.jsonl")
    val = val[: args.n_val_examples]
    return train, val

def prep_data(args, template, generator, G, batch, step):
    prompts = [template.format(question=ex["question"]) for ex in batch]
    params = make_sampling_params(args.seed * 100_000 + step, G, args.max_tokens)
    comps = generator.generate_completions(prompts, params, batch_size=args.gen_batch_size)

    # Repeat IN PLACE (p0 x G, p1 x G, ...) so index i lines up with the i-th completion.
    repeated_prompts = [p for p in prompts for _ in range(G)]
    repeated_gts = [ex["ground_truth"] for ex in batch for _ in range(G)]
    responses = [c.text for c in comps]
    assert len(responses) == len(repeated_prompts) == len(batch) * G
    return comps, repeated_prompts, repeated_gts, responses

def main(args):

    if args.backend == "vllm" and args.device == "cpu":
        raise SystemExit("--backend vllm needs a GPU; use --backend hf for CPU debugging")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train, val = load_data(args)

    G = args.group_size
    n_prompts = args.rollout_batch_size // G  # 256 // 8 = 32 prompts per step -> 200 steps = 1 epoch

    wandb_run = wandb.init( entity="manoogim-personal", project=args.wandb_project, tags=['seed-study'], name=f"{args.run_name}", config=vars(args), mode=args.wandb_mode,)
    
    policy, tokenizer, optimizer, generator = init_models(args)

    template = Path(PROMPT_FILE).read_text()

    total_step_elapsed, total_eval_elapsed, total_spw_elapsed = 0.0, 0.0, 0.0
    for step in range(args.n_steps):
        t0 = time.time()
        logs: dict = {}

        start_time = time.time()
        # 1. Put the current policy weights into the generator (vLLM: NCCL copy; HF: no-op).
        generator.sync_policy_weights(policy)
        spw_elapsed = time.time() - start_time
        total_spw_elapsed += spw_elapsed

        # 2. Periodic validation, on the freshly synced weights.
        if is_cadence_hit(step, args.eval_every):
            eval_elapsed, eval_dd = evaluate(generator, template, val, args.seed, args.max_tokens, args.gen_batch_size)
            total_eval_elapsed += eval_elapsed
            logs.update(eval_dd)

        # 3. Rollouts: n_prompts prompts x G samples each, returned prompt-major.
        start = (step * n_prompts) % len(train)
        batch = train[start : start + n_prompts]
        comps, repeated_prompts, repeated_gts, responses = prep_data(args, template, generator, G, batch, step)

        # --- off-policy hook (later) ---------------------------------------------------------
        # With train_batch_size < rollout_batch_size, loop over minibatches here. old_log_probs
        # must come from the policy BEFORE the first update on this rollout batch, and must use
        # the same tokenization/padding as each minibatch (grpo_train_step tokenizes internally,
        # so the padded width is per-minibatch). Precompute them per minibatch under no_grad.
        # --------------------------------------------------------------------------------------

        # 4. One policy-gradient step on the whole rollout batch (on-policy).
        start_time = time.time()
        loss, metadata = grpo_train_step( model=policy, tokenizer=tokenizer, optimizer=optimizer, gradient_accumulation_steps=args.grad_accum_steps, max_grad_norm=args.max_grad_norm,
            reward_fn=r1_zero_reward_fn, repeated_prompts=repeated_prompts, rollout_responses=responses, repeated_ground_truths=repeated_gts, group_size=G,)
        step_elapsed = time.time() - start_time
        total_step_elapsed += step_elapsed

        # 5. Logging.
        logs["train/loss"] = loss.item()
        meta_dd = {f"train/{k}": v for k, v in metadata.items() if isinstance(v, (int, float))}
        logs.update(meta_dd)
        logs["train/avg_response_len"] = sum(len(c.token_ids) for c in comps) / len(comps)
        logs["time/step_seconds"] = time.time() - t0
        logs["time/sync_policy_weights"] = spw_elapsed
        if is_cadence_hit(step, args.log_rollouts_every):
            rows = []
            rng = random.Random(args.seed * 100_000 + step)  # local RNG: reproducible, doesn't disturb global state
            lresp = len(responses)
            for i in sorted(rng.sample(range(lresp), k=min(16, lresp))):
                p, r, gt = repeated_prompts[i], responses[i], repeated_gts[i]
                r1z = r1_zero_reward_fn(r, gt)
                rows.append([step, p, r, gt, r1z["reward"], r1z["format_reward"]])
            logs["rollouts"] = wandb.Table(columns=["step", "prompt", "response", "ground_truth", "reward", "format_reward"], data=rows)

        wandb_run.log(logs, step=step)
        print(f"step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, float)), flush=True)

    # Final validation on the final weights, then save.
    generator.sync_policy_weights(policy)
    eval_time, dd = evaluate(generator, template, val, args.seed, args.max_tokens, args.gen_batch_size)
    total_eval_elapsed += eval_time
    wandb_run.log(dd, step=args.n_steps)
    print(f'Total eval time: {total_eval_elapsed:.2f}s, | total step time: {total_step_elapsed:.2f}s, | total sync_policy_weights time: {total_spw_elapsed:.2f}s', flush=True       )
# not saving anything on Modal - TODO make saving configurable with flag
    out = Path(args.out_dir) / f"{args.run_name}_seed{args.seed}"
    # print(f'Saving final policy and tokenizer to {out}')
    policy.save_pretrained(save_directory=out)
    # tokenizer.save_pretrained(save_directory=out)
    # print(f'Final policy and tokenizer saved to {out}')
    generator.stop()
    wandb_run.finish()

def print_env_versions():
    import importlib.metadata as md, shutil, sys
    def v(pkg):
        try: return md.version(pkg)
        except md.PackageNotFoundError: return "not installed"
    print(f"python: {sys.executable} | which python: {shutil.which('python')}", flush=True)
    print(f"torch {torch.__version__} (cuda {torch.version.cuda}), transformers {v('transformers')}, "
          f"vllm {v('vllm')}, flash-attn {v('flash-attn')}, gpus: {torch.cuda.device_count()}", flush=True)

if __name__ == "__main__":
    args = parse_args()
    print_env_versions()
    main(args)
