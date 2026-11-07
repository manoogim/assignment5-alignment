"""
Prompting baselines for OLMo-2-0425-1B on GSM8K (CS336 A5, prompting_baselines).

Usage (Kaggle GPU, vLLM):
    python eval_prompting_baselines.py --prompt r1_zero --backend vllm
Usage (laptop CPU, HF generate, small subset):
    python eval_prompting_baselines.py --prompt r1_zero --backend hf --limit 20 --batch-size 4

Run once per prompt: question_only, r1_zero, r1_zero_three_shot.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.hf_generation import HFGenerator
from cs336_alignment.vllm_utils import VLLMServer

MODEL_ID = "allenai/OLMo-2-0425-1B"
PROMPT_DIR = Path("cs336_alignment/prompts")

# prompt name -> (prompt file, reward fn, use </answer> stop string?)
PROMPTS = {
    "question_only": ("question_only.prompt", question_only_reward_fn, False),
    "r1_zero": ("r1_zero.prompt", r1_zero_reward_fn, True),
    "r1_zero_three_shot": ("r1_zero_three_shot_gsm8k.prompt", r1_zero_reward_fn, True),
}


# ---------- data ----------
def load_gsm8k(path: str, limit: int | None, seed: int):
    rows = [json.loads(line) for line in open(path)]
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)  # random subset, not just first N
    examples = []
    for r in rows:
        # ground truth is the part after "####"
        gt = r["answer"].split("####")[-1].strip()
        examples.append({"question": r["question"], "ground_truth": gt})
    return examples


def format_prompt(template: str, question: str) -> str:
    # .format also turns the "{{}}" in question_only.prompt into "{}"
    return template.format(question=question)


# ---------- generation backend ----------
def make_generator(args):
    """Both backends expose .generate_completions(prompts, sampling_params, batch_size)
    -> list[VLLMCompletion], so everything downstream is backend-agnostic."""
    if args.backend == "vllm":
        # VLLMServer defaults to gpu=1 (train on 0, infer on 1); we only have one GPU here.
        server = VLLMServer(model_id=MODEL_ID, gpu=0, seed=args.seed, gpu_memory_utilization=0.9)
        server.start()
        return server
    else:
        return HFGenerator(MODEL_ID, None)


# ---------- categorization ----------
def categorize(rewards: dict) -> str:
    fmt, ans = rewards["format_reward"], rewards["answer_reward"]
    if fmt == 1 and ans == 1:
        return "cat1_format1_answer1"
    if fmt == 1 and ans == 0:
        return "cat2_format1_answer0"
    if fmt == 0 and ans == 0:
        return "cat3_format0_answer0"
    return "other_format0_answer1"  # shouldn't happen with these graders, but don't drop it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=PROMPTS, required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="hf")
    ap.add_argument("--data", default="data/gsm8k/test.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="evaluate a random subset")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="eval_outputs")
    ap.add_argument("--wandb", action="store_true", default=True, help="log summary + response table to W&B")
    ap.add_argument("--wandb-project", default="cs336-a5-baselines")
    args = ap.parse_args()

    prompt_file, reward_fn, use_stop = PROMPTS[args.prompt]
    template = (PROMPT_DIR / prompt_file).read_text()

    examples = load_gsm8k(args.data, args.limit, args.seed)
    prompts = [format_prompt(template, ex["question"]) for ex in examples]

    # vllm_utils.generate_completions requires "n" and "seed" (KeyError otherwise) and does not
    # send top_p, so the server default (1.0) applies.
    sampling = {"temperature": 1.0, "max_tokens": 512, "n": 1, "seed": args.seed}
    if use_stop:  # only for r1_zero / r1_zero_three_shot, NOT question_only
        sampling["stop"] = ["</answer>"]
        sampling["include_stop_str_in_output"] = True

    generator = make_generator(args)
    completions = generator.generate_completions(prompts, sampling, batch_size=args.batch_size)

    # ---- grade + categorize ----
    records, counts = [], Counter()
    for ex, prompt, comp in zip(examples, prompts, completions):
        resp = comp.text
        rewards = reward_fn(resp, ex["ground_truth"])
        cat = categorize(rewards)
        counts[cat] += 1
        records.append(
            {
                "prompt": prompt,
                "response": resp,
                "finish_reason": comp.finish_reason,  # "length" = hit the 512-token cap
                "ground_truth": ex["ground_truth"],
                "rewards": rewards,
                "category": cat,
            }
        )

    n = len(records)
    summary = {
        "prompt": args.prompt,
        "n_examples": n,
        "counts": dict(counts),
        "accuracy": sum(r["rewards"]["answer_reward"] for r in records) / n,
        "format_rate": sum(r["rewards"]["format_reward"] for r in records) / n,
    }
    print(json.dumps(summary, indent=2))

    # ---- save everything for manual inspection ----
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    with open(out / f"{args.prompt}_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    json.dump(summary, open(out / f"{args.prompt}_summary.json", "w"), indent=2)

    # ---- optional W&B: summary metrics + a filterable table of every response ----
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=f"{args.prompt}_{args.backend}_n{n}",
            config={**vars(args), "model": MODEL_ID, "sampling": sampling},
        )
        run.summary.update(
            {
                "accuracy": summary["accuracy"],
                "format_rate": summary["format_rate"],
                "n_examples": n,
                **{f"count/{k}": v for k, v in counts.items()},
            }
        )
        table = wandb.Table(
            columns=["category", "finish_reason", "format_reward", "answer_reward", "ground_truth", "response", "prompt"],
            data=[
                [
                    r["category"], r["finish_reason"], r["rewards"]["format_reward"],r["rewards"]["answer_reward"], r["ground_truth"], r["response"],  r["prompt"],
                ]
                for r in records
            ],
        )
        run.log({"responses": table})
        run.finish()

    # Dump categories 2 and 3 separately: part (a) asks you to read >= 10 of
    # category 2 and judge by hand how many were actually correct but mis-parsed.
    for cat in ("cat2_format1_answer0", "cat3_format0_answer0"):
        subset = [r for r in records if r["category"] == cat]
        with open(out / f"{args.prompt}_{cat}.jsonl", "w") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()