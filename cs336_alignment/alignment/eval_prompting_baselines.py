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


# ---------- generation backends ----------
def generate_vllm(prompts, sampling, batch_size):
    from cs336_alignment.vllm_utils import VLLMServer  # only imported on GPU machines

    server = VLLMServer(model_id=MODEL_ID, gpu=0, seed=0, gpu_memory_utilization=0.9)
    server.start()
    completions = server.generate_completions(prompts, sampling, batch_size=batch_size)
    return [c.text for c in completions]


def generate_hf(prompts, sampling, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32  # fp32 is faster on CPU
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()

    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(device)
        gen_kwargs = dict(
            do_sample=True,
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            max_new_tokens=sampling["max_tokens"],
            pad_token_id=tok.pad_token_id,
        )
        if "stop" in sampling:
            gen_kwargs["stop_strings"] = sampling["stop"]
            gen_kwargs["tokenizer"] = tok
        with torch.inference_mode():
            out = model.generate(**enc, **gen_kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        outs.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outs


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
    ap.add_argument("--prompt", choices=PROMPTS, default="r1_zero_three_shot", help="which prompt to evaluate")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="hf", help="vllm is faster but requires a GPU; hf works on CPU")
    ap.add_argument("--data", default="data/gsm8k/test.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="evaluate a random subset")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="eval_outputs")
    args = ap.parse_args()

    prompt_file, reward_fn, use_stop = PROMPTS[args.prompt]
    template = (PROMPT_DIR / prompt_file).read_text()

    examples = load_gsm8k(args.data, args.limit, args.seed)
    prompts = [format_prompt(template, ex["question"]) for ex in examples]

    sampling = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512}
    if use_stop:  # only for r1_zero / r1_zero_three_shot, NOT question_only
        sampling["stop"] = ["</answer>"]
        sampling["include_stop_str_in_output"] = True

    gen = generate_vllm if args.backend == "vllm" else generate_hf
    responses = gen(prompts, sampling, args.batch_size)

    # ---- grade + categorize ----
    records, counts = [], Counter()
    for ex, prompt, resp in zip(examples, prompts, responses):
        rewards = reward_fn(resp, ex["ground_truth"])
        cat = categorize(rewards)
        counts[cat] += 1
        records.append(
            {
                "prompt": prompt,
                "response": resp,
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

    # Dump categories 2 and 3 separately: part (a) asks you to read >= 10 of
    # category 2 and judge by hand how many were actually correct but mis-parsed.
    for cat in ("cat2_format1_answer0", "cat3_format0_answer0"):
        subset = [r for r in records if r["category"] == cat]
        with open(out / f"{args.prompt}_{cat}.jsonl", "w") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()