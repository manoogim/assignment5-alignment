"""
HF `generate` backend exposing the same interface as VLLMServer.generate_completions,
so eval / GRPO scripts can swap backends without changing call sites.

Put this in cs336_alignment/hf_generation.py.

Accepts the same sampling_params dict that vllm_utils.generate_completions expects:
    temperature, max_tokens, n, seed, and optionally stop / include_stop_str_in_output
(top_p is optional here; vllm_utils doesn't send it, so the server default of 1.0 applies).
Returns list[VLLMCompletion] ordered prompt-major (prompt 0's n samples, then prompt 1's, ...),
matching the vLLM completions API.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.vllm_utils import VLLMCompletion  # safe to import without vllm installed


class HFGenerator:
    def __init__(self, model_id: str | None = None, model=None, tokenizer=None, device: str | None = None):
        """Either pass model_id to load, or pass an already-loaded model + tokenizer
        (e.g. the policy you're training, for a CPU smoke test)."""
        if model is None:
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device == "cuda" else torch.float32  # T4: no bf16; CPU: fp32 faster
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, attn_implementation="sdpa"
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or str(next(model.parameters()).device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def generate_completions(
        self,
        prompts: list[str],
        sampling_params: dict,
        batch_size: int | None = None,
    ) -> list[VLLMCompletion]:
        n = sampling_params.get("n", 1)
        max_tokens = sampling_params["max_tokens"]
        stop = sampling_params.get("stop")
        include_stop = sampling_params.get("include_stop_str_in_output", False)
        if sampling_params.get("seed") is not None:
            torch.manual_seed(sampling_params["seed"])
        batch_size = batch_size or len(prompts)
        pad_id = self.tokenizer.pad_token_id

        was_training = self.model.training
        old_side = self.tokenizer.padding_side
        self.model.eval()
        self.tokenizer.padding_side = "left"  # required for batched decoder-only generation

        # Longest prompts first: batches pad to their longest prompt, so grouping similar lengths
        # cuts padding/KV memory, and the worst-case batch runs first so an OOM shows up immediately
        # instead of hours in.
        order = sorted(range(len(prompts)), key=lambda j: len(prompts[j]), reverse=True)
        per_prompt = [None] * len(prompts)  # original index -> list of n completions
        try:
            for i in range(0, len(order), batch_size):
                idxs = order[i : i + batch_size]
                batch = [prompts[j] for j in idxs]
                enc = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.device)
                kwargs = dict(
                    do_sample=True,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params.get("top_p", 1.0),
                    max_new_tokens=max_tokens,
                    num_return_sequences=n,
                    pad_token_id=pad_id,
                )
                if stop:
                    kwargs["stop_strings"] = stop
                    kwargs["tokenizer"] = self.tokenizer
                out = self.model.generate(**enc, **kwargs)
                new_tokens = out[:, enc["input_ids"].shape[1] :].tolist()

                batch_completions = []
                for ids in new_tokens:
                    while ids and ids[-1] == pad_id:  # strip right padding (and the EOS if pad == EOS)
                        ids.pop()
                    finish = "length" if len(ids) >= max_tokens else "stop"
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    if stop:
                        # HF stops at the end of the token containing the stop string; vLLM truncates
                        # exactly at the stop string. Match vLLM.
                        hits = [(text.find(s), s) for s in stop if s in text]
                        if hits:
                            start, s = min(hits)  # earliest stop string wins
                            text = text[: start + (len(s) if include_stop else 0)]
                    batch_completions.append(VLLMCompletion(text=text, token_ids=ids, finish_reason=finish))
                for k, j in enumerate(idxs):  # HF returns n consecutive rows per prompt
                    per_prompt[j] = batch_completions[k * n : (k + 1) * n]
                del out, enc
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()  # release the KV cache so fragmentation doesn't accumulate
                print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)} prompts", flush=True)
        finally:
            self.tokenizer.padding_side = old_side
            if was_training:
                self.model.train()
        return [c for group in per_prompt for c in group]  # original prompt order, prompt-major