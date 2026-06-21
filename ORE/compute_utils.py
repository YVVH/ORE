import torch
import torch.distributed as dist
from pathlib import Path
import unicodedata
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

CONTEXT_TEMPLATES_CACHE = None

def pca(tensor):
    original_dtype = tensor.dtype

    centered_tensor = tensor - tensor.mean(dim=0, keepdim=True)

    centered_tensor_float32 = centered_tensor.to(torch.float32)

    U, S, Vh = torch.linalg.svd(centered_tensor_float32, full_matrices=False)

    principal_component = Vh[0, :]

    return principal_component.to(original_dtype)

@torch.no_grad()
def _last_valid_index(attn_mask):
    return attn_mask.sum(dim=1).long() - 1

def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return t
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t

def l2_normalize(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

def scale_to_match(vec: torch.Tensor, ref: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return vec * (ref.norm(p=2, dim=-1, keepdim=True) / (vec.norm(p=2, dim=-1, keepdim=True) + eps))

def to_prompt(question, statement=''):
    start = 'Below is a factual question-and-answer task. Please provide the answer directly, without any additional explanation or comment. Format your output strictly as: <final>{answer}</final>\n'
    q = f'Question：{question}\n'
    end = f'Answer:'

    if statement == '':
        prompt = start + q + end
    else:
        statement += '\n'
        prompt = start + statement + q + end
    return prompt

def compute_unrel_subspace(layer_hidden_states: torch.Tensor, r: int) -> torch.Tensor:
    X = (layer_hidden_states - layer_hidden_states.mean(dim=0, keepdim=True)).float()
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    U = Vh[:r, :].T.contiguous()

    U = U.to(layer_hidden_states.dtype)
    return U

def remove_subspace(v: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    U = U.to(v.dtype)
    return v - (v @ U) @ U.T

def proj_energy(v: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    z = v @ U.to(v.dtype)
    return (z * z).sum(dim=-1)

@torch.no_grad()
def last_valid_index(attn_mask):
    return attn_mask.sum(dim=1).long() - 1


CONTEXT_TEMPLATES_CACHE = None


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        length = 10
        n_gen = 5

        print(f"Generating context templates for augmentation...")

        base_template = ["{}"]

        generated_prefixes = generate_fast(
            model,
            tok,
            ["The", "Therefore", "Because", "I", "You"],
            n_gen_per_prompt=n_gen // 5,
            max_out_len=length,
        )

        augmented_templates = [
            f.replace("{", " ").replace("}", " ") + ". {}"
            for f in generated_prefixes
        ]

        CONTEXT_TEMPLATES_CACHE = [base_template] + [augmented_templates]
        print(f"Cached context templates: {len(base_template) + len(augmented_templates)} items")

    return CONTEXT_TEMPLATES_CACHE


def generate_fast(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        prompts: List[str],
        n_gen_per_prompt: int = 1,
        top_k: int = 5,
        max_out_len: int = 200,
):
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(model.device)
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x, skip_special_tokens=True) for x in input_ids.detach().cpu().numpy().tolist()]
    return [unicodedata.normalize("NFKD", x).replace("\n", " ") for x in txt]