from __future__ import annotations

import contextlib
import os
import json
import math
import time
import argparse
import typing
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import List, Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from safetensors.torch import save_file
from scipy.stats import hmean
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
import gc
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, BitsAndBytesConfig, \
    get_cosine_schedule_with_warmup
from .ORE_hparams import OREHyperParams
from .ORE_rep_head import RepHead, LayerRepAdapter, ExternalRepHeadModel, _rep_heads
from .compute_utils import *
from .ORE_loss import orth_loss, kl_loss, ce_loss, gate_bce_loss
from util import nethook
from util.generate import generate_fast
from util.device import empty_cache

UNRELATED_CORPUS = None


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0;
        self.avg = 0;
        self.sum = 0;
        self.count = 0

    def update(self, val, n=1):
        self.val = val;
        self.sum += val * n;
        self.count += n
        self.avg = self.sum / self.count


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def build_loc_pool(ds_class, data_dir, tok, edit_size: int, pool_size: int = 2000) -> List[str]:
    full = ds_class(data_dir, tok=tok, size=edit_size + pool_size)
    pool = []
    for i in range(edit_size, len(full)):
        rr = full[i]["requested_rewrite"]
        loc = rr.get("loc")
        pool.append(loc if loc else rr["prompt"].format(rr["subject"]))
    if len(pool) >= 5:
        return pool
    with open(r"./ORE/unrelated_prompts.json", "r", encoding="utf-8") as f:
        return json.load(f)


def train_step(
        model: 'ExternalRepHeadModel',
        tok: AutoTokenizer,
        hparams: OREHyperParams,
        batch,
        all_unrelated_subspaces,
        current_it
):
    device = model.device

    inner_model = model
    while hasattr(inner_model, "module"):
        inner_model = inner_model.module

    src_inputs = {k: v.to(device) for k, v in batch['src'].items()}
    alt_statement_inputs = {k: v.to(device) for k, v in batch['alt_statement'].items()}
    pred_statement_inputs = {k: v.to(device) for k, v in batch['pred_statement'].items()}
    alt_inputs = {k: v.to(device) for k, v in batch['alt'].items()}
    loc_inputs = {k: v.to(device) for k, v in batch['loc'].items()}

    with torch.no_grad():
        with _rep_heads(inner_model, False):
            alt_statement_inner = model(**alt_statement_inputs, output_hidden_states=True).hidden_states
            pred_statement_inner = model(**pred_statement_inputs, output_hidden_states=True).hidden_states

            teacher_out_orth = model(**src_inputs, output_hidden_states=True)
            teacher_hidden_states_orth = teacher_out_orth.hidden_states
            teacher_logits_kl = model(**loc_inputs).logits

    for l in hparams.unfreeze_layers:
        if hasattr(inner_model.backbone.model.layers[l], '_last_basis_weights'):
            inner_model.backbone.model.layers[l]._last_basis_weights = None

    with _rep_heads(inner_model, True):
        prompt = src_inputs
        ans = alt_inputs
        B = prompt["input_ids"].size(0)
        device = prompt["input_ids"].device

        prompt_len = prompt["attention_mask"].sum(dim=1).long()

        seqs = []
        ans_spans = []
        for b in range(B):
            p_ids = prompt["input_ids"][b][prompt["attention_mask"][b].bool()]
            a_ids = ans["input_ids"][b][ans["attention_mask"][b].bool()]
            if a_ids.numel() == 0 or a_ids[-1].item() != tok.eos_token_id:
                a_ids = torch.cat([a_ids, torch.tensor([tok.eos_token_id], device=device, dtype=a_ids.dtype)])
            seqs.append(torch.cat([p_ids, a_ids]))
            ans_spans.append((int(p_ids.numel()), int(a_ids.numel())))

        max_len = max(int(s.numel()) for s in seqs)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ce_input_ids = torch.full((B, max_len), pad_id, device=device, dtype=prompt["input_ids"].dtype)
        ce_attn_mask = torch.zeros((B, max_len), device=device, dtype=prompt["attention_mask"].dtype)
        ce_labels = torch.full((B, max_len), -100, device=device, dtype=torch.long)
        for b, s in enumerate(seqs):
            n = int(s.numel())
            ce_input_ids[b, :n] = s
            ce_attn_mask[b, :n] = 1
            start, la = ans_spans[b]
            ce_labels[b, start:start + la] = s[start:start + la]

        if hasattr(inner_model, "set_intervention_boundary"):
            inner_model.set_intervention_boundary(prompt_len)

        out_ce = model(input_ids=ce_input_ids, attention_mask=ce_attn_mask, labels=ce_labels,
                       use_cache=False, output_hidden_states=True)

        max_prompt_len = prompt_len.max().item()
        src_inner_cache = {}
        for l in hparams.unfreeze_layers:
            src_inner_cache[l] = out_ce.hidden_states[l][:, :max_prompt_len, :]

        src_gate_scores = []
        for l in hparams.unfreeze_layers:
            w = inner_model.backbone.model.layers[l]._last_sig_scores
            if w is not None:
                src_gate_scores.append(w)

        if hasattr(inner_model, "set_intervention_boundary"):
            inner_model.set_intervention_boundary(None)

        if "attention_mask" in loc_inputs:
            loc_lens = loc_inputs["attention_mask"].sum(dim=1).long()
            if hasattr(inner_model, "set_intervention_boundary"):
                inner_model.set_intervention_boundary(loc_lens)

        student_logits_kl = model(**loc_inputs).logits

        loc_gate_scores = []
        for l in hparams.unfreeze_layers:
            w = inner_model.backbone.model.layers[l]._last_sig_scores
            if w is not None:
                loc_gate_scores.append(w)

        if hasattr(inner_model, "set_intervention_boundary"):
            inner_model.set_intervention_boundary(None)

    if hparams.orth_coef > 0:
        orth_mask = src_inputs["attention_mask"][:, :max_prompt_len]
        orth = orth_loss(
            hparams=hparams,
            src_origin_inner=teacher_hidden_states_orth,
            alt_statement_inner=alt_statement_inner,
            pred_statement_inner=pred_statement_inner,
            src_inner_cache=src_inner_cache,
            all_unrelated_subspaces=all_unrelated_subspaces,
            attention_mask=orth_mask,
            method=hparams.orth_method
        )
    else:
        orth = torch.tensor([0.0], device=device).mean()

    if hparams.ce_coef > 0:
        ce = ce_loss(
            out_ce=out_ce,
            tok=tok,
            prompt_len=prompt_len,
            ans_input_ids=ans["input_ids"],
            use_margin=False,
        )
    else:
        ce = torch.tensor([0.0], device=device).mean()

    if hparams.kl_coef > 0:
        kl = kl_loss(
            teacher_logits=teacher_logits_kl,
            student_logits=student_logits_kl,
            prompt_mask=loc_inputs["attention_mask"].bool()
        )
    else:
        kl = torch.tensor([0.0], device=device).mean()

    prompt_seq_len = batch['src']['attention_mask'].shape[1]
    truncated_src_gate_scores = [
        w[:, :prompt_seq_len, :] if w is not None else None
        for w in src_gate_scores
    ]

    if current_it < 10:
        loss_gate_src = gate_bce_loss(
            truncated_src_gate_scores,
            batch.get('src_subj_ranges'),
            src_inputs['attention_mask'],
            is_unrelated=False,
            pos_weight=2.0,
        )
        loss_gate_loc = gate_bce_loss(
            loc_gate_scores,
            None,
            loc_inputs['attention_mask'],
            is_unrelated=True
        )
    else:
        loss_gate_src = torch.tensor([0.0], device=device).mean()
        loss_gate_loc = torch.tensor([0.0], device=device).mean()

    total = (hparams.ce_coef * ce +
             hparams.kl_coef * kl +
             hparams.orth_coef * orth
             + loss_gate_src
             + loss_gate_loc
             )

    for l in hparams.unfreeze_layers:
        inner_model.backbone.model.layers[l]._last_ffn_out = None

    del teacher_out_orth, teacher_hidden_states_orth
    del teacher_logits_kl
    del alt_statement_inner, pred_statement_inner
    del student_logits_kl
    del src_inner_cache
    del out_ce

    return {
        "total": total,
        "kl": hparams.kl_coef * kl,
        "ce": hparams.ce_coef * ce,
        "orth": hparams.orth_coef * orth,
        "gate_src": loss_gate_src,
        "gate_loc": loss_gate_loc,
    }


def tokenized(
        tok: AutoTokenizer,
        requests: List[Dict],
        keys_to_tokenize: List[str],
        model,
        loc_pool: List[str] = None,
):
    tokenized_requests = []
    print("Pre-tokenizing all requests (Robust Splitting Method)...")

    prefixes = [
        "Therefore", "Thus",
        "However", "But", "but", "was",
        "Moreover", "Also",
        "Because", "Since",
        "Recently", "Eventually",
        "Actually", "In fact",
        "I think", "Personally",
        "Maybe", "Perhaps",
        "Well", "So", "so", "to",
        "This", "That",
        "Suddenly", "Once",
        "Reportedly", "Surprisingly", "Fortunately",
        "Finally", "Overall",
        "Alice said:", "Why"
    ]

    if not loc_pool:
        raise ValueError(
            "tokenized() requires a non-empty loc_pool "
            "(held-out prompts from the current dataset for the KL locality loss)."
        )
    loc_templates = list(loc_pool)

    raw_ctx_templates = get_context_templates(model, tok, prefixes)
    templates = [t for sublist in raw_ctx_templates for t in sublist]

    for raw in requests:
        r = deepcopy(raw)

        ctx_template = random.choice(templates[1:])
        prompts = [templates[0].format(r["prompt"]), ctx_template.format(r["prompt"])]

        for prompt in prompts:
            subject_str = r["subject"]
            target_str = r["target_new"]["str"]

            full_text = prompt.format(subject_str) + " " + target_str
            prompt_text = prompt.format(subject_str)

            r['src'] = prompt_text
            r['alt'] = target_str


            k = min(5, len(loc_templates))
            r['loc'] = random.sample(loc_templates, k)

            r['alt_statement'] = to_prompt(prompt_text, r["alt_statement"])
            r['pred_statement'] = to_prompt(prompt_text, r["pred_statement"])

            tokenized_r = {}

            full_enc = tok(full_text, add_special_tokens=True, return_offsets_mapping=True)
            full_ids = full_enc['input_ids']
            full_mask = full_enc['attention_mask']
            full_offsets = full_enc['offset_mapping']

            prompt_enc_naive = tok(prompt_text, add_special_tokens=True)['input_ids']

            split_idx = 0
            min_len = min(len(full_ids), len(prompt_enc_naive))
            for k in range(min_len):
                if full_ids[k] == prompt_enc_naive[k]:
                    split_idx += 1
                else:
                    break

            src_ids = full_ids[:split_idx]
            alt_ids = full_ids[split_idx:]
            src_mask = full_mask[:split_idx]
            alt_mask = full_mask[split_idx:]

            tokenized_r['src'] = {}
            tokenized_r['src']['input_ids'] = torch.tensor(src_ids, dtype=torch.long)
            tokenized_r['src']['attention_mask'] = torch.tensor(src_mask, dtype=torch.long)

            char_start = prompt_text.find(subject_str)
            char_end = char_start + len(subject_str)
            token_start = None
            token_end = None

            for idx, (os, oe) in enumerate(full_offsets[:split_idx]):
                if token_start is None and os <= char_start < oe:
                    token_start = idx
                if token_start is not None and oe >= char_end:
                    token_end = idx + 1
                    break

            if token_start is None: token_start = 0
            if token_end is None: token_end = len(src_ids)

            tokenized_r['src']['subj_token_range'] = [token_start, token_end]

            tokenized_r['alt'] = {
                'input_ids': torch.tensor(alt_ids, dtype=torch.long),
                'attention_mask': torch.tensor(alt_mask, dtype=torch.long)
            }

            for key in keys_to_tokenize:
                if key in ['src', 'alt']: continue

                if key == 'loc':
                    tokenized_r[key] = [
                        tok(p, add_special_tokens=True) for p in r[key]
                    ]
                else:
                    tokenized_r[key] = tok(r[key], add_special_tokens=True)

            tokenized_requests.append(tokenized_r)

    return tokenized_requests


def execute_ore(
        model: 'ExternalRepHeadModel',
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: OREHyperParams,
        all_unrelated_subspaces,
        **kwargs: Any,
):
    is_ddp = dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0
    world_size = dist.get_world_size() if is_ddp else 1
    device = model.device

    requests = deepcopy(requests)

    keys_to_tokenize = ['alt_statement', 'pred_statement', 'alt', 'loc', 'src']

    tokenized_requests = tokenized(tok, requests, keys_to_tokenize, model, loc_pool=kwargs.get("loc_pool"))

    del requests
    if rank == 0:
        print(f"Tokenization complete. {len(tokenized_requests)} requests processed.")

    total_len = len(tokenized_requests)

    local_requests = [tokenized_requests[i] for i in range(rank, total_len, world_size)]

    max_local_len = math.ceil(total_len / world_size)

    while len(local_requests) < max_local_len:
        if len(local_requests) > 0:
            local_requests.append(deepcopy(local_requests[-1]))

    if is_ddp:
        print(f"Rank {rank}: Processing {len(local_requests)} requests (After padding)")

    inner_model = model.module if hasattr(model, "module") else model

    weights = {n: p for n, p in inner_model.rep_adapters.named_parameters()}

    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    adapter_params = []
    for name, param in inner_model.rep_adapters.named_parameters():
        adapter_params.append(param)

    model.requires_grad_(False)
    for param in adapter_params:
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=hparams.lr,
    )

    steps_per_epoch = len(local_requests) // hparams.batch_size
    total_steps = steps_per_epoch * hparams.epochs
    num_warmup_steps = int(0.1 * total_steps)
    num_main_steps = total_steps - num_warmup_steps

    warmup_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_main_steps,
        eta_min=hparams.lr_min
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[num_warmup_steps]
    )

    loss_meter = AverageMeter()
    from tqdm import tqdm
    num_batches = -(-len(local_requests) // hparams.batch_size)
    pbar = tqdm(total=hparams.epochs * num_batches, desc="Training",
                disable=(rank != 0), dynamic_ncols=True)
    for it in range(hparams.epochs):
        if it == 0:
            for layer_id, adapter in inner_model.rep_adapters.items():
                for param in adapter.head.dynamic_gate.parameters():
                    param.requires_grad = True
                adapter.scale.requires_grad = True

        if it == 10:
            for layer_id, adapter in inner_model.rep_adapters.items():
                for param in adapter.head.dynamic_gate.parameters():
                    param.requires_grad = False
                adapter.scale.requires_grad = False

        loss_meter.reset()

        random.shuffle(local_requests)

        for step, batch_requests in enumerate(chunks(local_requests, hparams.batch_size)):

            batch = {}
            for key in keys_to_tokenize:
                if key == 'src':
                    src_items = [
                        {'input_ids': r['src']['input_ids'].clone().detach(),
                         'attention_mask': r['src']['attention_mask'].clone().detach()}
                        for r in batch_requests]
                    batch['src'] = tok.pad(src_items, padding=True, return_tensors="pt")
                    batch['src_subj_ranges'] = [r['src']['subj_token_range'] for r in batch_requests]
                elif key == 'alt':
                    alt_items = [
                        {'input_ids': r['alt']['input_ids'].clone().detach(),
                         'attention_mask': r['alt']['attention_mask'].clone().detach()}
                        for r in batch_requests]
                    batch['alt'] = tok.pad(alt_items, padding=True, return_tensors="pt")
                elif key == 'loc':
                    loc_flattened = []
                    for r in batch_requests:
                        loc_flattened.extend(r[key])
                    if len(loc_flattened) > 0:
                        features_keys = loc_flattened[0].keys()
                        batch_features = {
                            k: [item[k] for item in loc_flattened]
                            for k in features_keys
                        }
                        batch[key] = tok.pad(
                            batch_features,
                            padding=True,
                            return_tensors="pt"
                        )
                else:
                    items_to_pad = [r[key] for r in batch_requests]
                    batch[key] = tok.pad(items_to_pad, padding=True, return_tensors="pt")

            optimizer.zero_grad()

            losses = train_step(
                model,
                tok,
                hparams,
                batch,
                all_unrelated_subspaces,
                current_it=it
            )

            loss = losses["total"]

            loss_meter.update(loss.item(), n=len(batch_requests))

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({"ep": it, **{k: f"{v.item():.3f}" for k, v in losses.items()}})

            loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(
                parameters=[v for _, v in weights.items()], max_norm=1.0
            )
            if torch.isfinite(total_norm):
                optimizer.step()
                scheduler.step()
        if rank == 0:
            pbar.write(f"[Epoch {it}] avg loss {loss_meter.avg:.4f}")

    pbar.close()

    if is_ddp:
        dist.barrier()

    trained_weights = {k: weights[k].detach().clone() for k in weights}

    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    if rank == 0:
        print(f"Trained ORE weights successfully computed for {list(weights.keys())}")

    if is_ddp:
        dist.barrier()
        print(f"Rank {rank}: Training complete, returning to main")

    return trained_weights, tokenized_requests


def apply_ORE_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: OREHyperParams,
        copy=False,
        return_orig_weights=False,
        **kwargs: Any,
):
    random.seed(42)

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    current_device = model.device

    if isinstance(model, ExternalRepHeadModel):
        wrapped = model
    else:
        print("Not ExternalRepHeadModel. Creating..")
        for p in model.parameters():
            p.requires_grad = False
        model.eval()


        wrapped = ExternalRepHeadModel(
            backbone=model,
            layers=hparams.unfreeze_layers,
            proj_dim=hparams.proj_dim,
            use_ln=True,
            use_rep_heads=True,
            device=current_device,
            intervention_strategy="all"
        ).to(torch.bfloat16)

    if dist.is_initialized():
        wrapped = DDP(
            wrapped,
            device_ids=[current_device],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    cache_dir = Path(f"./data/stats/{hparams.model_name}/unrelated_subspaces/")
    cache_file_name = (
        f"ore_unrelated_subspaces"
        f"_layers{'_'.join(map(str, hparams.unfreeze_layers))}"
        f"_rank{hparams.unrel_rank}"
        f"_method{hparams.orth_method}.pt"
    )
    cache_path = cache_dir / cache_file_name

    device = wrapped.device

    if is_main_process():
        if not cache_path.exists():
            print(f"First run: Calculating unrelated subspaces. Will save to {cache_path}")
            cache_dir.mkdir(parents=True, exist_ok=True)

            global UNRELATED_CORPUS
            with open(r'./ORE/unrelated_prompts.json', 'r', encoding='utf-8') as f:
                UNRELATED_CORPUS = json.load(f)

            unrelated_prompts = random.sample(UNRELATED_CORPUS, 2000)

            r = hparams.unrel_rank
            cal_batch_size = 128
            num_prompts = len(unrelated_prompts)
            target_layers = list(hparams.unfreeze_layers)
            reduced_states = {l: [] for l in target_layers}

            print(f"Calculating unrelated subspaces with rank {r}...")
            print(f"Total prompts: {num_prompts}, Batch size: {cal_batch_size}")
            num_batches = -(-num_prompts // cal_batch_size)

            with torch.no_grad():
                with _rep_heads(wrapped, False):
                    for i in range(0, num_prompts, cal_batch_size):
                        print(f"  Processing batch {i // cal_batch_size + 1} / {num_batches}...")
                        chunk_prompts = unrelated_prompts[i: i + cal_batch_size]

                        unrelated_inputs_chunk = tok(
                            chunk_prompts,
                            padding=True,
                            truncation=True,
                            max_length=32,
                            return_tensors="pt",
                            add_special_tokens=True,
                        )
                        unrelated_inputs_chunk = {k: v.to(device) for k, v in unrelated_inputs_chunk.items()}

                        am = unrelated_inputs_chunk["attention_mask"]
                        last_idx = (am.sum(dim=1) - 1).clamp(min=0).long()

                        chunk_states = wrapped(**unrelated_inputs_chunk, output_hidden_states=True).hidden_states

                        for l in target_layers:
                            hs = chunk_states[l]
                            if hparams.orth_method == 'mean':
                                m = am.unsqueeze(-1).to(hs.dtype)
                                reduced = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
                            else:
                                reduced = hs[torch.arange(hs.size(0), device=hs.device), last_idx, :]
                            reduced_states[l].append(reduced.float().cpu())

                        del chunk_states, unrelated_inputs_chunk
            empty_cache()

            print("Concatenating all batches...")
            all_unrelated_subspaces = {}
            for target_layer in target_layers:
                layer_hidden_states = torch.cat(reduced_states[target_layer], dim=0).to(device)
                with torch.no_grad():
                    U = compute_unrel_subspace(layer_hidden_states, r=r)
                all_unrelated_subspaces[target_layer] = U
                del layer_hidden_states
            del reduced_states
            empty_cache()

            print(f"Saving computed subspaces to {cache_path}")
            torch.save(all_unrelated_subspaces, cache_path)

            print("Unrelated subspaces computed and cached.")

    if dist.is_initialized():
        dist.barrier()

    print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Loading subspaces...")
    all_unrelated_subspaces = torch.load(cache_path, map_location=wrapped.device)

    adapter_weights, tokenized_requests = execute_ore(
        wrapped,
        tok,
        requests,
        hparams,
        all_unrelated_subspaces,
        **kwargs
    )

    inner_model = wrapped.module if hasattr(wrapped, "module") else wrapped

    current_adapter_params = {n: p for n, p in inner_model.rep_adapters.named_parameters()}

    with torch.no_grad():
        for w_name, final_weights in adapter_weights.items():
            w = current_adapter_params[w_name]

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w.copy_(final_weights)

    return (wrapped.module if hasattr(wrapped, "module") else wrapped), weights_copy


def get_context_templates(model, tok, prefix):
    bos_token = tok.bos_token if tok.bos_token else "<|begin_of_text|>"
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    try:
        with _rep_heads(model, False):
            templates = [["{}"]] + [
                [
                    f.replace(bos_token, "").replace("{", " ").replace("}", " ") + ". {}"
                    for f in generate_fast(
                        model,
                        tok,
                        prefix,
                        n_gen_per_prompt=n_gen // len(prefix),
                        max_out_len=length,
                    )
                ]
                for length, n_gen in [(10, len(prefix))]
            ]
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print(f"Cached context templates {templates}")
    return templates