
import typing
from itertools import chain
import os

import numpy as np
import torch

from util.device import DEVICE as device

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets


def compute_rewrite_quality_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:

    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    target_tok = tok(" " + target_new["str"])["input_ids"]
    if 'llama' in model.config._name_or_path.lower():
        target_tok = target_tok[1:]
    inp_prompts_og = list(chain(*prob_prompts))
    inp_prompts = [
        el + tok.decode(target_tok[:i])
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]
    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]
    inp_boundaries = [
        len(tok(el)["input_ids"])
        for el in inp_prompts_og
        for _ in range(len(target_tok))
    ]

    stuff_probs = test_batch_prediction_acc(
        model, tok, inp_prompts, inp_targets, boundaries=inp_boundaries
    )

    neighborhood_correct = test_batch_prediction_acc(
        model,
        tok,
        [
            el["prompt"].format(record["requested_rewrite"])
            for el in neighborhood_prompts
        ],
        [el["target"] for el in neighborhood_prompts],
    )

    probs = stuff_probs + neighborhood_correct

    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret["neighborhood_prompts_correct"] = neighborhood_correct

    return ret


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target, boundaries=None):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    inner = getattr(model, "module", model)
    set_b = getattr(inner, "set_intervention_boundary", None)
    if boundaries is not None:
        boundary = torch.as_tensor(boundaries, device=model.device).long()
    else:
        boundary = prompt_tok["attention_mask"].sum(dim=1).long()

    with torch.no_grad():
        if set_b is not None:
            set_b(boundary)
        try:
            logits = model(**prompt_tok).logits
        finally:
            if set_b is not None:
                set_b(None)
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt").to(model.device)[
            "input_ids"
        ]
        if 'llama' in model.config._name_or_path.lower():
            correct_id = correct_id[:, 1].squeeze()
        else:
            correct_id = correct_id[:, 0].squeeze()

        return (ans == correct_id).detach().cpu().numpy().tolist()
