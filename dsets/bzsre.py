import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre_mend_eval.json"


class BZsREDataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "zsre_mend_eval.json"
        zsre_loc_zh = data_dir / "zsre_mend_eval_chinese.json"

        with open(zsre_loc, "r", encoding="utf-8") as f:
            raw_en = json.load(f)

        with open(zsre_loc_zh, "r", encoding="utf-8") as f:
            raw_zh = json.load(f)

        if size is None:
            src_en = raw_en
            src_zh = raw_zh
        else:
            half_size = size // 2

            src_en = raw_en[:half_size]
            src_zh = raw_zh[:size - half_size]

        raw = []
        chunk_size = 100
        max_len = max(len(src_en), len(src_zh))

        for i in range(0, max_len, chunk_size):
            raw.extend(src_en[i: i + chunk_size])
            raw.extend(src_zh[i: i + chunk_size])

        data = []
        for i, record in enumerate(raw):
            ans_toks = tok(" " + record["loc_ans"])["input_ids"]
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        "target_new": {"str": record["answers"][0]},
                        "target_true": {"str": "<|endoftext|>"},
                        "pred_statement": record.get("gold_statement", ""),
                        "alt_statement": record.get("alt_statement", ""),
                        "loc": record["loc"] + "?",
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )

        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
