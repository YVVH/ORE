import os
import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
from datetime import timedelta
import gc

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=1800)
        )
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        print(f"Initialized DDP: Rank {rank}/{world_size}, Local Rank {local_rank}, Device {device}")
        return True, rank, local_rank, world_size, device
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return False, 0, 0, 1, device


is_ddp, rank, local_rank, world_size, device = setup_ddp()

from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    MQUAKEDataset,
    get_tfidf_vectorizer,
    KnownsDataset,
    BZsREDataset,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_mquake import compute_rewrite_quality_mquake
from util.globals import *
from ORE import OREHyperParams
from ORE.ORE_main import apply_ORE_to_model, build_loc_pool

ALG_DICT = {
    "ORE": (OREHyperParams, apply_ORE_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "mquake": (MQUAKEDataset, compute_rewrite_quality_mquake),
    "bzsre": (BZsREDataset, compute_rewrite_quality_zsre),
}


def main(
        alg_name: str,
        model_name: Union[str, Tuple],
        hparams_fname: str,
        ds_name: str,
        dataset_size_limit: int,
        continue_from_run: str,
        skip_generation_tests: bool,
        generation_test_interval: int,
        conserve_memory: bool,
        dir_name: str,
        num_edits: int = 1,
        use_cache: bool = False,
):
    params_class, apply_algo = ALG_DICT[alg_name]

    global is_ddp

    alg_dir = RESULTS_DIR / dir_name
    run_id = 0

    if rank == 0:
        if continue_from_run is not None:
            if not (alg_dir / continue_from_run).exists():
                continue_from_run = None

        if continue_from_run is None:
            if alg_dir.exists():
                id_list = [
                    int(str(x).split("_")[-1])
                    for x in alg_dir.iterdir()
                    if str(x).split("_")[-1].isnumeric()
                ]
                run_id = 0 if not id_list else max(id_list) + 1
            else:
                run_id = 0
        else:
            parts = str(continue_from_run).split("_")
            if len(parts) > 1 and parts[-1].isnumeric():
                run_id = int(parts[-1])
            else:
                run_id = 0

    if is_ddp:
        run_id_tensor = torch.tensor([run_id], dtype=torch.long, device=device)
        dist.broadcast(run_id_tensor, src=0)
        run_id = run_id_tensor.item()

    run_dir = alg_dir / f"run_{str(run_id).zfill(3)}"

    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be stored at {run_dir}")

    if is_ddp:
        dist.barrier()

    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )

    if rank == 0:
        if not (run_dir / "params.json").exists():
            shutil.copyfile(params_path, run_dir / "params.json")

    if is_ddp:
        dist.barrier()

    hparams = params_class.from_json(params_path)

    if rank == 0:
        print(f"Executing {alg_name} with parameters {hparams}")

    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    loc_pool = (
        build_loc_pool(ds_class, DATA_DIR, tok, dataset_size_limit)
        if alg_name == "ORE" else None
    )

    cnt = 0

    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        print(
            f"=================================================================={cnt + 1}_edit==================================================================")

        already_finished = True
        for record in record_chunks:
            if not Path(case_result_template.format(num_edits, record["case_id"])).exists():
                already_finished = False
                break
        if already_finished:
            continue

        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=device)
            if conserve_memory
            else dict()
        )
        start = time()
        tok.padding_side = "right"
        edited_model, _ = apply_algo(
            model,
            tok,
            [
                {"case_id": record["case_id"], **rewrite_dict}
                for record in record_chunks
                for rewrite_dict in (
                record["requested_rewrite"]
                if isinstance(record["requested_rewrite"], list)
                else [record["requested_rewrite"]]
            )
            ],
            hparams,
            return_orig_weights=False,
            loc_pool=loc_pool,
            **args_conserve_memory,
        )
        model = edited_model
        exec_time = time() - start
        cnt += 1
        print("Execution took", exec_time)

    if is_ddp:
        dist.barrier()
        print(f"Rank {rank}: Editing phase complete. Starting parallel evaluation.")

    if rank == 0:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab")
    if is_ddp:
        dist.barrier()

    if hasattr(edited_model, 'module'):
        eval_model = edited_model.module
    else:
        eval_model = edited_model

    eval_model.eval()

    start_time = time()
    gen_test_vars = [snips, vec]

    processed_count = 0

    from tqdm import tqdm
    my_total = len(range(rank, len(ds), world_size))
    pbar = tqdm(total=my_total, desc="Evaluating", disable=(rank != 0), dynamic_ncols=True)

    for i, record in enumerate(ds):
        if i % world_size != rank:
            continue

        processed_count += 1
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))

        if out_file.exists():
            pbar.update(1)
            continue

        metrics = {
            "ds_name": ds_name,
            "case_id": record["case_id"],
            "grouped_case_ids": case_ids,
            "num_edits": num_edits,
            "requested_rewrite": record["requested_rewrite"],
            "time": exec_time,
            "post": ds_eval_method(
                eval_model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),
            ),
        }

        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)
        pbar.update(1)

    pbar.close()
    print(f"Rank {rank}: Finished processing {processed_count} records in {time() - start_time:.2f}s")

    if is_ddp:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier()
        finally:
            dist.destroy_process_group()

    print(f"Rank {rank}: Process exiting naturally.")
    return


def window(seq, n=2):
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i: i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ORE"],
        default="ORE",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
             "where a new run_id is generated on each run. "
             "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "mquake", "bzsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
             "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
             "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
    )