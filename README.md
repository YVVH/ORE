# ORE: Orthogonal Representation Editing

Official code for "Orthogonal Representation Editing: Decoupling Semantic
Entanglement in Batch Knowledge Editing of LLMs" (Findings of ACL 2026). This code was ported from Ascend NPU to NVIDIA CUDA.

## Installation

```bash
conda create -n ore python=3.12 -y && conda activate ore
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
 
## Quick Start

Edit Qwen2.5-7B with 100 edits on ZsRE.

Single GPU:

```bash
python -m experiments.evaluate \
    --alg_name=ORE --model_name=Qwen/Qwen2.5-7B-Instruct \
    --hparams_fname=Qwen2.5-7B.json --ds_name=zsre \
    --dataset_size_limit=100 --num_edits=100
```

Multi-GPU:

```bash
torchrun --nproc_per_node=2 -m experiments.evaluate \
    --alg_name=ORE --model_name=Qwen/Qwen2.5-7B-Instruct \
    --hparams_fname=Qwen2.5-7B.json --ds_name=zsre \
    --dataset_size_limit=100 --num_edits=100
```

For cross-lingual editing:

```bash
python -m experiments.evaluate \
    --alg_name=ORE --model_name=Qwen/Qwen2.5-7B-Instruct \
    --hparams_fname=Qwen2.5-7B.json --ds_name=bzsre \
    --dataset_size_limit=100 --num_edits=100
```

Results are written to `results/<alg_name>/run_<id>/`. Summarize with:

```bash
python -m experiments.summarize --dir_name=ORE --runs=run_<id>
```

## Citation

## Acknowledgments

This codebase builds on [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit), and
[MEMIT](https://github.com/kmeng01/memit).
