from typing import Literal
from transformers import AutoTokenizer
from .ORE_hparams import OREHyperParams
from .ORE_rep_head import ExternalRepHeadModel
from .compute_utils import *
import torch
import torch.nn.functional as F


def orth_loss(
        hparams: OREHyperParams,
        src_origin_inner,
        alt_statement_inner,
        pred_statement_inner,
        src_inner_cache,
        all_unrelated_subspaces,
        attention_mask,
        method: Literal["last_token", "mean"] = "last_token"
):
    losses = []
    device = src_inner_cache[hparams.unfreeze_layers[0]].device

    valid_lengths = attention_mask.sum(dim=1) - 1
    valid_lengths = valid_lengths.clamp(min=0).long().to(device)
    gather_idx = valid_lengths.view(-1, 1, 1)

    for target_layer in hparams.unfreeze_layers:
        hidden_dim = src_origin_inner[target_layer].shape[-1]
        gather_idx_expanded = gather_idx.expand(-1, 1, hidden_dim)


        if method == 'last_token':
            def get_last(tensor_seq):
                return tensor_seq.gather(1, gather_idx_expanded).squeeze(1)
            alt_statement_tensor = get_last(alt_statement_inner[target_layer])
            pred_statement_tensor = get_last(pred_statement_inner[target_layer])
            src_origin_tensor = get_last(src_origin_inner[target_layer])
            src_tensor = get_last(src_inner_cache[target_layer])
        elif method == 'mean':
            mask = attention_mask.unsqueeze(-1).to(device)
            def get_mean(tensor_seq):
                sum_val = (tensor_seq * mask).sum(dim=1)
                count_val = mask.sum(dim=1).clamp(min=1e-9)
                return sum_val / count_val
            alt_statement_tensor = get_mean(alt_statement_inner[target_layer])
            pred_statement_tensor = get_mean(pred_statement_inner[target_layer])
            src_origin_tensor = get_mean(src_origin_inner[target_layer])
            src_tensor = get_mean(src_inner_cache[target_layer])

        U_unrel = all_unrelated_subspaces[target_layer].to(device)
        delta = alt_statement_tensor - pred_statement_tensor
        delta = remove_subspace(delta, U_unrel)
        target_tensor = (src_origin_tensor.detach() + delta).detach()
        src_n = F.normalize(src_tensor, dim=-1)
        pos = 1 - F.cosine_similarity(src_n, F.normalize(target_tensor, dim=-1), dim=-1).mean()
        loss = pos
        losses.append(loss)

    loss = torch.stack(losses).mean()

    del alt_statement_tensor, pred_statement_tensor, src_origin_tensor
    del src_tensor, delta, target_tensor, src_n

    return loss


def kl_loss(
        teacher_logits,
        student_logits,
        prompt_mask
):
    if not prompt_mask.any():
        return student_logits.new_zeros(())

    s = student_logits[prompt_mask].float()
    t = teacher_logits[prompt_mask].float()
    log_p = F.log_softmax(s, dim=-1)
    log_q = F.log_softmax(t, dim=-1)
    loss_kl = F.kl_div(log_p, log_q.detach(), reduction='batchmean', log_target=True)
    return loss_kl.to(student_logits.dtype)


def ce_loss(
        out_ce,
        tok: AutoTokenizer,
        prompt_len,
        ans_input_ids,
        use_margin: bool = False,
        margin: float = 1.5,
        lambda_margin: float = 0.2
):
    loss = out_ce.loss

    if use_margin:
        B = ans_input_ids.size(0)
        pos_idx = (prompt_len - 1).clamp_min(0)
        logits_first = out_ce.logits[torch.arange(B), pos_idx, :]
        gold_first = ans_input_ids[:, 0]
        gold_logit = logits_first.gather(1, gold_first.unsqueeze(1)).squeeze(1)
        one_hot = F.one_hot(gold_first, num_classes=logits_first.size(-1)).bool()
        runnerup_logit = logits_first.masked_fill(one_hot, float('-inf')).amax(dim=-1)
        gap = gold_logit - runnerup_logit
        loss_margin = F.relu(margin - gap).mean()
        loss = loss + lambda_margin * loss_margin
    return loss


def gate_bce_loss(
        gate_scores_list: list[torch.Tensor],
        subj_ranges: list[list[int]],
        attention_mask: torch.Tensor,
        is_unrelated: bool = False,
        pos_weight: float = 1.0
):
    total_loss = 0.0
    count = 0
    epsilon = 1e-6

    for w in gate_scores_list:
        if w is None: continue

        preds = w.squeeze(-1)
        device = preds.device
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        targets = torch.zeros_like(preds)
        if not is_unrelated and subj_ranges is not None:
            B, S = preds.shape
            for b in range(B):
                if subj_ranges[b] is not None:
                    start, end = subj_ranges[b]
                    end = min(end, S)
                    if start < end:
                        targets[b, start:end] = 1.0

        preds = preds.to(torch.float32)
        targets = targets.to(torch.float32)
        weight_matrix = torch.ones_like(preds)

        weight_matrix[targets == 1.0] = pos_weight

        preds_clamped = torch.clamp(preds, epsilon, 1.0 - epsilon)

        loss = F.binary_cross_entropy(preds_clamped, targets, weight=weight_matrix, reduction='none')

        masked_loss = loss * attention_mask.to(torch.float32)

        layer_loss = masked_loss.sum() / (attention_mask.sum() + epsilon)

        total_loss += layer_loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=gate_scores_list[0].device, dtype=torch.float32)

    return total_loss / count