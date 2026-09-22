"""Multi-task model pieces shared by the baseline, the main model and the ablation.

Head A: shared representation -> 11-way event-type logits.
Head B: [shared representation ; softmax(Head A)] -> 1 hurt logit. The soft probability
vector (not an argmax) is concatenated, and gradients flow through it end to end.
Ablation: Head B sees the shared representation only (use_event_probs=False).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline import N_EVENT_CLASSES, UNLABELED


def _head(in_dim, out_dim, hidden, dropout):
    if hidden:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(),
                             nn.Dropout(dropout), nn.Linear(hidden, out_dim))
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, out_dim))


class MultiTaskHeads(nn.Module):
    def __init__(self, in_dim, use_event_probs=True, hidden=0, dropout=0.1):
        super().__init__()
        self.use_event_probs = use_event_probs
        self.head_a = _head(in_dim, N_EVENT_CLASSES, hidden, dropout)
        self.head_b = _head(in_dim + (N_EVENT_CLASSES if use_event_probs else 0), 1, hidden, dropout)

    def forward(self, h, event_probs_override=None):
        logits_a = self.head_a(h)
        if self.use_event_probs:
            probs_a = F.softmax(logits_a, dim=-1) if event_probs_override is None else event_probs_override
            logit_b = self.head_b(torch.cat([h, probs_a], dim=-1)).squeeze(-1)
        else:
            logit_b = self.head_b(h).squeeze(-1)
        return logits_a, logit_b


class EmbeddingMultiTask(nn.Module):
    """Baseline: frozen sentence embeddings in, heads on top."""

    def __init__(self, in_dim, use_event_probs=True, hidden=256, dropout=0.2):
        super().__init__()
        self.heads = MultiTaskHeads(in_dim, use_event_probs, hidden, dropout)

    def forward(self, emb, event_probs_override=None):
        return self.heads(emb, event_probs_override)


class EncoderMultiTask(nn.Module):
    """Main model: fine-tuned transformer encoder (CLS token) + linear heads.

    If the encoder is a Longformer, the CLS token also gets global attention.
    """

    def __init__(self, model_name, use_event_probs=True, dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        self.is_longformer = "longformer" in self.encoder.config.model_type
        self.heads = MultiTaskHeads(self.encoder.config.hidden_size, use_event_probs, hidden=0, dropout=dropout)

    def forward(self, input_ids, attention_mask, event_probs_override=None):
        kwargs = {}
        if self.is_longformer:
            g = torch.zeros_like(input_ids)
            g[:, 0] = 1
            kwargs["global_attention_mask"] = g
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state[:, 0]
        return self.heads(h, event_probs_override)


def multitask_loss(logits_a, logit_b, y_event, y_harm, w_a=1.0, w_b=1.0):
    """w_a * CE(event) + w_b * BCE(harm), each over the rows that have that label."""
    zero = logit_b.sum() * 0.0
    ma = y_event != UNLABELED
    mb = y_harm != UNLABELED
    ce = F.cross_entropy(logits_a[ma], y_event[ma]) if ma.any() else zero
    bce = F.binary_cross_entropy_with_logits(logit_b[mb], y_harm[mb].float()) if mb.any() else zero
    return w_a * ce + w_b * bce, ce.detach(), bce.detach()


@torch.no_grad()
def predict(model, batches, device, event_prior=None):
    """Run the model over `batches` (iterable of (inputs_dict, y_event, y_harm)).

    Returns probs_a (N, 11), p_hurt (N,), and, if Head B uses Head A and `event_prior`
    is given, p_hurt_prior: Head B's output with Head A's vector replaced by the train
    prior -- used by the event-type-reliance guardrail to see whether Head B relies on Head A.
    """
    model.eval()
    probs_a, p_hurt, p_prior = [], [], []
    uses_a = model.heads.use_event_probs if hasattr(model, "heads") else False
    for inputs, _, _ in batches:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits_a, logit_b = model(**inputs)
        probs_a.append(F.softmax(logits_a, -1).float().cpu())
        p_hurt.append(torch.sigmoid(logit_b).float().cpu())
        if uses_a and event_prior is not None:
            prior = torch.as_tensor(event_prior, dtype=logits_a.dtype, device=device).expand_as(logits_a)
            _, lb = model(**inputs, event_probs_override=prior)
            p_prior.append(torch.sigmoid(lb).float().cpu())
    out = {"probs_a": torch.cat(probs_a).numpy(), "p_hurt": torch.cat(p_hurt).numpy()}
    if p_prior:
        out["p_hurt_prior"] = torch.cat(p_prior).numpy()
    return out


def linear_warmup_decay(optimizer, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def event_prior_from_labels(y_event):
    y = np.asarray(y_event)
    y = y[y != UNLABELED]
    return np.bincount(y, minlength=N_EVENT_CLASSES) / len(y)


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
