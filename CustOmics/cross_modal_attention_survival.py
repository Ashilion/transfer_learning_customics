"""
Cross-modal attention model for multi-omics survival analysis.

This is a standalone replacement for the CustOmics fusion step used in the
nested-CV evaluation pipeline. It does NOT mimic the CustOmics API
(fit/get_latent_representation/plot_loss_detailed with the exact same
signatures) -- it exposes a small, explicit surface so you can wire it into
your own eval script (ncv_eval_cross_modal_attention.py or similar) however
you like:

    model = CrossModalAttentionSurvivalModel(
        input_dims={"rna": 5000, "cnv": 5000, "meth": 5000},
        latent_dim=32,
        fusion_type="pairwise",   # or "cls_token"
    )
    history = model.fit(
        omics_train=omics_train_outer,        # dict[str, np.ndarray] (n_samples, n_features)
        present_mask=present_mask_train,       # np.ndarray (n_samples, n_modalities) bool, True = present
        clinical_df=clinical_df, samples=samples_train_outer,
        event="status", surv_time="time",
        n_epochs=300, patience=15, lambda_surv=5.0, switch_epoch=100,
    )
    Z_train = model.get_latent_representation(omics_train_outer, present_mask_train)
    # -> feed Z_train into fit_coxnet(...) exactly as before.

Two fusion strategies are implemented and selectable at construction time so
both can be benchmarked against each other on the same folds:

  * "pairwise"  : every modality queries every OTHER modality directly
                  (true modality-to-modality cross-attention: query = one
                  modality, key/value = the rest). The per-modality
                  contextualised embeddings are then mean-pooled over the
                  PRESENT modalities into one patient vector.
  * "cls_token" : a single learned fusion token attends over the stacked
                  modality embeddings (Perceiver/CLS-style layer). The
                  updated token is the patient representation.

Missing modalities are handled natively inside the attention (key/value
masking via `present_mask`), not by relying on zero-imputation upstream:
a missing modality never contributes as a key/value to anyone else's
context, and it is excluded from the final pooling in "pairwise" fusion.

Two OPTIONAL contrastive losses, each independently switchable via its own
lambda (0 = off) so you can run clean ablations:

  * lambda_contrastive_modality : aligns, in the PRE-FUSION latent space,
    the per-modality embeddings that belong to the SAME patient (bidirectional
    InfoNCE / CLIP-style, computed per modality pair, in-batch negatives).
    Samples missing one of the two modalities of a pair are simply excluded
    from that pair's loss term. This is meant to make the fused
    representation more robust to missing modalities at test time.

  * lambda_contrastive_clinical : a soft nearest-neighbour loss on the
    POST-FUSION latent (`fused`), pulling patients that are clinically
    similar closer together. Two modes:
      - "stage": positives = same clinical stage in the batch (hard labels).
      - "knn"  : positives = the k nearest neighbours in a (continuous)
                 clinical feature space, soft-weighted by proximity.
    Use this one with caution -- see the discussion in the accompanying
    conversation: it risks (a) duplicating supervision already provided by
    lambda_surv + the downstream concat of raw clinical features, and (b)
    pulling the omics representation toward being a proxy for stage, which
    can undercut the very point of showing omics/TL adds value beyond the
    clinical baseline. Treat it as an ablation, not a default-on feature.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

def _mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int, dropout: float = 0.2) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class PairwiseCrossAttentionFusion(nn.Module):
    """Each modality queries every OTHER modality (self excluded), then the
    per-modality contextualised embeddings are masked-mean-pooled over the
    present modalities."""

    def __init__(self, latent_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, embeddings: torch.Tensor, present_mask: torch.Tensor):
        # embeddings: [B, M, D] ; present_mask: [B, M] bool, True = present
        B, M, D = embeddings.shape
        base_key_padding = ~present_mask  # True = ignore this key/value

        contexts = []
        for m in range(M):
            query = embeddings[:, m:m + 1, :]  # [B, 1, D]
            key_padding_mask = base_key_padding.clone()
            key_padding_mask[:, m] = True  # exclude self from keys/values

            # guard against rows with no valid key at all (would produce NaN
            # softmax) -- e.g. a patient with only this one modality present.
            # In that degenerate case, fall back to attending to itself.
            no_valid_key = key_padding_mask.all(dim=1)
            if no_valid_key.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[no_valid_key, m] = False

            out, _ = self.attn(query, embeddings, embeddings,
                                key_padding_mask=key_padding_mask, need_weights=False)
            contexts.append(out)  # [B, 1, D]

        contexts = torch.cat(contexts, dim=1)  # [B, M, D]
        contexts = self.norm(contexts + embeddings)

        mask_f = present_mask.unsqueeze(-1).float()  # [B, M, 1]
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        fused = (contexts * mask_f).sum(dim=1) / denom  # [B, D]
        return fused, contexts  # contexts returned for inspection/debugging


class CLSTokenFusion(nn.Module):
    """A single learned fusion token attends over the stacked modality
    embeddings (key/value-masked for missing modalities)."""

    def __init__(self, latent_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, embeddings: torch.Tensor, present_mask: torch.Tensor):
        B, M, D = embeddings.shape
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        key_padding_mask = ~present_mask  # [B, M]

        no_valid_key = key_padding_mask.all(dim=1)
        if no_valid_key.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[no_valid_key, :] = False  # degenerate: attend to everything

        fused, attn_weights = self.attn(cls, embeddings, embeddings,
                                         key_padding_mask=key_padding_mask,
                                         need_weights=True, average_attn_weights=True)
        fused = self.norm(fused.squeeze(1) + cls.squeeze(1))
        return fused, attn_weights.squeeze(1)  # [B, M] per-modality attention, for interpretability


# --------------------------------------------------------------------------- #
# Contrastive-loss helpers
# --------------------------------------------------------------------------- #

def build_stage_affinity(stage_labels: torch.Tensor) -> torch.Tensor:
    """stage_labels: LongTensor [B] of (encoded) clinical stage. Returns a
    [B, B] 0/1 affinity matrix, 1 where two samples share the same stage,
    diagonal zeroed out."""
    same = (stage_labels.unsqueeze(0) == stage_labels.unsqueeze(1)).float()
    same.fill_diagonal_(0.0)
    return same


def build_knn_affinity(clinical_features: torch.Tensor, k: int = 10, temperature: float = 1.0) -> torch.Tensor:
    """clinical_features: FloatTensor [B, C] (should already be standardised
    upstream, e.g. with the same scaler used elsewhere in the pipeline).
    Returns a [B, B] soft affinity matrix: for each row, a softmax over the
    k nearest neighbours' negative distance (temperature-scaled), zero
    elsewhere. Diagonal is zero (no self-affinity)."""
    B = clinical_features.shape[0]
    dist = torch.cdist(clinical_features, clinical_features, p=2)  # [B, B]
    dist = dist + torch.eye(B, device=clinical_features.device) * 1e6  # exclude self
    k_eff = max(1, min(k, B - 1))
    neg_dist, idx = torch.topk(-dist, k=k_eff, dim=-1)  # closest = largest -dist
    weights = F.softmax(neg_dist / max(temperature, 1e-6), dim=-1)
    affinity = torch.zeros(B, B, device=clinical_features.device)
    affinity.scatter_(1, idx, weights)
    return affinity


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #

class CrossModalAttentionSurvivalModel(nn.Module):
    def __init__(self, input_dims: Dict[str, int], latent_dim: int = 32,
                 modality_hidden_dims: Optional[Dict[str, Sequence[int]]] = None,
                 fusion_type: str = "pairwise", n_heads: int = 4, dropout: float = 0.2,
                 device: Optional[torch.device] = None):
        super().__init__()
        assert fusion_type in ("pairwise", "cls_token"), fusion_type
        self.sources = list(input_dims.keys())
        self.latent_dim = latent_dim
        self.fusion_type = fusion_type
        self.device = device or torch.device("cpu")

        modality_hidden_dims = modality_hidden_dims or {s: [256, 64] for s in self.sources}

        self.encoders = nn.ModuleDict({
            s: _mlp(input_dims[s], modality_hidden_dims[s], latent_dim, dropout=dropout)
            for s in self.sources
        })
        self.decoders = nn.ModuleDict({
            s: _mlp(latent_dim, list(reversed(list(modality_hidden_dims[s]))), input_dims[s], dropout=dropout)
            for s in self.sources
        })

        self.fusion = (PairwiseCrossAttentionFusion(latent_dim, n_heads, dropout)
                       if fusion_type == "pairwise" else
                       CLSTokenFusion(latent_dim, n_heads, dropout))

        self.risk_head = nn.Linear(latent_dim, 1, bias=False)

        self.history = {"total": [], "recon": [], "surv": [],
                         "contrastive_modality": [], "contrastive_clinical": []}
        self._phase1_epochs = None  # set by fit() when two_phase_training=True, used by the plots
        self.to(self.device)

    # ---- forward pieces ---------------------------------------------------

    def encode_modalities(self, omics_batch: Dict[str, torch.Tensor], present_mask: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for i, s in enumerate(self.sources):
            z = self.encoders[s](omics_batch[s])
            z = z * present_mask[:, i:i + 1].float()  # zero out missing modalities' embeddings
            embeddings.append(z)
        return torch.stack(embeddings, dim=1)  # [B, M, D]

    def forward(self, omics_batch: Dict[str, torch.Tensor], present_mask: torch.Tensor):
        embeddings = self.encode_modalities(omics_batch, present_mask)
        fused, attn_info = self.fusion(embeddings, present_mask)
        reconstructions = {s: self.decoders[s](fused) for s in self.sources}
        risk = self.risk_head(fused).squeeze(-1)
        return embeddings, fused, reconstructions, risk, attn_info

    def forward_autoencoder_only(self, omics_batch: Dict[str, torch.Tensor], present_mask: torch.Tensor):
        """Phase-1 forward pass: each modality is encoded and decoded on its
        own (encoder_s -> embedding_s -> decoder_s -> reconstruction_s), with
        NO cross-attention fusion and no risk head involved -- true
        per-modality autoencoders. Used only when two_phase_training=True."""
        embeddings = self.encode_modalities(omics_batch, present_mask)  # [B, M, D]
        reconstructions = {s: self.decoders[s](embeddings[:, i, :]) for i, s in enumerate(self.sources)}
        return embeddings, reconstructions

    @staticmethod
    def _masked_recon_loss(reconstructions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                            present_mask: torch.Tensor, sources: Sequence[str]) -> torch.Tensor:
        """Per-modality MSE, averaged over modalities, each term computed
        ONLY over the samples where that modality is actually present (so a
        zero-filled missing modality never teaches the decoder to output
        zero)."""
        terms = []
        for i, s in enumerate(sources):
            valid = present_mask[:, i]
            if valid.sum() == 0:
                continue
            terms.append(F.mse_loss(reconstructions[s][valid], targets[s][valid]))
        if not terms:
            return reconstructions[sources[0]].new_zeros(())
        return torch.stack(terms).mean()

    # ---- Cox partial likelihood (Breslow approximation) --------------------

    @staticmethod
    def _cox_partial_nll(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(time, descending=True)
        risk = risk[order]
        event = event[order]
        log_cumsum = torch.logcumsumexp(risk, dim=0)
        diff = risk - log_cumsum
        n_events = event.sum()
        if n_events == 0:
            return torch.zeros((), device=risk.device)
        return -(diff * event).sum() / n_events

    # ---- contrastive losses ------------------------------------------------

    @staticmethod
    def _modality_contrastive_loss(embeddings: torch.Tensor, present_mask: torch.Tensor,
                                    temperature: float = 0.1) -> torch.Tensor:
        """Bidirectional InfoNCE between every pair of modalities, computed
        only over samples where BOTH modalities of the pair are present.
        Positive = same patient index across the two modalities; negatives =
        other patients in the (present-for-both) subset of the batch."""
        B, M, D = embeddings.shape
        normed = F.normalize(embeddings, dim=-1)
        total_loss = embeddings.new_zeros(())
        n_pairs = 0
        for i in range(M):
            for j in range(i + 1, M):
                valid = present_mask[:, i] & present_mask[:, j]
                n_valid = int(valid.sum().item())
                if n_valid < 2:
                    continue  # need >=2 samples to have any negative
                zi = normed[valid, i, :]
                zj = normed[valid, j, :]
                logits = zi @ zj.T / temperature  # [n_valid, n_valid]
                labels = torch.arange(n_valid, device=embeddings.device)
                loss_ij = F.cross_entropy(logits, labels)
                loss_ji = F.cross_entropy(logits.T, labels)
                total_loss = total_loss + (loss_ij + loss_ji) / 2
                n_pairs += 1
        if n_pairs == 0:
            return embeddings.new_zeros(())
        return total_loss / n_pairs

    @staticmethod
    def _soft_nn_contrastive_loss(z: torch.Tensor, target_affinity: torch.Tensor,
                                   temperature: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
        """Soft-nearest-neighbour loss: pulls z_i towards a (weighted) set of
        "positive" targets defined by `target_affinity` (row-wise weights,
        diagonal expected to be 0), independently of any hard notion of
        modality. Rows with an all-zero target (no positive in the batch)
        are excluded from the mean rather than contributing a spurious loss."""
        B = z.shape[0]
        if B < 3:
            return z.new_zeros(())
        zn = F.normalize(z, dim=-1)
        logits = zn @ zn.T / temperature
        logits = logits.masked_fill(torch.eye(B, dtype=torch.bool, device=z.device), float("-inf"))
        log_probs = F.log_softmax(logits, dim=-1)

        row_sums = target_affinity.sum(dim=-1, keepdim=True)
        valid_rows = (row_sums.squeeze(-1) > eps)
        if valid_rows.sum() == 0:
            return z.new_zeros(())
        target = target_affinity / row_sums.clamp(min=eps)
        # target is 0 wherever log_probs is -inf (masked diagonal), but
        # 0 * -inf = nan in IEEE float, so guard the product explicitly
        # instead of relying on target zeroing it out.
        weighted = torch.where(target > 0, target * log_probs, torch.zeros_like(log_probs))
        loss_per_row = -weighted.sum(dim=-1)
        return loss_per_row[valid_rows].mean()

    # ---- training -----------------------------------------------------------

    def fit(self, omics_train: Dict[str, np.ndarray], present_mask: np.ndarray,
            clinical_df, samples: Sequence[str], event: str, surv_time: str,
            batch_size: int = 32, n_epochs: int = 300, lr: float = 1e-3, weight_decay: float = 0.0,
            lambda_recon: float = 1.0, lambda_surv: float = 0.0, switch_epoch: Optional[int] = None,
            patience: Optional[int] = None, min_delta: float = 1e-4, verbose: bool = True,
            # --- modality-alignment contrastive loss (pre-fusion) ---
            lambda_contrastive_modality: float = 0.0,
            contrastive_modality_temperature: float = 0.1,
            # --- clinical-similarity contrastive loss (post-fusion) ---
            lambda_contrastive_clinical: float = 0.0,
            clinical_contrastive_mode: str = "stage",  # "stage" or "knn"
            clinical_stage: Optional[np.ndarray] = None,       # required if mode == "stage"
            clinical_features: Optional[np.ndarray] = None,    # required if mode == "knn"
            clinical_knn_k: int = 10,
            clinical_contrastive_temperature: float = 0.5,
            clinical_knn_temperature: float = 1.0,
            # --- optional two-phase training: autoencoders alone, then everything else ---
            two_phase_training: bool = False,
            phase1_epochs: int = 100,
            ):
        """
        lambda_surv == 0            -> purely unsupervised (reconstruction only).
        lambda_surv > 0, switch_epoch=None -> survival loss on from epoch 0.
        lambda_surv > 0, switch_epoch=k    -> reconstruction-only warmup for
                                              k epochs, then reconstruction + survival
                                              (mirrors CustOmics' unsupervised/switch_epoch).
        NOTE: the Cox loss is computed per mini-batch (Breslow approx. over the
        batch's risk set), as is standard practice (Katzman et al., DeepSurv).
        Prefer batch_size large enough to contain a reasonable number of events.

        Both contrastive losses are OFF by default (lambda == 0) -- set the
        corresponding lambda to enable one or both for an ablation. They are
        applied every epoch (not gated by switch_epoch); if you want them to
        only kick in during a supervised phase, set switch_epoch and simply
        keep their lambdas at 0 until you call fit() again, or extend this
        loop to gate them the same way `use_surv` is gated.

        clinical_stage / clinical_features must be aligned 1:1 with `samples`
        (same order, same length) if the corresponding contrastive mode is used.

        two_phase_training (OFF by default): if True, training is split in two:
          - Phase 1 (epochs < phase1_epochs): TRUE per-modality autoencoders,
            trained independently of each other -- encoder_s -> embedding_s ->
            decoder_s -> reconstruction_s, with NO cross-attention fusion and
            no risk head involved (`forward_autoencoder_only`). Reconstruction
            for a given modality is computed only over the samples where that
            modality is actually present. `lambda_contrastive_modality`, if
            >0, also applies here (it's inherently a pre-fusion loss, so it
            fits naturally in this phase); lambda_surv and
            lambda_contrastive_clinical are ignored during phase 1 (they need
            `fused`/`risk`, which don't exist without the fusion step).
          - Phase 2 (epochs >= phase1_epochs): the usual joint training
            (`forward`), exactly as when two_phase_training=False, with the
            per-modality encoders/decoders now warm-started from phase 1
            rather than random init. A single Adam optimizer is used
            throughout; the fusion/risk_head parameters simply receive no
            gradient during phase 1 (never used in that phase's forward
            pass), so nothing needs to be frozen/unfrozen manually.
          If `switch_epoch` is left at None, it defaults to `phase1_epochs`
          so the survival loss turns on as soon as phase 2 starts; pass an
          explicit switch_epoch (>= phase1_epochs) if you want an extra
          reconstruction-only warmup within phase 2 itself.
        """
        assert clinical_contrastive_mode in ("stage", "knn")
        if two_phase_training and switch_epoch is None:
            switch_epoch = phase1_epochs
        self._phase1_epochs = phase1_epochs if two_phase_training else None

        n = len(samples)
        present_mask_t = torch.as_tensor(present_mask, dtype=torch.bool, device=self.device)
        omics_t = {s: torch.as_tensor(omics_train[s], dtype=torch.float32, device=self.device)
                   for s in self.sources}
        time_t = torch.as_tensor(clinical_df.loc[samples, surv_time].values.copy(),
                                  dtype=torch.float32, device=self.device)
        event_t = torch.as_tensor(clinical_df.loc[samples, event].values.copy(),
                                   dtype=torch.float32, device=self.device)

        stage_t = None
        if lambda_contrastive_clinical > 0 and clinical_contrastive_mode == "stage":
            if clinical_stage is None:
                raise ValueError("clinical_contrastive_mode='stage' requires clinical_stage.")
            stage_t = torch.as_tensor(np.asarray(clinical_stage), dtype=torch.long, device=self.device)

        clinical_feat_t = None
        if lambda_contrastive_clinical > 0 and clinical_contrastive_mode == "knn":
            if clinical_features is None:
                raise ValueError("clinical_contrastive_mode='knn' requires clinical_features.")
            clinical_feat_t = torch.as_tensor(np.asarray(clinical_features), dtype=torch.float32,
                                               device=self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            self.train()
            perm = torch.randperm(n, device=self.device)
            in_phase1 = two_phase_training and epoch < phase1_epochs
            if two_phase_training and epoch == phase1_epochs and verbose:
                print(f"  -- fin de la phase 1 (autoencodeurs seuls), epoch {epoch} : "
                      f"passage en phase 2 (fusion + reste) --")
            use_surv = (not in_phase1) and lambda_surv > 0 and (switch_epoch is None or epoch >= switch_epoch)

            epoch_losses = {"total": 0.0, "recon": 0.0, "surv": 0.0,
                             "contrastive_modality": 0.0, "contrastive_clinical": 0.0}
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                batch_omics = {s: omics_t[s][idx] for s in self.sources}
                batch_mask = present_mask_t[idx]

                if in_phase1:
                    embeddings, recon = self.forward_autoencoder_only(batch_omics, batch_mask)
                    recon_loss = self._masked_recon_loss(recon, batch_omics, batch_mask, self.sources)
                    loss = lambda_recon * recon_loss
                    surv_loss = embeddings.new_zeros(())
                    contrastive_clinical_loss = embeddings.new_zeros(())

                    contrastive_modality_loss = embeddings.new_zeros(())
                    if lambda_contrastive_modality > 0:
                        contrastive_modality_loss = self._modality_contrastive_loss(
                            embeddings, batch_mask, temperature=contrastive_modality_temperature)
                        loss = loss + lambda_contrastive_modality * contrastive_modality_loss

                else:
                    batch_use_surv = use_surv and event_t[idx].sum() > 0  # Cox loss needs >=1 event
                    embeddings, fused, recon, risk, _ = self.forward(batch_omics, batch_mask)

                    recon_loss = sum(F.mse_loss(recon[s], batch_omics[s]) for s in self.sources) / len(self.sources)
                    loss = lambda_recon * recon_loss

                    surv_loss = embeddings.new_zeros(())
                    if batch_use_surv:
                        surv_loss = self._cox_partial_nll(risk, time_t[idx], event_t[idx])
                        loss = loss + lambda_surv * surv_loss

                    contrastive_modality_loss = embeddings.new_zeros(())
                    if lambda_contrastive_modality > 0:
                        contrastive_modality_loss = self._modality_contrastive_loss(
                            embeddings, batch_mask, temperature=contrastive_modality_temperature)
                        loss = loss + lambda_contrastive_modality * contrastive_modality_loss

                    contrastive_clinical_loss = embeddings.new_zeros(())
                    if lambda_contrastive_clinical > 0:
                        if clinical_contrastive_mode == "stage":
                            affinity = build_stage_affinity(stage_t[idx])
                        else:
                            affinity = build_knn_affinity(clinical_feat_t[idx], k=clinical_knn_k,
                                                           temperature=clinical_knn_temperature)
                        contrastive_clinical_loss = self._soft_nn_contrastive_loss(
                            fused, affinity, temperature=clinical_contrastive_temperature)
                        loss = loss + lambda_contrastive_clinical * contrastive_clinical_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses["recon"] += recon_loss.item()
                epoch_losses["surv"] += surv_loss.item()
                epoch_losses["contrastive_modality"] += contrastive_modality_loss.item()
                epoch_losses["contrastive_clinical"] += contrastive_clinical_loss.item()
                epoch_losses["total"] += loss.item()
                n_batches += 1

            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)
                self.history[k].append(epoch_losses[k])

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"  epoch {epoch:4d} | total {epoch_losses['total']:.4f} | recon {epoch_losses['recon']:.4f} "
                      f"| surv {epoch_losses['surv']:.4f} | cma {epoch_losses['contrastive_modality']:.4f} "
                      f"| ccl {epoch_losses['contrastive_clinical']:.4f}")

            # Early stopping only applies within phase 2 when two_phase_training is on:
            # phase 1 always runs its full phase1_epochs (a plateau in pure
            # reconstruction is not a reason to skip fusion/survival training
            # entirely), and the moment phase 2 starts the loss composition
            # changes (new terms added), so the best/patience trackers are
            # reset then to compare like with like.
            if two_phase_training and epoch == phase1_epochs:
                best_loss = float("inf")
                epochs_no_improve = 0

            if patience is not None and not in_phase1:
                if epoch_losses["total"] < best_loss - min_delta:
                    best_loss = epoch_losses["total"]
                    best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if verbose:
                            print(f"  early stopping at epoch {epoch} (best loss {best_loss:.4f})")
                        break

        if patience is not None and best_state is not None:
            self.load_state_dict(best_state)

        return self.history

    # ---- inference ------------------------------------------------------

    @torch.no_grad()
    def get_latent_representation(self, omics: Dict[str, np.ndarray], present_mask: np.ndarray) -> np.ndarray:
        self.eval()
        omics_t = {s: torch.as_tensor(omics[s], dtype=torch.float32, device=self.device) for s in self.sources}
        mask_t = torch.as_tensor(present_mask, dtype=torch.bool, device=self.device)
        _, fused, _, _, _ = self.forward(omics_t, mask_t)
        return fused.cpu().numpy()

    @torch.no_grad()
    def get_fusion_attention(self, omics: Dict[str, np.ndarray], present_mask: np.ndarray):
        """Returns, for interpretability:
          - "pairwise": the [B, M, D] contextualised per-modality embeddings
            (post cross-attention, pre-pooling).
          - "cls_token": the [B, M] attention weights of the fusion token
            over each modality.
        """
        self.eval()
        omics_t = {s: torch.as_tensor(omics[s], dtype=torch.float32, device=self.device) for s in self.sources}
        mask_t = torch.as_tensor(present_mask, dtype=torch.bool, device=self.device)
        embeddings = self.encode_modalities(omics_t, mask_t)
        _, attn_info = self.fusion(embeddings, mask_t)
        return attn_info.cpu().numpy()

    # ---- diagnostics plots (mirrors CustOmics' plot_loss_detailed*) -----

    def plot_loss_detailed(self, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        keys = [k for k in ["total", "recon", "surv", "contrastive_modality", "contrastive_clinical"]
                if any(v != 0 for v in self.history[k]) or k in ("total", "recon")]
        titles = {"total": "Total loss", "recon": "Reconstruction", "surv": "Survival (Cox)",
                  "contrastive_modality": "Contrastive (modality alignment)",
                  "contrastive_clinical": "Contrastive (clinical similarity)"}
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
        if len(keys) == 1:
            axes = [axes]
        for ax, key in zip(axes, keys):
            ax.plot(self.history[key])
            ax.set_title(titles[key])
            ax.set_xlabel("epoch")
            if self._phase1_epochs is not None:
                ax.axvline(self._phase1_epochs, color="gray", linestyle="--", linewidth=1,
                           label="fin phase 1 (autoencodeurs)")
                ax.legend(fontsize=8)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig

    def plot_loss_detailed_stacked(self, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        components = ["recon", "surv", "contrastive_modality", "contrastive_clinical"]
        active = [k for k in components if any(v != 0 for v in self.history[k])] or ["recon"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.stackplot(range(len(self.history["recon"])),
                      *[self.history[k] for k in active], labels=active)
        if self._phase1_epochs is not None:
            ax.axvline(self._phase1_epochs, color="black", linestyle="--", linewidth=1,
                       label="fin phase 1 (autoencodeurs)")
        ax.legend(loc="upper right")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig


# --------------------------------------------------------------------------- #
# Helper: build a present_mask array from your existing missing-modality
# machinery (missing_data_load_all.apply_missing_modalities). Adapt the
# branch below to whatever exact structure `mask_train_outer` has in your
# pipeline -- this covers the two most common shapes.
# --------------------------------------------------------------------------- #

def present_mask_from_missing(samples: Sequence[str], sources: Sequence[str], missing) -> np.ndarray:
    """
    `missing` can be:
      - dict[sample_id -> set/list of missing source names], or
      - dict[sample_id -> single missing source name or None].
    Returns a bool array [n_samples, n_sources], True = present.
    """
    mask = np.ones((len(samples), len(sources)), dtype=bool)
    for i, sid in enumerate(samples):
        entry = missing.get(sid) if missing else None
        if entry is None:
            continue
        missing_sources = {entry} if isinstance(entry, str) else set(entry)
        for j, s in enumerate(sources):
            if s in missing_sources:
                mask[i, j] = False
    return mask


# --------------------------------------------------------------------------- #
# Self-test with synthetic data (both fusion types, missing modalities, and
# both contrastive losses in both clinical modes)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import pandas as pd

    torch.manual_seed(0)
    np.random.seed(0)

    n_samples = 64
    sources = ["rna", "cnv", "meth"]
    input_dims = {"rna": 100, "cnv": 80, "meth": 60}

    samples = [f"s{i}" for i in range(n_samples)]
    omics = {s: np.random.randn(n_samples, input_dims[s]).astype(np.float32) for s in sources}

    # simulate missing modalities: ~20% of patients miss one random modality
    present_mask = np.ones((n_samples, len(sources)), dtype=bool)
    rng = np.random.default_rng(0)
    for i in range(n_samples):
        if rng.random() < 0.2:
            j = rng.integers(len(sources))
            present_mask[i, j] = False
            omics[sources[j]][i] = 0.0  # zero-fill, as upstream pipeline does

    clinical_df = pd.DataFrame({
        "time": np.abs(np.random.randn(n_samples)) * 50 + 1,
        "status": (np.random.rand(n_samples) > 0.4).astype(int),
    }, index=samples)

    clinical_stage = rng.integers(0, 4, size=n_samples)  # 4 fake stages
    clinical_features = np.random.randn(n_samples, 5).astype(np.float32)  # fake standardized clinical vars

    configs = [
        dict(fusion_type="pairwise", lambda_contrastive_modality=0.0, lambda_contrastive_clinical=0.0),
        dict(fusion_type="cls_token", lambda_contrastive_modality=0.0, lambda_contrastive_clinical=0.0),
        dict(fusion_type="pairwise", lambda_contrastive_modality=0.5, lambda_contrastive_clinical=0.0),
        dict(fusion_type="pairwise", lambda_contrastive_modality=0.0, lambda_contrastive_clinical=0.3,
             clinical_contrastive_mode="stage"),
        dict(fusion_type="cls_token", lambda_contrastive_modality=0.3, lambda_contrastive_clinical=0.3,
             clinical_contrastive_mode="knn"),
        dict(fusion_type="pairwise", lambda_contrastive_modality=0.5, lambda_contrastive_clinical=0.0,
             two_phase_training=True, phase1_epochs=5),
        dict(fusion_type="cls_token", lambda_contrastive_modality=0.0, lambda_contrastive_clinical=0.3,
             clinical_contrastive_mode="stage", two_phase_training=True, phase1_epochs=5, patience=4),
    ]

    for cfg in configs:
        fusion_type = cfg.pop("fusion_type")
        print(f"\n=== fusion_type={fusion_type} | cfg={cfg} ===")
        model = CrossModalAttentionSurvivalModel(
            input_dims=input_dims, latent_dim=16, fusion_type=fusion_type, n_heads=4,
        )
        fit_kwargs = dict(
            omics_train=omics, present_mask=present_mask, clinical_df=clinical_df, samples=samples,
            event="status", surv_time="time",
            batch_size=16, n_epochs=15, lambda_surv=1.0, switch_epoch=5, patience=None, verbose=False,
        )
        fit_kwargs.update(cfg)
        if cfg.get("lambda_contrastive_clinical", 0.0) > 0:
            if cfg.get("clinical_contrastive_mode") == "knn":
                fit_kwargs["clinical_features"] = clinical_features
            else:
                fit_kwargs["clinical_stage"] = clinical_stage

        history = model.fit(**fit_kwargs)
        Z = model.get_latent_representation(omics, present_mask)
        assert Z.shape == (n_samples, 16)
        assert np.isfinite(Z).all()
        print("latent shape:", Z.shape, "| final losses:", {k: round(v[-1], 4) for k, v in history.items()})

    print("\nOK: all fusion types x contrastive-loss combinations trained and produced finite latents.")