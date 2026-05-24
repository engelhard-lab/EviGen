# model_evigen.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemperatureScheduler:
    def __init__(self, start: float, end: float, num_anneal_steps: int):
        """
        Temperature scheduler with linear annealing.

        Args:
            start: Initial temperature (high exploration)
            end: Final temperature (low exploration)
            num_anneal_steps: Number of training steps to anneal over
        """
        self.start = float(start)
        self.end = float(end)
        self.num_anneal_steps = max(1, int(num_anneal_steps))

    def get(self, step_idx: int) -> float:
        """
        Linear annealing from start to end over num_anneal_steps.
        After that, returns 'end' temperature.

        Args:
            step_idx: Current training step (batch index)

        Returns:
            Current temperature value
        """
        if step_idx >= self.num_anneal_steps - 1:
            return self.end
        alpha = step_idx / float(self.num_anneal_steps - 1)
        return (1.0 - alpha) * self.start + alpha * self.end


class EviGen(nn.Module):
    """
    EviGen with:
      - Shared always-on note query q_0 (retrieves notes only)
      - Note-only queries: indices [0 .. num_note_queries-1]
      - Code-only queries: indices [num_note_queries .. num_queries-1]
      - Note-level summary activates note queries
      - Code-level summary activates code queries
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.emb_dim
        n_q = cfg.num_queries

        # Query split
        self.num_note_queries = cfg.num_note_queries
        self.num_code_queries = cfg.num_code_queries

        if self.num_note_queries < 1:
            raise ValueError("num_note_queries must be >= 1 (query 0 is shared note query).")
        if self.num_note_queries + self.num_code_queries != n_q:
            raise ValueError(
                f"num_note_queries + num_code_queries must equal num_queries "
                f"({self.num_note_queries} + {self.num_code_queries} != {n_q})"
            )

        # Learnable queries:
        #   q_0 .. q_{num_note_queries-1}: note queries (including shared q_0)
        #   q_{num_note_queries} .. q_{N-1}: code queries
        self.query_vectors = nn.Parameter(torch.randn(n_q, d) * (1.0 / math.sqrt(d)))

        # Trainable thresholds for dynamic queries (1..N-1), in logit space.
        # Shared query 0 has no threshold.
        self.gating_threshold_logits = nn.Parameter(torch.zeros(n_q - 1))

        # Shared encoder for note/code summaries
        if cfg.aggregator.lower() == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=cfg.transformer_heads,
                dim_feedforward=cfg.transformer_dim_ff,
                dropout=cfg.dropout,
                batch_first=False,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=cfg.transformer_layers
            )
        else:
            self.encoder = None

        self.summary_ln = nn.LayerNorm(d)

        # Per-query expert MLPs with pre-LN + residual
        self.expert_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_q)])
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, cfg.expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.expert_hidden_dim, d),
                nn.Dropout(cfg.dropout),
            )
            for _ in range(n_q)
        ])

        self.final_ln = nn.LayerNorm(d)
        self.classifier = nn.Linear(d, 1)

        self.attn_temperature = cfg.attn_temperature

    # ---- Modality-specific summary ----

    def summarize_modality(self, chunks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        chunks: [B, L, d]
        mask:   [B, L] (bool)
        returns summary: [B, d]
        """
        if self.encoder is None:
            masked = chunks * mask.unsqueeze(-1)  # zero out padding
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            summary = masked.sum(dim=1) / lengths
        else:
            x = chunks.transpose(0, 1)  # [L, B, d]
            src_key_padding_mask = ~mask  # True for padding
            encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
            encoded = encoded.transpose(0, 1)  # [B, L, d]
            masked = encoded * mask.unsqueeze(-1)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            summary = masked.sum(dim=1) / lengths

        summary = self.summary_ln(summary)
        return summary  # [B, d]

    # ---- Hybrid gating with note & code summaries ----

    def hybrid_gating(
        self,
        summary_note: torch.Tensor,
        summary_code: torch.Tensor | None = None,
    ):
        """
        summary_note: [B, d]
        summary_code: [B, d] or None

        Returns:
          mask:    [B, N] binary (no grad) hard mask
          mask_st: [B, N] straight-through version (used for mixing)
          probs:   [B, N] sigmoid(cos) scores (note queries followed by code queries)
        """
        B, d = summary_note.shape
        device = summary_note.device
        N = self.query_vectors.size(0)
        Nn = self.num_note_queries
        Nc = self.num_code_queries

        # --- Note queries (indices [0..Nn-1]) ---
        q_notes = self.query_vectors[:Nn, :]  # [Nn, d]

        summary_note_norm = F.normalize(summary_note, dim=-1)    # [B, d]
        q_notes_norm = F.normalize(q_notes, dim=-1)              # [Nn, d]

        sims_note = torch.matmul(summary_note_norm, q_notes_norm.t())  # [B, Nn]
        probs_note = torch.sigmoid(sims_note)                           # [B, Nn]

        # Dynamic note queries: indices 1..Nn-1
        if Nn > 1:
            thr_note_logits = self.gating_threshold_logits[:Nn - 1]     # [Nn-1]
            thr_note = torch.sigmoid(thr_note_logits).unsqueeze(0).expand(B, -1)
            g_cont_note = probs_note[:, 1:] - thr_note                  # [B, Nn-1]
            g_hard_note = (g_cont_note > 0).float()
            g_st_note = g_hard_note + g_cont_note - g_cont_note.detach()
        else:
            # Only shared note query exists
            g_hard_note = summary_note.new_zeros(B, 0)
            g_st_note = summary_note.new_zeros(B, 0)

        # Shared note query 0 is always on
        shared = torch.ones(B, 1, device=device)
        mask_note = torch.cat([shared, g_hard_note], dim=1)      # [B, Nn]
        mask_st_note = torch.cat([shared, g_st_note], dim=1)     # [B, Nn]

        # --- Code queries (indices [Nn..N-1]) ---
        if Nc > 0:
            if summary_code is None:
                # Fallback: if no code summary is provided, reuse note summary
                summary_code = summary_note

            q_codes = self.query_vectors[Nn:, :]                 # [Nc, d]
            summary_code_norm = F.normalize(summary_code, dim=-1)
            q_codes_norm = F.normalize(q_codes, dim=-1)

            sims_code = torch.matmul(summary_code_norm, q_codes_norm.t())  # [B, Nc]
            probs_code = torch.sigmoid(sims_code)                           # [B, Nc]

            # Thresholds for code queries: remaining logits
            thr_code_logits = self.gating_threshold_logits[Nn - 1:]        # [Nc]
            thr_code = torch.sigmoid(thr_code_logits).unsqueeze(0).expand(B, -1)
            g_cont_code = probs_code - thr_code
            g_hard_code = (g_cont_code > 0).float()
            g_st_code = g_hard_code + g_cont_code - g_cont_code.detach()

            mask_code = g_hard_code                                       # [B, Nc]
            mask_st_code = g_st_code
        else:
            probs_code = summary_note.new_zeros(B, 0)
            mask_code = summary_note.new_zeros(B, 0)
            mask_st_code = summary_note.new_zeros(B, 0)

        probs = torch.cat([probs_note, probs_code], dim=1)       # [B, N]
        mask = torch.cat([mask_note, mask_code], dim=1)
        mask_st = torch.cat([mask_st_note, mask_st_code], dim=1)

        return mask, mask_st, probs

    # ---- Retrieval per modality ----

    def retrieve_chunks_modality(
        self,
        chunks: torch.Tensor,
        mask: torch.Tensor,
        queries: torch.Tensor,
        temperature: float,
        use_sampling: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        chunks:  [B, L, d]   (notes or codes)
        mask:    [B, L] bool
        queries: [Q, d]      queries assigned to this modality

        Returns:
          selected:      [B, Q, K, d]
          selected_mask: [B, Q, K] bool
        """
        cfg = self.cfg
        B, L, d = chunks.shape
        Q = queries.size(0)
        device = chunks.device
        K_target = max(1, int(cfg.k_retrieval))

        if Q == 0:
            # No queries for this modality; return an empty tensor with a dummy K dimension
            return chunks.new_zeros(B, 0, K_target, d), torch.zeros(
                B, 0, K_target, dtype=torch.bool, device=device
            )
        if L == 0:
            return chunks.new_zeros(B, Q, K_target, d), torch.zeros(
                B, Q, K_target, dtype=torch.bool, device=device
            )

        # Normalize queries only. The stored embeddings are already near unit norm,
        # so normalizing chunks is redundant in practice and makes zero-baseline
        # attribution ill-posed downstream.
        queries_norm = F.normalize(queries, dim=-1)            # [Q, d]

        # sim[b, q, l] = dot(chunks[b, l], queries_norm[q])
        sim = torch.einsum("bld,qd->bql", chunks, queries_norm)  # [B, Q, L]

        # Broadcast mask to [B, Q, L]
        mask_exp = mask.unsqueeze(1).expand(B, Q, L)             # bool

        sim_flat = sim.reshape(B * Q, L)
        mask_flat = mask_exp.reshape(B * Q, L)

        # Number of valid chunks per (b, q). Retrieval width is patient/query-specific:
        # short or missing sequences are padded instead of shrinking K for the whole batch.
        valid_counts = mask_flat.sum(dim=-1)                     # [B*Q]

        idx_flat = torch.zeros(B * Q, K_target, dtype=torch.long, device=device)
        idx_mask_flat = torch.zeros(B * Q, K_target, dtype=torch.bool, device=device)

        for row_idx in range(B * Q):
            valid_count = int(valid_counts[row_idx].item())
            if valid_count <= 0:
                continue

            row_logits = sim_flat[row_idx]
            row_mask = mask_flat[row_idx]
            k_row = min(K_target, valid_count)

            if use_sampling:
                logits = row_logits / max(temperature, 1e-6)
                logits = torch.where(
                    row_mask, logits, torch.full_like(logits, float("-1e9"))
                )
                probs = F.softmax(logits, dim=-1) * row_mask.float()
                probs_sum = probs.sum()
                if probs_sum <= 0:
                    probs = row_mask.float() / max(1, valid_count)
                else:
                    probs = probs / probs_sum
                chosen = torch.multinomial(probs, num_samples=k_row, replacement=False)
            else:
                logits = torch.where(
                    row_mask, row_logits, torch.full_like(row_logits, float("-1e9"))
                )
                chosen = torch.topk(logits, k=k_row, dim=-1).indices

            idx_flat[row_idx, :k_row] = chosen
            idx_mask_flat[row_idx, :k_row] = True

        # Gather embeddings
        chunks_exp = chunks.unsqueeze(1).expand(B, Q, L, d).contiguous()
        chunks_flat = chunks_exp.reshape(B * Q, L, d)            # [B*Q, L, d]
        idx_flat_exp = idx_flat.unsqueeze(-1).expand(-1, -1, d)  # [B*Q, K, d]
        selected_flat = torch.gather(chunks_flat, 1, idx_flat_exp)  # [B*Q, K, d]
        selected_flat = selected_flat * idx_mask_flat.unsqueeze(-1)
        selected = selected_flat.reshape(B, Q, K_target, d)      # [B, Q, K, d]
        selected_mask = idx_mask_flat.reshape(B, Q, K_target)

        return selected, selected_mask

    def linear_attention(
        self,
        selected: torch.Tensor,
        queries: torch.Tensor,
        selected_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        IRIS-style linear attention for a given modality.

        selected: [B, Q, K, d]
        queries:  [Q, d]
        Returns:
          context: [B, Q, d]
        """
        B, Q, K, d = selected.shape
        if Q == 0 or K == 0:
            return selected.new_zeros(B, 0, d)
        
        # Normalize queries only. Keeping embeddings in their original space avoids
        # the normalization singularity at the zero vector during IG.
        q_norm = F.normalize(queries, dim=-1)            # [Q, d]

        q_exp = q_norm.unsqueeze(0).expand(B, -1, -1)           # [B, Q, d]
        scores = torch.einsum("bqd,bqkd->bqk", q_exp, selected)  # [B, Q, K]
        if selected_mask is None:
            selected_mask = torch.ones(B, Q, K, dtype=torch.bool, device=selected.device)

        very_neg = torch.full_like(scores, float("-1e9"))
        masked_scores = torch.where(selected_mask, scores, very_neg)
        weights = F.softmax(masked_scores / self.attn_temperature, dim=-1)
        weights = weights * selected_mask.float()
        denom = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(denom > 0, weights / denom, torch.zeros_like(weights))
        context = torch.einsum("bqk,bqkd->bqd", weights, selected)  # [B, Q, d]
        return context

    # ---- Auxiliary loss for dynamic queries ----

    def moe_aux_loss(self):
        """
        Dynamic-MoE style diversity + simplicity loss on dynamic queries only
        (exclude shared query 0).

        Returns:
            total_loss: diversity + simplicity_weight * simplicity
            diversity: diversity loss value
            simplicity: simplicity loss value
        """
        N, d = self.query_vectors.shape
        device = self.query_vectors.device

        if N <= 1:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero

        # Dynamic queries: q_1..q_{N-1}
        q_dyn = self.query_vectors[1:, :]      # [N-1, d]
        K = q_dyn.size(0)

        # Diversity: || Q_norm Q_norm^T - I ||_F^2
        q_norm = F.normalize(q_dyn, dim=-1)
        gram = torch.matmul(q_norm, q_norm.t())           # [K, K]
        I = torch.eye(K, device=device)
        diversity = torch.sum((gram - I) ** 2)

        # Simplicity: average L2 norm squared
        simplicity = q_dyn.pow(2).sum(dim=-1).mean()

        # Weighted combination
        total_loss = diversity + self.cfg.simplicity_loss_weight * simplicity

        return total_loss, diversity, simplicity

    # ---- Forward ----

    def forward(
        self,
        note_chunks: torch.Tensor,
        note_mask: torch.Tensor,
        code_chunks: torch.Tensor,
        code_mask: torch.Tensor,
        temperature: float = 1.0,
        use_sampling_retrieval: bool = False,
    ):
        """
        note_chunks: [B, L_n, d]
        note_mask:   [B, L_n] bool
        code_chunks: [B, L_c, d]
        code_mask:   [B, L_c] bool

        Returns:
          logits: [B]
          gating_info: dict with 'mask', 'mask_st', 'probs'
          summary: [B, d]  (note-level summary)
          context: [B, N, d]
        """
        B, L_n, d = note_chunks.shape
        B2, L_c, d2 = code_chunks.shape
        assert B == B2 and d == d2, "Batch size / dim mismatch between notes and codes"

        # 1) Modality-specific summaries
        summary_note = self.summarize_modality(note_chunks, note_mask)  # [B, d]
        summary_code = None
        if self.num_code_queries > 0:
            summary_code = self.summarize_modality(code_chunks, code_mask)  # [B, d]

        # 2) Gating
        if getattr(self.cfg, "enable_dynamic_gating", True):
            mask_q, mask_q_st, probs = self.hybrid_gating(summary_note, summary_code)  # [B, N]
        else:
            N = self.cfg.num_queries
            ones = torch.ones(B, N, device=note_chunks.device, dtype=note_chunks.dtype)
            mask_q, mask_q_st, probs = ones, ones, ones

        # 3) Retrieval: separate for notes and codes
        q_notes = self.query_vectors[: self.num_note_queries, :]         # [Qn, d]
        selected_notes, selected_notes_mask = self.retrieve_chunks_modality(
            note_chunks, note_mask, q_notes, temperature, use_sampling_retrieval
        )  # [B, Qn, K_n, d]

        if self.num_code_queries > 0:
            q_codes = self.query_vectors[self.num_note_queries :, :]     # [Qc, d]
            selected_codes, selected_codes_mask = self.retrieve_chunks_modality(
                code_chunks, code_mask, q_codes, temperature, use_sampling_retrieval
            )  # [B, Qc, K_c, d]
            context_codes = self.linear_attention(
                selected_codes, q_codes, selected_codes_mask
            )   # [B, Qc, d]
        else:
            context_codes = note_chunks.new_zeros(B, 0, d)  # [B, 0, d]
            selected_codes_mask = torch.zeros(B, 0, max(1, self.cfg.k_retrieval), dtype=torch.bool, device=note_chunks.device)

        # 4) IRIS-style linear attention for note queries
        context_notes = self.linear_attention(
            selected_notes, q_notes, selected_notes_mask
        )   # [B, Qn, d]

        # Concatenate back to [B, N, d] in query index order
        context = torch.cat([context_notes, context_codes], dim=1)       # [B, N, d]

        # 5) Query-specific experts (pre-LN + residual)
        N = self.cfg.num_queries
        expert_outputs = []
        for i in range(N):
            v = context[:, i, :]                   # [B, d]
            z = self.expert_norms[i](v)
            delta = self.experts[i](z)            # [B, d]
            h = v + delta
            expert_outputs.append(h)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, N, d]

        # 6) Combine active experts with straight-through gating
        active_counts = mask_q.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
        mixed = (expert_outputs * mask_q_st.unsqueeze(-1)).sum(dim=1) / active_counts
        mixed = self.final_ln(mixed)  # [B, d]

        logits = self.classifier(mixed).squeeze(-1)  # [B]

        gating_info = {
            "mask": mask_q,       # hard binary mask
            "mask_st": mask_q_st,
            "probs": probs,
        }

        # Return note-level summary as 'summary' for compatibility
        return logits, gating_info, summary_note, context
