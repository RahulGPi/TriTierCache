import torch
import math

from .constants import (
    DEVICE, 
    UPDATE_THRESHOLD,
    CHUNK_SIZE,
    SINK_SIZE
)

class TriTierCache():
    def __init__(self, max_seq_len, head_dim, num_heads, R_size, H_ratio, num_q_heads=None, score_decay=0.999, k_group_size=16, pbs_metadata_dtype="fp16"):
        """
        Initialise the buffer size 
        R_size -> Recent Window Size
        H_ratio -> Heavy Hitter Percentile
        """
        self.R_size = R_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads if num_q_heads is not None else num_heads
        self.score_decay = score_decay
        self.k_group_size = 32 if k_group_size == 32 else 16
        self.pbs_metadata_dtype = "fp32" if str(pbs_metadata_dtype).lower() == "fp32" else "fp16"
        self.current_threshold = 0.0

        self.max_heavy_hitters = math.ceil(max_seq_len * H_ratio)
        self.max_background_tokens = max_seq_len - R_size - self.max_heavy_hitters

        # Try to initialize C++ cache engine
        try:
            from tri_tier import _C
            decay_val = 1.0 if score_decay is None else float(score_decay)
            self._engine = _C.TriTierCacheEngine(
                self.num_q_heads,
                self.num_heads,
                self.head_dim,
                max_seq_len,
                SINK_SIZE,
                R_size,
                H_ratio,
                decay_val,
                UPDATE_THRESHOLD,
                self.k_group_size,
                self.pbs_metadata_dtype
            )
        except Exception:
            self._engine = None

        if self._engine is not None:
            self.S_K_Buffer = torch.from_numpy(self._engine.get_S_K())
            self.S_V_Buffer = torch.from_numpy(self._engine.get_S_V())
            self.RW_K_Buffer = torch.from_numpy(self._engine.get_RW_K())
            self.RW_V_Buffer = torch.from_numpy(self._engine.get_RW_V())
            self.HH_K_Buffer = torch.from_numpy(self._engine.get_HH_K())
            self.HH_V_Buffer = torch.from_numpy(self._engine.get_HH_V())
            self.HH_token_ids = torch.from_numpy(self._engine.get_HH_token_ids())
            self.HH_scores = torch.from_numpy(self._engine.get_HH_scores())
            self.Global_Attn_Scr = torch.from_numpy(self._engine.get_global_attn_scores())
        else:
            self.S_K_Buffer = torch.empty((SINK_SIZE, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
            self.S_V_Buffer = torch.empty((SINK_SIZE, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
            self.RW_K_Buffer = torch.empty((R_size, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
            self.RW_V_Buffer = torch.empty((R_size, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
            self.HH_K_Buffer = torch.empty((self.max_heavy_hitters, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
            self.HH_V_Buffer = torch.empty((self.max_heavy_hitters, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
            self.HH_token_ids = torch.full((self.max_heavy_hitters,), -1, dtype=torch.int64, device=DEVICE)
            self.HH_scores = torch.empty((self.max_heavy_hitters,), dtype=torch.float32, device=DEVICE)
            self.Global_Attn_Scr = torch.zeros((max_seq_len,), dtype=torch.float32, device=DEVICE)

        self.RW_count = 0
        self.RW_head_index = 0
        self.HH_count = 0
        self.S_count = 0

        # Waiting Room
        self.WR_K_Buffer = torch.empty((CHUNK_SIZE, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
        self.WR_V_Buffer = torch.empty((CHUNK_SIZE, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
        self.WR_token_ids = torch.full((CHUNK_SIZE,), -1, dtype=torch.int64, device=DEVICE)
        self.WR_count = 0

        # Packed 2bit storage
        self.num_blocks = math.ceil(self.max_background_tokens / CHUNK_SIZE)
        self.quant_head_dim = math.ceil(head_dim/16)
        self.PBS_K_Packed = torch.empty((self.num_blocks, num_heads, head_dim), dtype=torch.int32, device=DEVICE)
        self.PBS_K_Scales = torch.empty((self.num_blocks, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
        self.PBS_K_Zeroes = torch.empty((self.num_blocks, num_heads, head_dim), dtype=torch.float32, device=DEVICE)

        self.PBS_V_Packed = torch.empty((self.num_blocks * CHUNK_SIZE, num_heads, self.quant_head_dim), dtype=torch.int32, device=DEVICE)
        self.PBS_V_Scales = torch.empty((self.num_blocks * CHUNK_SIZE, num_heads, 1), dtype=torch.float32, device=DEVICE)
        self.PBS_V_Zeroes = torch.empty((self.num_blocks * CHUNK_SIZE, num_heads, 1), dtype=torch.float32, device=DEVICE)
        
        self.PBS_token_ids = torch.full((self.num_blocks * CHUNK_SIZE,), -1, dtype=torch.int64, device=DEVICE)
        self.PBS_is_full = False
        self.PBS_block_head_index = 0
        self.PBS_count = 0

        self.Total_Processed_Tokens = 0
        self.total_evictions = 0

    def step(self, Q: torch.Tensor, K_new: torch.Tensor, V_new: torch.Tensor, attn_output: torch.Tensor):
        """Unified C++ entry point executing token ingestion, attention, and scoring in one call."""
        if self._engine is not None:
            self._engine.step(
                Q.data_ptr(),
                K_new.data_ptr(),
                V_new.data_ptr(),
                attn_output.data_ptr()
            )
            self.Total_Processed_Tokens = self._engine.total_processed_tokens
            self.total_evictions = self._engine.total_evictions
            self.PBS_count = self._engine.pbs_count
            self.RW_count = self._engine.rw_count
            self.RW_head_index = self._engine.rw_head_index
            self.S_count = self._engine.s_count
            self.HH_count = self._engine.hh_count
        else:
            raise NotImplementedError("C++ engine not initialized")

    def prefill(self, K_tokens: torch.Tensor, V_tokens: torch.Tensor) -> None:
        """
        Ingests a batch of prompt tokens into the cache at once.
        K_tokens, V_tokens: [q_len, num_heads, head_dim]
        """
        q_len = K_tokens.shape[0]
        if self._engine is not None:
            K_contig = K_tokens.contiguous()
            V_contig = V_tokens.contiguous()
            self._engine.prefill(K_contig.data_ptr(), V_contig.data_ptr(), q_len)
            self.Total_Processed_Tokens = self._engine.total_processed_tokens
            self.total_evictions = self._engine.total_evictions
            self.PBS_count = self._engine.pbs_count
            self.RW_count = self._engine.rw_count
            self.RW_head_index = self._engine.rw_head_index
            self.S_count = self._engine.s_count
            self.HH_count = self._engine.hh_count
        else:
            for i in range(q_len):
                self.ingest_token(K_tokens[i], V_tokens[i])

    def get_buffer_bytes(self) -> int:
        """Computes explicit allocated buffer memory in bytes across all live cache tensors."""
        if self._engine is not None:
            return self._engine.get_buffer_bytes()
        tensors = [
            self.S_K_Buffer, self.S_V_Buffer,
            self.RW_K_Buffer, self.RW_V_Buffer,
            self.HH_K_Buffer, self.HH_V_Buffer, self.HH_token_ids, self.HH_scores,
            self.WR_K_Buffer, self.WR_V_Buffer, self.WR_token_ids,
            self.PBS_K_Packed, self.PBS_K_Scales, self.PBS_K_Zeroes,
            self.PBS_V_Packed, self.PBS_V_Scales, self.PBS_V_Zeroes, self.PBS_token_ids,
            self.Global_Attn_Scr
        ]
        return sum(t.element_size() * t.nelement() for t in tensors if t is not None)

    def assert_evictions_occurred(self) -> None:
        """Explicit assertion ensuring benchmark runs exercised Tier 2 / Tier 3 compression tiers."""
        assert self.total_evictions > 0 or self.PBS_count > 0 or self.HH_count > 0, (
            f"TriTierCache ran with 0 evictions (total_tokens={self.Total_Processed_Tokens}, "
            f"R_size={self.R_size}, HH_count={self.HH_count}, PBS_count={self.PBS_count}). "
            "Quality benchmarks must run at context length >= 4x R_size to exercise compression tiers!"
        )

    def accumulate_attn_scrs(self, attn_weights: torch.Tensor, full_ids=None) -> None:
        """
        Squeeze attn weights and avg across heads
        then add to global attn score
        """
        if attn_weights.dim() == 4:
            if attn_weights.shape[2] > 1:
                raise RuntimeError("Multi-token prefill not supported in single-step accumulate")
            step_scores = attn_weights.squeeze(0).squeeze(1)
        elif attn_weights.dim() == 3:
            step_scores = attn_weights.squeeze(1)
        else:
            step_scores = attn_weights

        mean_scores = step_scores.mean(dim=0)
        n_scores = mean_scores.shape[0]

        if full_ids is not None:
            valid_mask = (full_ids != -1)
            valid_ids = full_ids[valid_mask]
            valid_scores = mean_scores[valid_mask]
            self.Global_Attn_Scr.index_add_(0, valid_ids, valid_scores)
        else:
            self.Global_Attn_Scr[0:n_scores] += mean_scores

    def fetch_or_calc_thresh(self, total_processed_tokens: int):
        """
        Extract only valid tokens, compute percentile threshold
        """
        if (total_processed_tokens % UPDATE_THRESHOLD == 0) or (
            (isinstance(self.current_threshold, (int, float)) and self.current_threshold == 0.0) or
            (isinstance(self.current_threshold, torch.Tensor) and self.current_threshold.item() == 0.0)
        ):
            n_tokens = self.Total_Processed_Tokens if self.Total_Processed_Tokens > 0 else total_processed_tokens
            valid_scores = self.Global_Attn_Scr[:n_tokens]
            if len(valid_scores) > 0:
                self.current_threshold = torch.quantile(valid_scores, 0.95)
        
        if isinstance(self.current_threshold, (int, float)):
            return torch.tensor(float(self.current_threshold))
        return self.current_threshold

    def ingest_token(self, K: torch.Tensor, V: torch.Tensor) -> None:
        """
        Ingests the new token into the recent window 
        Routes the evicted token if the recent window is full
        """
        if self.S_count < SINK_SIZE:
            self.S_K_Buffer[self.S_count] = K
            self.S_V_Buffer[self.S_count] = V
            self.S_count += 1
            self.Total_Processed_Tokens += 1
            return

        if self.RW_count < self.R_size:
            self.RW_K_Buffer[self.RW_count] = K
            self.RW_V_Buffer[self.RW_count] = V
            self.RW_count += 1
            self.Total_Processed_Tokens += 1
            return

        oldest_slot = self.RW_head_index
        evicted_token_id = self.Total_Processed_Tokens - self.R_size
        evicted_K = self.RW_K_Buffer[oldest_slot].clone()
        evicted_V = self.RW_V_Buffer[oldest_slot].clone()

        self.RW_K_Buffer[oldest_slot] = K
        self.RW_V_Buffer[oldest_slot] = V
        self.RW_head_index = (self.RW_head_index + 1) % self.R_size
        self.Total_Processed_Tokens += 1

        self.route_evicted_token(evicted_K, evicted_V, evicted_token_id)

    def route_evicted_token(self, K: torch.Tensor, V: torch.Tensor, token_id: int):
        """
        Routes the evicted token to either Heavy Hitter or Background Tier
        """
        self.total_evictions += 1
        score = self.Global_Attn_Scr[token_id].item() if isinstance(self.Global_Attn_Scr[token_id], torch.Tensor) else float(self.Global_Attn_Scr[token_id])
        current_threshold = self.current_threshold.item() if isinstance(self.current_threshold, torch.Tensor) else float(self.current_threshold)

        if score >= current_threshold:
            if self.HH_count < self.max_heavy_hitters:
                self.HH_K_Buffer[self.HH_count] = K
                self.HH_V_Buffer[self.HH_count] = V
                self.HH_token_ids[self.HH_count] = token_id
                self.HH_scores[self.HH_count] = score
                self.HH_count += 1
                return

            min_score, min_idx = torch.min(self.HH_scores[:self.HH_count], dim=0)
            if score > min_score.item():
                demoted_K = self.HH_K_Buffer[min_idx].clone()
                demoted_V = self.HH_V_Buffer[min_idx].clone()
                demoted_id = self.HH_token_ids[min_idx].clone()

                self.HH_K_Buffer[min_idx] = K
                self.HH_V_Buffer[min_idx] = V
                self.HH_token_ids[min_idx] = token_id
                self.HH_scores[min_idx] = score

                self.compress_and_store(demoted_K, demoted_V, demoted_id)
                return

        self.compress_and_store(K, V, token_id)

    def compress_and_store(self, K: torch.Tensor, V: torch.Tensor, token_id: int):
        self.WR_K_Buffer[self.WR_count] = K
        self.WR_V_Buffer[self.WR_count] = V
        self.WR_token_ids[self.WR_count] = token_id
        self.WR_count += 1

        if self.WR_count == CHUNK_SIZE:
            self._quantize_and_store()

    def _quantize_and_store(self):
        target_block = self.PBS_block_head_index

        k_chunk = self.WR_K_Buffer.unsqueeze(0)
        v_chunk = self.WR_V_Buffer.unsqueeze(0)

        # Vectorized fallback or C++ quantizer
        min_k = k_chunk.min(dim=1, keepdim=True).values
        max_k = k_chunk.max(dim=1, keepdim=True).values
        scale_k = (max_k - min_k) / 3.0
        scale_k[scale_k == 0] = 1.0
        q_k = torch.clamp(torch.round((k_chunk - min_k) / scale_k), 0, 3).to(torch.int32)
        shifts = torch.arange(0, 32, 2, dtype=torch.int32, device=DEVICE).view(1, CHUNK_SIZE, 1, 1)
        packed_k = torch.bitwise_left_shift(q_k, shifts).sum(dim=1).squeeze(0)

        self.PBS_K_Packed[target_block] = packed_k
        self.PBS_K_Scales[target_block] = scale_k.squeeze(0).squeeze(0)
        self.PBS_K_Zeroes[target_block] = min_k.squeeze(0).squeeze(0)

        min_v = v_chunk.min(dim=-1, keepdim=True).values
        max_v = v_chunk.max(dim=-1, keepdim=True).values
        scale_v = (max_v - min_v) / 3.0
        scale_v[scale_v == 0] = 1.0
        q_v = torch.clamp(torch.round((v_chunk - min_v) / scale_v), 0, 3).to(torch.int32)
        
        v_shifts = torch.arange(0, 32, 2, dtype=torch.int32, device=DEVICE).view(1, 1, 1, 16)
        padded_hd = self.quant_head_dim * 16
        if self.head_dim < padded_hd:
            q_v_pad = torch.nn.functional.pad(q_v, (0, padded_hd - self.head_dim))
        else:
            q_v_pad = q_v
        q_v_reshaped = q_v_pad.view(1, CHUNK_SIZE, self.num_heads, self.quant_head_dim, 16)
        packed_v = torch.bitwise_left_shift(q_v_reshaped, v_shifts).sum(dim=-1).squeeze(0)

        v_start = target_block * CHUNK_SIZE
        v_end = v_start + CHUNK_SIZE
        self.PBS_V_Packed[v_start:v_end] = packed_v
        self.PBS_V_Scales[v_start:v_end] = scale_v.squeeze(0)
        self.PBS_V_Zeroes[v_start:v_end] = min_v.squeeze(0)

        self.PBS_token_ids[v_start:v_end] = self.WR_token_ids.clone()
        self.PBS_count += CHUNK_SIZE
        self.PBS_block_head_index = (self.PBS_block_head_index + 1) % self.num_blocks

        self.WR_count = 0
        self.WR_token_ids.fill_(-1)

    def reconstruct_full_cache(self):
        Sink_K = self.S_K_Buffer[0 : self.S_count]
        Sink_V = self.S_V_Buffer[0 : self.S_count]
        Sinks_ids = torch.arange(0, self.S_count)

        Shifts = torch.arange(0, 32, 2, dtype=torch.int32, device=DEVICE).view(1, 16, 1, 1)
        K_Unpacked = torch.bitwise_and(torch.bitwise_right_shift(self.PBS_K_Packed.unsqueeze(1), Shifts), 0b11)
        K_Unpacked = K_Unpacked.reshape(self.num_blocks * CHUNK_SIZE, self.num_heads, self.head_dim)
        K_Scale_Expanded = self.PBS_K_Scales.repeat_interleave(CHUNK_SIZE, dim=0)
        K_Zero_Expanded = self.PBS_K_Zeroes.repeat_interleave(CHUNK_SIZE, dim=0)
        Tier3_K = K_Unpacked.float() * K_Scale_Expanded + K_Zero_Expanded

        V_Shifts = torch.arange(0, 32 , 2, dtype=torch.int32, device=DEVICE)
        V_Unpacked = torch.bitwise_and(torch.bitwise_right_shift(self.PBS_V_Packed.unsqueeze(-1), V_Shifts), 0b11)
        V_Unpacked = V_Unpacked.reshape(self.num_blocks * CHUNK_SIZE, self.num_heads, self.quant_head_dim * 16)[..., :self.head_dim]
        Tier3_V = V_Unpacked.float() * self.PBS_V_Scales + self.PBS_V_Zeroes

        valid_mask = (self.PBS_token_ids != -1)
        Tier3_K, Tier3_V, Tier3_token_ids = Tier3_K[valid_mask], Tier3_V[valid_mask], self.PBS_token_ids[valid_mask]

        HH_K_Buf = self.HH_K_Buffer[0 : self.HH_count]
        HH_V_Buf = self.HH_V_Buffer[0 : self.HH_count]
        HH_ids = self.HH_token_ids[0 : self.HH_count]

        Middle_K = torch.cat([HH_K_Buf, Tier3_K], dim=0)
        Middle_V = torch.cat([HH_V_Buf, Tier3_V], dim=0)
        Middle_ids= torch.cat([HH_ids, Tier3_token_ids], dim=0)

        if len(Middle_ids) > 0:
            Sort_Perm = torch.argsort(Middle_ids)
            Middle_K = Middle_K[Sort_Perm]
            Middle_V = Middle_V[Sort_Perm]
            Middle_ids = Middle_ids[Sort_Perm]

        if self.RW_count < self.R_size:
            RW_K_Buff = self.RW_K_Buffer[0 : self.RW_count]
            RW_V_Buff = self.RW_V_Buffer[0 : self.RW_count]
            RW_ids = torch.arange(self.S_count, self.S_count + self.RW_count)
        else:
            RW_K_Buff = torch.roll(self.RW_K_Buffer, shifts=-self.RW_head_index, dims=0)
            RW_V_Buff = torch.roll(self.RW_V_Buffer, shifts=-self.RW_head_index, dims=0)
            RW_ids = torch.arange(self.Total_Processed_Tokens - self.R_size, self.Total_Processed_Tokens)

        K_Full = torch.cat([Sink_K, Middle_K, RW_K_Buff], dim=0)
        V_Full = torch.cat([Sink_V, Middle_V, RW_V_Buff], dim=0)
        Full_ids = torch.cat([Sinks_ids, Middle_ids, RW_ids], dim=0)

        return K_Full, V_Full, Full_ids