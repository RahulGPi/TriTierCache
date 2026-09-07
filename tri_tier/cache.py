import torch

import math

#TODO: cpp handlers

from tri_tier.constants import (
    DEVICE, 
    UPDATE_THRESHOLD,
    CHUNK_SIZE,
    SINK_SIZE)

class TriTierCache():
    def __init__(self, max_seq_len, head_dim, num_heads, R_size, H_ratio):
        """
        Initialise the buffer size 
        R_size -> Recent Window Size
        H_ratio -> Heavy Hitter Percentile
        """
        

        self.R_size = R_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.current_threshold = 0.0

        self.max_heavy_hitters = math.ceil(max_seq_len * H_ratio)
        self.max_background_tokens = max_seq_len - R_size - self.max_heavy_hitters

        #Recennt Window, [R_size, num_heads, head_dim], FP32
        self.RW_K_Buffer = torch.empty((R_size, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
        self.RW_V_Buffer = torch.empty((R_size, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
        self.RW_count = 0
        self.RW_head_index = 0

        #Heavy Hitter, [max_heavy_hitters, num_heads, head_dim], FP32
        self.HH_K_Buffer = torch.empty((self.max_heavy_hitters, num_heads, head_dim), dtype= torch.float32, device= DEVICE)
        self.HH_V_Buffer = torch.empty((self.max_heavy_hitters, num_heads, head_dim), dtype= torch.float32, device= DEVICE)
        self.HH_token_ids = torch.full((self.max_heavy_hitters,), -1, dtype= torch.int64, device= DEVICE)
        self.HH_scores = torch.empty((self.max_heavy_hitters,), dtype=torch.float32, device= DEVICE)
        self.HH_count = 0

        #Waiting Room, [Chunk size, num_heads, head_dim]
        self.WR_K_Buffer = torch.empty((CHUNK_SIZE, num_heads, head_dim), dtype=torch.float32, device= DEVICE)
        self.WR_V_Buffer = torch.empty((CHUNK_SIZE, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
        self.WR_token_ids = torch.full((CHUNK_SIZE,), -1, dtype=torch.int64, device=DEVICE)
        self.WR_count = 0

        #Attention sink for unbounded conversation length
        #[Sink size, num heads, head_dim]
        self.S_K_Buffer = torch.empty((SINK_SIZE, num_heads, head_dim), dtype=torch.float32, device= DEVICE)
        self.S_V_Buffer = torch.empty((SINK_SIZE, num_heads, head_dim), dtype=torch.float32, device=DEVICE)
        self.S_count = 0

        #Packed 2bit storage
        #K quant through channels, V through per-token
        #[max_background_tokens, num_heads, head_dim]
        self.num_blocks = math.ceil(self.max_background_tokens / CHUNK_SIZE)
        self.quant_head_dim = math.ceil(head_dim/16)
        self.PBS_K_Packed = torch.empty((self.num_blocks, num_heads, head_dim), dtype=torch.int32, device= DEVICE)
        self.PBS_K_Scales = torch.empty((self.num_blocks, num_heads, head_dim), dtype=torch.float32, device= DEVICE)
        self.PBS_K_Zeroes = torch.empty((self.num_blocks, num_heads, head_dim), dtype=torch.float32, device= DEVICE)

        self.PBS_V_Packed = torch.empty((self.num_blocks * CHUNK_SIZE, num_heads, self.quant_head_dim), dtype=torch.int32, device= DEVICE)
        self.PBS_V_Scales = torch.empty((self.num_blocks * CHUNK_SIZE, num_heads, 1), dtype=torch.float32, device= DEVICE)
        self.PBS_V_Zeroes = torch.empty((self.num_blocks * CHUNK_SIZE, num_heads, 1), dtype=torch.float32, device= DEVICE)
        
        self.PBS_token_ids = torch.full((self.num_blocks * CHUNK_SIZE,), -1, dtype=torch.int64, device= DEVICE)
        self.PBS_is_full = False
        self.PBS_block_head_index = 0
        self.PBS_count = 0

        #Global Attention Score 
        self.Global_Attn_Scr = torch.zeros((max_seq_len), dtype= torch.float32, device= DEVICE)
        self.Total_Processed_Tokens = 0

    def accumulate_attn_scrs(self, attn_weights : torch.Tensor, full_ids) -> None:
        """
        Squeeze attn weights and avg acroos heads
        then add to global attn score
        """

        if attn_weights.dim() == 4:
            step_scores = attn_weights.squeeze(0).squeeze(1)
        elif attn_weights.dim() == 3:
            step_scores = attn_weights.squeeze(1)
        else:
            step_scores = attn_weights

        # curren_seq_len = step_scores.shape[-1]
        mean_head_scores = torch.mean(step_scores, dim=0)

        self.Global_Attn_Scr.index_add_(0, full_ids, mean_head_scores)

    def ingest_token(self, new_k, new_v) -> None:
        """
        Filling the Tier 1- Recent Window
        if current rw count is less than window size(R_size)
        fills else
        overwrites 
        and routes to either t2 or t3
        """
        if self.S_count < SINK_SIZE:
            self.S_K_Buffer[self.S_count] = new_k
            self.S_V_Buffer[self.S_count] = new_v
            self.S_count += 1 

        elif self.RW_count < self.R_size:
            self.RW_K_Buffer[self.RW_count] = new_k
            self.RW_V_Buffer[self.RW_count] = new_v
            self.RW_count += 1

        else:
            evicted_k = self.RW_K_Buffer[self.RW_head_index].clone()
            evicted_v = self.RW_V_Buffer[self.RW_head_index].clone()

            evicted_token_id = self.Total_Processed_Tokens - self.R_size

            self.RW_K_Buffer[self.RW_head_index] = new_k
            self.RW_V_Buffer[self.RW_head_index] = new_v

            self.RW_head_index = (self.RW_head_index + 1) % self.R_size

            self.route_evicted_token(evicted_k, evicted_v, evicted_token_id)

        self.Total_Processed_Tokens += 1

    def fetch_or_calc_thresh(self, evicted_token_id):
        """
        As the preallocated space is 0, slice and find the ones till updated threshold
        """

        if (evicted_token_id % UPDATE_THRESHOLD == 0) or self.current_threshold == 0.0:

            active_scores = self.Global_Attn_Scr[0: self.Total_Processed_Tokens]

            self.current_threshold = torch.quantile(active_scores, 0.95)

        return self.current_threshold    
    
    def route_evicted_token(self, evicted_k, evicted_v, evicted_token_id):
        """
        Check if heavy hitter
        else compress nd store in PBS
        """
        
        score = self.Global_Attn_Scr[evicted_token_id]

        threshold = self.fetch_or_calc_thresh(evicted_token_id=evicted_token_id)

        #Checking Heavy Hitter
        if score >= threshold:
            
            #if heavy hitter has empty positions
            if self.HH_count < self.max_heavy_hitters:
                self.HH_K_Buffer[self.HH_count] = evicted_k
                self.HH_V_Buffer[self.HH_count] = evicted_v
                self.HH_scores[self.HH_count] = score
                self.HH_token_ids[self.HH_count] = evicted_token_id
                self.HH_count += 1

            #HH is not full, 
            #Find lowest value and check if to replace it in HH
            else:
                
                min_index = torch.argmin(self.HH_scores)

                if score > self.HH_scores[min_index]:

                    removed_k = self.HH_K_Buffer[min_index].clone()
                    removed_v = self.HH_V_Buffer[min_index].clone()
                    removed_id = self.HH_token_ids[min_index].clone()

                    #TODO : compress_and_store()
                    self.compress_and_store(removed_k, removed_v, removed_id)

                    self.HH_K_Buffer[min_index] = evicted_k
                    self.HH_V_Buffer[min_index] = evicted_v
                    self.HH_scores[min_index] = score
                    self.HH_token_ids[min_index] = evicted_token_id
                
                #New token not in 5% 
                else:
                    self.compress_and_store(evicted_k, evicted_v, evicted_token_id)
        
        #New token not good
        else:
            self.compress_and_store(evicted_k, evicted_v, evicted_token_id)

            

    def compress_and_store(self, removed_k, removed_v, removed_token_id):
        """
        puts in waiting room, 
        checks if WR is full
        if is quantises keys per channel, and values by token
        then puts in PBS storage in 2bit values
        """


        #Put in the waiting room
        self.WR_K_Buffer[self.WR_count] = removed_k
        self.WR_V_Buffer[self.WR_count] = removed_v
        self.WR_token_ids[self.WR_count] = removed_token_id
        self.WR_count += 1

        #check if waiting room is full to be pushed into PBS
        if self.WR_count < CHUNK_SIZE:
            return
        
        #quantise keys per channel, across the WR
        K_Min = torch.amin(self.WR_K_Buffer, dim=0, keepdim=True)
        K_Max = torch.amax(self.WR_K_Buffer, dim = 0, keepdim= True)
        K_Scale = torch.clamp_min((K_Max - K_Min) / 3.0, min=1e-9)
        K_Quant = torch.clamp(torch.round((self.WR_K_Buffer - K_Min) / K_Scale), 0, 3).to(dtype=torch.int32)

        #quantise values per token, across head dim
        V_Min = torch.amin(self.WR_V_Buffer, dim=-1, keepdim=True)
        V_Max = torch.amax(self.WR_V_Buffer, dim = -1, keepdim= True)
        V_Scale = torch.clamp_min((V_Max - V_Min) / 3.0, min=1e-9)
        V_Quant = torch.clamp(torch.round((self.WR_V_Buffer - V_Min) / V_Scale), 0, 3).to(dtype=torch.int32)

        #bit packing 16 tokens -> 1 int 32, per-channel
        K_shifts = torch.arange(0, 32, 2, dtype=torch.int32).reshape(16, 1, 1)
        K_Shifted = torch.bitwise_left_shift(K_Quant, K_shifts)
        Packed_K  = torch.sum(K_Shifted, dim=0, dtype=torch.int32)

        #bit packing 16 channels -> 1 int 32, per token
        V_Grouped = V_Quant.reshape(CHUNK_SIZE, self.num_heads, self.quant_head_dim, 16)
        V_shifts = torch.arange(0, 32, 2, dtype=torch.int32).reshape(1, 1, 1, 16)
        Packed_V = torch.sum(torch.bitwise_left_shift(V_Grouped, V_shifts), dim=-1, dtype=torch.int32)

        #Write into storage
        if not self.PBS_is_full:
            block_idx = self.PBS_count // CHUNK_SIZE

        else:
            block_idx = self.PBS_block_head_index
            self.PBS_block_head_index = (self.PBS_block_head_index + 1) % self.num_blocks

        token_offset = block_idx * CHUNK_SIZE
        

        self.PBS_K_Packed[block_idx] = Packed_K
        self.PBS_K_Scales[block_idx] = K_Scale.squeeze(0)
        self.PBS_K_Zeroes[block_idx] = K_Min.squeeze(0)

        self.PBS_V_Packed[token_offset : token_offset + CHUNK_SIZE] = Packed_V
        self.PBS_V_Scales[token_offset : token_offset + CHUNK_SIZE] = V_Scale
        self.PBS_V_Zeroes[token_offset : token_offset + CHUNK_SIZE] = V_Min
        self.PBS_token_ids[token_offset : token_offset + CHUNK_SIZE] = self.WR_token_ids.clone()

        if not self.PBS_is_full:
            self.PBS_count += CHUNK_SIZE
            if self.PBS_count >= self.max_background_tokens:
                self.PBS_is_full = True


        self.WR_count = 0

    def reconstruct_full_cache(self):


        Sink_K = self.S_K_Buffer[0 : self.S_count]
        Sink_V = self.S_V_Buffer[0 : self.S_count]
        Sinks_ids = torch.arange(0, self.S_count)

        #dequantise keys from pbs per channel
        K_Shifts = torch.arange(0, 32 , 2, dtype=torch.int32, device=DEVICE).reshape(16, 1, 1, 1)
        K_Unpacked = torch.bitwise_and(torch.bitwise_right_shift(self.PBS_K_Packed.unsqueeze(0), K_Shifts), 0b11)
        K_Unpacked = K_Unpacked.permute(1, 0, 2, 3).reshape(self.num_blocks * CHUNK_SIZE, self.num_heads, self.head_dim)

        K_Scale_Expanded = self.PBS_K_Scales.repeat_interleave(CHUNK_SIZE, dim=0)
        K_Zero_Expanded = self.PBS_K_Zeroes.repeat_interleave(CHUNK_SIZE, dim= 0)
        Tier3_K = K_Unpacked.float() * K_Scale_Expanded + K_Zero_Expanded

        #Dequantise values from pbs per token
        V_Shifts = torch.arange(0, 32 , 2, dtype=torch.int32, device=DEVICE)
        V_Unpacked = torch.bitwise_and(torch.bitwise_right_shift(self.PBS_V_Packed.unsqueeze(-1), V_Shifts), 0b11)
        V_Unpacked = V_Unpacked.reshape(self.num_blocks * CHUNK_SIZE, self.num_heads, self.head_dim)
        Tier3_V = V_Unpacked.float() * self.PBS_V_Scales + self.PBS_V_Zeroes

        #drop empty padding slots
        valid_mask = (self.PBS_token_ids != -1)
        Tier3_K, Tier3_V, Tier3_token_ids = Tier3_K[valid_mask], Tier3_V[valid_mask], self.PBS_token_ids[valid_mask]

        HH_K_Buf = self.HH_K_Buffer[0 : self.HH_count]
        HH_V_Buf = self.HH_V_Buffer[0 : self.HH_count]
        HH_ids = self.HH_token_ids[0 : self.HH_count]

        #merging HH and PBS
        Middle_K = torch.cat([HH_K_Buf, Tier3_K], dim=0)
        Middle_V = torch.cat([HH_V_Buf, Tier3_V], dim=0)
        Middle_ids= torch.cat([HH_ids, Tier3_token_ids], dim=0)

        Sort_Perm = torch.argsort(Middle_ids)
        Middle_K = Middle_K[Sort_Perm]
        Middle_V = Middle_V[Sort_Perm]
        Middle_ids = Middle_ids[Sort_Perm]

        if self.RW_count < self.R_size:
            RW_K_Buff = self.RW_K_Buffer[0 : self.RW_count]
            RW_V_Buff = self.RW_V_Buffer[0 : self.RW_count]
            RW_ids = torch.arange(self.S_count, self.S_count + self.RW_count)

        else:
            RW_K_Buff = torch.roll(self.RW_K_Buffer, shifts=self.RW_head_index, dims=0)
            RW_V_Buff = torch.roll(self.RW_V_Buffer, shifts=self.RW_head_index, dims=0)
            RW_ids = torch.arange(self.Total_Processed_Tokens - self.R_size, self.Total_Processed_Tokens)


        K_Full = torch.cat([Sink_K, Middle_K, RW_K_Buff], dim=0)
        V_Full = torch.cat([Sink_V, Middle_V, RW_V_Buff], dim=0)
        Full_ids = torch.cat([Sinks_ids, Middle_ids, RW_ids], dim=0)

        return K_Full, V_Full, Full_ids