import torch

import math

#TODO: cpp handlers

from tri_tier.constants import DEVICE, UPDATE_THRESHOLD, CHUNK_SIZE

class TriTierCache():
    def __init__(self, max_seq_len, head_dim, num_heads, R_size, H_ratio):
        """
        Initialise the buffer size 
        R_size -> Recent Window Size
        H_ratio -> Heavy Hitter Percentile
        """
        

        self.R_size = R_size

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


        #Packed 2bit storage
        #K quant through channels, V through per-token
        #[max_background_tokens, num_heads, head_dim]
        self.num_blocks = math.ceil(self.max_background_tokens / CHUNK_SIZE)
        self.quant_head_dim = math.ceil(head_dim/16)
        self.PBS_K_Packed = torch.empty((self.num_blocks, num_heads, head_dim), dtype=torch.int32, device= DEVICE)
        self.PBS_K_Scales = torch.empty((self.num_blocks, num_heads, head_dim), dtype=torch.float32, device= DEVICE)
        self.PBS_K_Zeroes = torch.empty((self.num_blocks, num_heads, head_dim), dtype=torch.float32, device= DEVICE)

        self.PBS_V_Packed = torch.empty((self.max_background_tokens, num_heads, self.quant_head_dim), dtype=torch.int32, device= DEVICE)
        self.PBS_V_Scales = torch.empty((self.max_background_tokens, num_heads, 1), dtype=torch.float32, device= DEVICE)
        self.PBS_V_Zeroes = torch.empty((self.max_background_tokens, num_heads, 1), dtype=torch.float32, device= DEVICE)
        
        self.PBS_token_ids = torch.full((self.max_background_tokens,), -1, dtype=torch.int64, device= DEVICE)
        self.PBS_count = 0

        #Global Attention Score 
        self.Global_Attn_Scr = torch.zeros((max_seq_len), dtype= torch.float32, device= DEVICE)
        self.Total_Processed_Tokens = 0

    def accumulate_attn_scrs(self, attn_weights : torch.Tensor) -> None:
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

        curren_seq_len = step_scores.shape[-1]
        mean_head_scores = torch.mean(step_scores, dim=0)

        self.Global_Attn_Scr[:curren_seq_len].add_(mean_head_scores)

    def ingest_token(self, new_k, new_v) -> None:
        """
        Filling the Tier 1- Recent Window
        if current rw count is less than window size(R_size)
        fills else
        overwrites 
        and routes to either t2 or t3
        """

        if self.RW_count < self.R_size:
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

                    removed_k = self.HH_K_Buffer[min_index]
                    removed_v = self.HH_V_Buffer[min_index]
                    removed_id = self.HH_token_ids[min_index]

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

            