import torch

import math

#TODO: cpp handlers

from tri_tier.constants import DEVICE

class TriTierCache():
    def __init__(self, max_seq_len, head_dim, num_heads, R_size, H_ratio):
        """
        Initialise the buffer size 
        R_size -> Recent Window Size
        H_ratio -> Heavy Hitter Percentile
        """
        
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
        self.HH_token_ids = torch.empty((self.max_heavy_hitters), -1, dtype= torch.int64, device= DEVICE)
        self.HH_scores = torch.empty((self.max_heavy_hitters), dtype=torch.float32, device= DEVICE)
        self.HH_count = 0

        #Packed 2bit storage
        #K quant through channels, V through per-token
        self.quant_head_dim = math.ceil(head_dim/16)
        self.PBS_K_Packed = torch.empty((self.max_background_tokens, num_heads, self.quant_head_dim), dtype=torch.int32, device= DEVICE)
        self.PBS_K_Scales = torch.empty((self.max_background_tokens, num_heads, head_dim), dtype=torch.float32, device= DEVICE)
        self.PBS_K_Zeroes = torch.empty((self.max_background_tokens, num_heads, head_dim), dtype=torch.float32, device= DEVICE)

        self.PBS_V_Packed = torch.empty((self.max_background_tokens, num_heads, self.quant_head_dim), dtype=torch.int32, device= DEVICE)
        self.PBS_V_Scales = torch.empty((self.max_background_tokens, num_heads, 1), dtype=torch.float32, device= DEVICE)
        self.PBS_V_Zeroes = torch.empty((self.max_background_tokens, num_heads, 1), dtype=torch.float32, device= DEVICE)
        
        self.PBS_token_ids = torch.empty((self.max_background_tokens), -1, dtype=torch.int64, device= DEVICE)
        self.PBS_count = 0

        #Global Attention Score 
        self.Global_Attn_Scr = torch.zeros((max_seq_len), dtype= torch.float32, device= DEVICE)
        self.Total_Processed_Tokens = 0

    def accumultae_attn_scrs(self, attn_weights : torch.Tensor) -> None:
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
