import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        B_c, N_c, C_c = context.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B_c, N_c, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class StageSpecificCTIHook(nn.Module):
    """
   Strategy 1: Layer-by-layer injection.
- Inject L0 Map at the beginning of Stage 1
- Inject L1 Map at the beginning of Stage 2
- Inject L2 Map at the beginning of Stage 3
- Inject L3 Map at the beginning of Stage 4 
    """
    def __init__(
        self,
        topo_in_channels,
        token_dim,
        n_blocks,
        n_stages=4,
        mixer_heads=None,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.token_dim = token_dim
        
        # Split the multi-layer blocks of DINOv3 into n_stages evenly
        blocks_per_stage = n_blocks // n_stages
        self.stage_ends_all = [blocks_per_stage * (i + 1) - 1 for i in range(n_stages)]
        self.stage_ends_all[-1] = n_blocks - 1
        
        self.stage_starts = [0] + [e + 1 for e in self.stage_ends_all[:-1]]
        self.stage_to_level = {0: "L0", 1: "L1", 2: "L2", 3: "L3"}
        
        # Independent feature projections and cross attention corresponding to four stages
        self.topo_projs = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        
        heads = mixer_heads if mixer_heads is not None else max(1, token_dim // 64)
        
        for _ in range(n_stages):
           # Align the dimensions of Topo Map to those of DINO
            self.topo_projs.append(
                nn.Sequential(
                    nn.Conv2d(topo_in_channels, token_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(token_dim),
                    nn.GELU()
                )
            )
            self.norms1.append(nn.LayerNorm(token_dim))
            self.cross_attns.append(
                CustomCrossAttention(
                    dim=token_dim, 
                    num_heads=heads, 
                    qkv_bias=True, 
                    attn_drop=attn_dropout, 
                    proj_drop=proj_dropout
                )
            )
            self.norms2.append(nn.LayerNorm(token_dim))

    def forward(self, block_idx, x_list, topo_levels=None):
        if topo_levels is None:
            return x_list
            
        # Interactions are injected only at the beginning block of each stage
        if block_idx in self.stage_starts:
            stage_idx = self.stage_starts.index(block_idx)
            level_key = self.stage_to_level.get(stage_idx, None)
            
            topo_map = topo_levels.get(level_key, None)
            if topo_map is None:
                return x_list 
                
            is_list = isinstance(x_list, list)
            x_input = x_list[-1] if is_list else x_list
            orig_dtype = x_input.dtype
        
            x = x_input.float()
            topo_map = topo_map.float()
            
            # Reduce the spatial dimension to the DINO feature layer
            B, C, H, W = topo_map.shape
            topo_feat = self.topo_projs[stage_idx](topo_map) # [B, token_dim, H, W]
            topo_tokens = topo_feat.flatten(2).transpose(1, 2) # [B, H*W, token_dim]
            
            # Cross Attention (Image Token borrows the fine-grained features corresponding to each deep system from Topo Map)
            x_norm = self.norms1[stage_idx](x)
            topo_norm = self.norms2[stage_idx](topo_tokens)
            
            delta = self.cross_attns[stage_idx](x_norm, topo_norm)
            x_out = x + delta
            
            x_out = x_out.to(orig_dtype)
            
            if is_list:
                x_list.append(x_out)
                return x_list
            return x_out
            
        return x_list

    def post_block(self, block_idx, x_list, topo_levels=None):
        return x_list


