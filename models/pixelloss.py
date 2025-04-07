import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import DropPath, Mlp


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).cuda()
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).cuda()
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class CausalAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            norm_layer: nn.Module = nn.LayerNorm
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CausalBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CausalAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MlmLayer(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, vocab_size))

    def forward(self, x, word_embeddings):
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(x, word_embeddings)
        logits = logits + self.bias
        return logits


class TimeStepLoss(nn.Module):
    def __init__(self, c_channels, width, depth, num_heads):
        super().__init__()

        self.cond_proj = nn.Linear(c_channels, width)
        self.timestamp_proj = nn.Linear(1, width)
        self.ln = nn.LayerNorm(width, eps=1e-6)

        self.blocks = nn.ModuleList([
            CausalBlock(width, num_heads=num_heads, mlp_ratio=4.0,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        proj_drop=0, attn_drop=0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width, eps=1e-6)

        self.out_proj = nn.Linear(width, 1)
        self.criterion = torch.nn.MSELoss()
        self.initialize_weights()

    def initialize_weights(self):
        # parameters
        # torch.nn.init.normal_(self.timestamp_proj.weight, std=.02)
        # torch.nn.init.normal_(self.cond_proj.weight, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def predict(self, target, cond_list):
        target = target.reshape(target.size(0), -1)
        timestamps = target  # target represents timestamps in this case
        
        # take only the middle condition
        cond = cond_list[0]
        
        # Project condition and timestamps
        cond_proj = self.cond_proj(cond).unsqueeze(1)
        time_proj = self.timestamp_proj(timestamps.unsqueeze(-1))
        
        # Concatenate projections
        x = torch.cat([cond_proj, time_proj], dim=1)
        x = self.ln(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Project to output dimension
        predictions = self.out_proj(x[:, 1])  # Take the timestamp position output
        
        return predictions, timestamps

    def forward(self, target, cond_list):
        """Training forward pass"""
        predictions, timestamps = self.predict(target, cond_list)
        loss = self.criterion(predictions, timestamps)
        return loss.mean()

    def sample(self, cond_list, temperature, cfg):
        """Generate time series predictions"""
        if cfg == 1.0:
            bsz = cond_list[0].size(0)
        else:
            bsz = cond_list[0].size(0) // 2

        # Initialize with zeros
        initial_values = torch.zeros(bsz, 1).cuda()
        
        if cfg == 1.0:
            predictions, _ = self.predict(initial_values, cond_list)
            predictions = predictions * temperature
        else:
            # Apply classifier-free guidance
            preds_all, _ = self.predict(
                torch.cat([initial_values, initial_values], dim=0), 
                cond_list
            )
            preds_all = preds_all * temperature
            
            # Split conditional and unconditional predictions
            cond_preds = preds_all[:bsz]
            uncond_preds = preds_all[bsz:]
            
            # Apply CFG
            predictions = uncond_preds + cfg * (cond_preds - uncond_preds)
        
        # Add small random noise for variation
        predictions = predictions + temperature * torch.randn_like(predictions) * 0.1
        
        return predictions


class MultiVariateTimeStepLoss(nn.Module):
    def __init__(self, c_channels, width, depth, num_heads, num_features):
        super().__init__()

        self.cond_proj = nn.Linear(c_channels, width)
        self.timestamp_proj = nn.Linear(num_features, width)
        self.ln = nn.LayerNorm(width, eps=1e-6)

        self.blocks = nn.ModuleList([
            CausalBlock(width, num_heads=num_heads, mlp_ratio=4.0,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        proj_drop=0, attn_drop=0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width, eps=1e-6)

        self.out_proj = nn.Linear(width, num_features)
        self.criterion = torch.nn.MSELoss()
        self.initialize_weights()

    def initialize_weights(self):
        # parameters
        # torch.nn.init.normal_(self.timestamp_proj.weight, std=.02)
        # torch.nn.init.normal_(self.cond_proj.weight, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def predict(self, target, cond_list):
        
        cond = cond_list[0]  # (B, c_channels)
        cond_proj = self.cond_proj(cond).unsqueeze(1)     # (B, 1, width)
        time_proj = self.timestamp_proj(target).unsqueeze(1)  # (B, 1, width)
        x = torch.cat([cond_proj, time_proj], dim=1)      # (B, 2, width)
        x = self.ln(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        predictions = self.out_proj(x[:, 1])  # predict the timestamp feature vector
        return predictions, target

    def forward(self, target, cond_list):
        predictions, target = self.predict(target, cond_list)
        loss = self.criterion(predictions, target)
        return loss.mean()

    def sample(self, cond_list, temperature, cfg):
        if cfg == 1.0:
            bsz = cond_list[0].size(0)
        else:
            bsz = cond_list[0].size(0) // 2

        # Initialize with zeros for each feature
        initial_values = torch.zeros(bsz, self.timestamp_proj.in_features).cuda()

        if cfg == 1.0:
            predictions, _ = self.predict(initial_values, cond_list)
            predictions = predictions * temperature
        else:
            # Classifier-Free Guidance
            preds_all, _ = self.predict(
                torch.cat([initial_values, initial_values], dim=0),
                cond_list
            )
            preds_all = preds_all * temperature
            cond_preds = preds_all[:bsz]
            uncond_preds = preds_all[bsz:]
            predictions = uncond_preds + cfg * (cond_preds - uncond_preds)

        # Add small Gaussian noise for diversity
        predictions = predictions + temperature * torch.randn_like(predictions) * 0.1

        return predictions


class PixelLoss(nn.Module):
    def __init__(self, c_channels, width, depth, num_heads, r_weight=1.0):
        super().__init__()

        self.pix_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.pix_std = torch.Tensor([0.229, 0.224, 0.225])

        self.cond_proj = nn.Linear(c_channels, width)
        self.r_codebook = nn.Embedding(256, width)
        self.g_codebook = nn.Embedding(256, width)
        self.b_codebook = nn.Embedding(256, width)

        self.ln = nn.LayerNorm(width, eps=1e-6)
        self.blocks = nn.ModuleList([
            CausalBlock(width, num_heads=num_heads, mlp_ratio=4.0,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        proj_drop=0, attn_drop=0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width, eps=1e-6)

        self.r_weight = r_weight
        self.r_mlm = MlmLayer(256)
        self.g_mlm = MlmLayer(256)
        self.b_mlm = MlmLayer(256)

        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

        self.initialize_weights()

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.r_codebook.weight, std=.02)
        torch.nn.init.normal_(self.g_codebook.weight, std=.02)
        torch.nn.init.normal_(self.b_codebook.weight, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def predict(self, target, cond_list):
        target = target.reshape(target.size(0), target.size(1))
        # back to [0, 255]
        mean = self.pix_mean.cuda().unsqueeze(0)
        std = self.pix_std.cuda().unsqueeze(0)
        target = target * std + mean
        # add a very small noice to avoid pixel distribution inconsistency caused by banker's rounding
        target = (target * 255 + 1e-2 * torch.randn_like(target)).round().long()

        # take only the middle condition
        cond = cond_list[0]
        x = torch.cat(
            [self.cond_proj(cond).unsqueeze(1), self.r_codebook(target[:, 0:1]), self.g_codebook(target[:, 1:2]),
             self.b_codebook(target[:, 2:3])], dim=1)
        x = self.ln(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        with torch.cuda.amp.autocast(enabled=False):
            r_logits = self.r_mlm(x[:, 0], self.r_codebook.weight)
            g_logits = self.g_mlm(x[:, 1], self.g_codebook.weight)
            b_logits = self.b_mlm(x[:, 2], self.b_codebook.weight)

        logits = torch.cat([r_logits.unsqueeze(1), g_logits.unsqueeze(1), b_logits.unsqueeze(1)], dim=1)
        return logits, target

    def forward(self, target, cond_list):
        """ training """
        logits, target = self.predict(target, cond_list)
        loss_r = self.criterion(logits[:, 0], target[:, 0])
        loss_g = self.criterion(logits[:, 1], target[:, 1])
        loss_b = self.criterion(logits[:, 2], target[:, 2])

        if self.training:
            loss = (self.r_weight * loss_r + loss_g + loss_b) / (self.r_weight + 2)
        else:
            # for NLL computation
            loss = (loss_r + loss_g + loss_b) / 3

        return loss.mean()

    def sample(self, cond_list, temperature, cfg, filter_threshold=0):
        """ generation """
        if cfg == 1.0:
            bsz = cond_list[0].size(0)
        else:
            bsz = cond_list[0].size(0) // 2
        pixel_values = torch.zeros(bsz, 3).cuda()

        for i in range(3):
            if cfg == 1.0:
                logits, _ = self.predict(pixel_values, cond_list)
            else:
                logits, _ = self.predict(torch.cat([pixel_values, pixel_values], dim=0), cond_list)
            logits = logits[:, i]
            logits = logits * temperature

            if not cfg == 1.0:
                cond_logits = logits[:bsz]
                uncond_logits = logits[bsz:]

                # very unlikely conditional logits will be suppressed
                cond_probs = torch.softmax(cond_logits, dim=-1)
                mask = cond_probs < filter_threshold
                uncond_logits[mask] = torch.max(
                    uncond_logits,
                    cond_logits - torch.max(cond_logits, dim=-1, keepdim=True)[0] + torch.max(uncond_logits, dim=-1, keepdim=True)[0]
                )[mask]

                logits = uncond_logits + cfg * (cond_logits - uncond_logits)

            # get token prediction
            probs = torch.softmax(logits, dim=-1)
            sampled_ids = torch.multinomial(probs, num_samples=1).reshape(-1)
            pixel_values[:, i] = (sampled_ids.float() / 255 - self.pix_mean[i]) / self.pix_std[i]

        # back to [0, 1]
        return pixel_values

def main_pixelloss():
    batch_size = 8
    c_channels = 16
    width = 64
    depth = 2
    num_heads = 4

    model = PixelLoss(
        c_channels=c_channels,
        width=width,
        depth=depth,
        num_heads=num_heads,
        r_weight=1.0
    ).cuda()

    target = torch.rand(batch_size, 3).cuda()  # normalized RGB values
    cond_vector = torch.rand(batch_size, c_channels).cuda()
    cond_list = [cond_vector]

    model.train()
    loss = model(target, cond_list)
    print(f"[PixelLoss] Training loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        eval_loss = model(target, cond_list)
        print(f"[PixelLoss] Evaluation loss: {eval_loss.item()}")

        samples = model.sample(cond_list, temperature=1.0, cfg=1.0)
        print(f"[PixelLoss] Sampled RGB pixels:\n{samples}")

def main_timestep_loss():
    batch_size = 8
    c_channels = 16
    width = 64
    depth = 2
    num_heads = 4

    model = TimeStepLoss(
        c_channels=c_channels,
        width=width,
        depth=depth,
        num_heads=num_heads
    ).cuda()

    target = torch.rand(batch_size, 1).cuda()  # scalar timestamps
    cond_vector = torch.rand(batch_size, c_channels).cuda()
    cond_list = [cond_vector]

    model.train()
    loss = model(target, cond_list)
    print(f"[TimeStepLoss] Training loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        eval_loss = model(target, cond_list)
        print(f"[TimeStepLoss] Evaluation loss: {eval_loss.item()}")

        samples = model.sample(cond_list, temperature=1.0, cfg=1.0)
        print(f"[TimeStepLoss] Sampled timestamps:\n{samples}")


def main_multivariate_timestep_loss():
    batch_size = 8
    c_channels = 16
    num_features = 5
    width = 64
    depth = 2
    num_heads = 4

    model = MultiVariateTimeStepLoss(
        c_channels=c_channels,
        width=width,
        depth=depth,
        num_heads=num_heads,
        num_features=num_features
    ).cuda()

    target = torch.rand(batch_size, num_features).cuda()
    cond_vector = torch.rand(batch_size, c_channels).cuda()
    cond_list = [cond_vector]

    model.train()
    loss = model(target, cond_list)
    print(f"[MultiVariateTimeStepLoss] Training loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        eval_loss = model(target, cond_list)
        print(f"[MultiVariateTimeStepLoss] Evaluation loss: {eval_loss.item()}")

        samples = model.sample(cond_list, temperature=1.0, cfg=1.0)
        print(f"[MultiVariateTimeStepLoss] Sampled multivariate timestamps:\n{samples}")


if __name__ == "__main__":
    main_multivariate_timestep_loss()
    main_pixelloss()
    main_timestep_loss()
