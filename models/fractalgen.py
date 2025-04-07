from functools import partial

import torch
import torch.nn as nn

from models.ar import AR, ARTimeSeries
from models.mar import MAR
from models.pixelloss import (MultiVariateTimeStepLoss,  # whichever fits best
                              PixelLoss, TimeStepLoss)


class FractalGenTimeSeries(nn.Module):
    """Fractal Time Series Generative Model"""

    def __init__(
        self,
        seq_len_list,
        input_dim_list,
        embed_dim_list,
        num_blocks_list,
        num_heads_list,
        generator_type_list,
        label_drop_prob=0.1,
        class_num=1000,
        attn_dropout=0.1,
        proj_dropout=0.1,
        num_conds=1,
        grad_checkpointing=False,
        fractal_level=0,
    ):
        super().__init__()

        self.fractal_level = fractal_level
        self.num_fractal_levels = len(seq_len_list)

        # Class embedding only at top level
        if self.fractal_level == 0:
            self.num_classes = class_num
            self.class_emb = nn.Embedding(class_num, embed_dim_list[0])
            self.label_drop_prob = label_drop_prob
            self.fake_latent = nn.Parameter(torch.zeros(1, embed_dim_list[0]))
            torch.nn.init.normal_(self.class_emb.weight, std=0.02)
            torch.nn.init.normal_(self.fake_latent, std=0.02)

        # Generator model
        if generator_type_list[fractal_level] == "ar":
            generator = ARTimeSeries
        else:
            raise NotImplementedError(f"Unsupported generator type: {generator_type_list[fractal_level]}")

        self.generator = generator(
            seq_len=seq_len_list[fractal_level],
            input_dim=input_dim_list[fractal_level],
            cond_embed_dim=embed_dim_list[fractal_level - 1] if fractal_level > 0 else embed_dim_list[0],
            embed_dim=embed_dim_list[fractal_level],
            num_blocks=num_blocks_list[fractal_level],
            num_heads=num_heads_list[fractal_level],
            grad_checkpointing=grad_checkpointing,
        )

        # Recursive next-level or loss head
        if self.fractal_level < self.num_fractal_levels - 1:
            self.next_fractal = FractalGenTimeSeries(
                seq_len_list=seq_len_list,
                input_dim_list=input_dim_list,
                embed_dim_list=embed_dim_list,
                num_blocks_list=num_blocks_list,
                num_heads_list=num_heads_list,
                generator_type_list=generator_type_list,
                label_drop_prob=label_drop_prob,
                class_num=class_num,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                num_conds=num_conds,
                grad_checkpointing=grad_checkpointing,
                fractal_level=fractal_level + 1,
            )
        else:
            self.next_fractal = TimeStepLoss(  # or MultiVariateTimeStepLoss depending on output
                c_channels=embed_dim_list[fractal_level],
                width=embed_dim_list[fractal_level],
                depth=num_blocks_list[fractal_level],
                num_heads=num_heads_list[fractal_level]
            )

    def forward(self, seq, cond_ids):
        if self.fractal_level == 0:
            class_embedding = self.class_emb(cond_ids)
            if self.training:
                drop_mask = (torch.rand(cond_ids.size(0)) < self.label_drop_prob).unsqueeze(-1).to(seq.device)
                class_embedding = drop_mask * self.fake_latent + (1 - drop_mask) * class_embedding
            cond_list = [class_embedding for _ in range(5)]

        seq_out, cond_list_next, _ = self.generator(seq, cond_list)
        loss = self.next_fractal(seq_out, cond_list_next)
        return loss

    def sample(
        self,
        cond_ids,
        num_iter_list,
        cfg,
        cfg_schedule,
        temperature,
        filter_threshold,
        fractal_level=0
    ):
        if self.fractal_level == 0:
            class_embedding = self.class_emb(cond_ids)
            cond_list = [class_embedding for _ in range(5)]
        else:
            cond_list = cond_ids

        if fractal_level < self.num_fractal_levels - 1:
            next_level_fn = partial(
                self.next_fractal.sample,
                num_iter_list=num_iter_list,
                cfg=cfg,
                cfg_schedule=cfg_schedule,
                temperature=temperature,
                filter_threshold=filter_threshold,
                fractal_level=fractal_level + 1,
            )
        else:
            next_level_fn = self.next_fractal.sample

        return self.generator.sample(
            cond_list=cond_list,
            num_steps=num_iter_list[fractal_level],
            cfg=cfg,
            cfg_schedule=cfg_schedule,
            temperature=temperature,
            filter_threshold=filter_threshold,
            next_level_sample_function=next_level_fn,
        )



class FractalGen(nn.Module):
    """Fractal Generative Model"""

    def __init__(
        self,
        img_size_list,
        embed_dim_list,
        num_blocks_list,
        num_heads_list,
        generator_type_list,
        label_drop_prob=0.1,
        class_num=1000,
        attn_dropout=0.1,
        proj_dropout=0.1,
        guiding_pixel=False,
        num_conds=1,
        r_weight=1.0,
        grad_checkpointing=False,
        fractal_level=0,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # fractal specifics
        self.fractal_level = fractal_level
        self.num_fractal_levels = len(img_size_list)

        # --------------------------------------------------------------------------
        # Class embedding for the first fractal level
        if self.fractal_level == 0:
            self.num_classes = class_num
            self.class_emb = nn.Embedding(class_num, embed_dim_list[0])
            self.label_drop_prob = label_drop_prob
            self.fake_latent = nn.Parameter(torch.zeros(1, embed_dim_list[0]))
            torch.nn.init.normal_(self.class_emb.weight, std=0.02)
            torch.nn.init.normal_(self.fake_latent, std=0.02)

        # --------------------------------------------------------------------------
        # Generator for the current level
        if generator_type_list[fractal_level] == "ar":
            generator = AR
        elif generator_type_list[fractal_level] == "mar":
            generator = MAR
        else:
            raise NotImplementedError
        self.generator = generator(
            seq_len=(img_size_list[fractal_level] // img_size_list[fractal_level + 1])
            ** 2,
            patch_size=img_size_list[fractal_level + 1],
            cond_embed_dim=(
                embed_dim_list[fractal_level - 1]
                if fractal_level > 0
                else embed_dim_list[0]
            ),
            embed_dim=embed_dim_list[fractal_level],
            num_blocks=num_blocks_list[fractal_level],
            num_heads=num_heads_list[fractal_level],
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            guiding_pixel=guiding_pixel if fractal_level > 0 else False,
            num_conds=num_conds,
            grad_checkpointing=grad_checkpointing,
        )

        # --------------------------------------------------------------------------
        # Build the next fractal level recursively
        if self.fractal_level < self.num_fractal_levels - 2:
            self.next_fractal = FractalGen(
                img_size_list=img_size_list,
                embed_dim_list=embed_dim_list,
                num_blocks_list=num_blocks_list,
                num_heads_list=num_heads_list,
                generator_type_list=generator_type_list,
                label_drop_prob=label_drop_prob,
                class_num=class_num,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                guiding_pixel=guiding_pixel,
                num_conds=num_conds,
                r_weight=r_weight,
                grad_checkpointing=grad_checkpointing,
                fractal_level=fractal_level + 1,
            )
        else:
            # The final fractal level uses PixelLoss.
            self.next_fractal = PixelLoss(
                c_channels=embed_dim_list[fractal_level],
                depth=num_blocks_list[fractal_level + 1],
                width=embed_dim_list[fractal_level + 1],
                num_heads=num_heads_list[fractal_level + 1],
                r_weight=r_weight,
            )

    def forward(self, imgs, cond_list):
        """
        Forward pass to get loss recursively.
        """
        if self.fractal_level == 0:
            # Compute class embedding conditions.
            class_embedding = self.class_emb(cond_list)
            if self.training:
                # Randomly drop labels according to label_drop_prob.
                drop_latent_mask = (
                    (torch.rand(cond_list.size(0)) < self.label_drop_prob)
                    .unsqueeze(-1)
                    .cuda()
                    .to(class_embedding.dtype)
                )
                class_embedding = (
                    drop_latent_mask * self.fake_latent
                    + (1 - drop_latent_mask) * class_embedding
                )
            else:
                # For evaluation (unconditional NLL), use a constant mask.
                drop_latent_mask = (
                    torch.ones(cond_list.size(0))
                    .unsqueeze(-1)
                    .cuda()
                    .to(class_embedding.dtype)
                )
                class_embedding = (
                    drop_latent_mask * self.fake_latent
                    + (1 - drop_latent_mask) * class_embedding
                )
            cond_list = [class_embedding for _ in range(5)]

        # Get image patches and conditions for the next level
        imgs, cond_list, guiding_pixel_loss = self.generator(imgs, cond_list)
        # Compute loss recursively from the next fractal level.
        loss = self.next_fractal(imgs, cond_list)
        return loss + guiding_pixel_loss

    def sample(
        self,
        cond_list,
        num_iter_list,
        cfg,
        cfg_schedule,
        temperature,
        filter_threshold,
        fractal_level,
        visualize=False,
    ):
        """
        Generate samples recursively.
        """
        if fractal_level < self.num_fractal_levels - 2:
            next_level_sample_function = partial(
                self.next_fractal.sample,
                num_iter_list=num_iter_list,
                cfg_schedule="constant",
                fractal_level=fractal_level + 1,
            )
        else:
            next_level_sample_function = self.next_fractal.sample

        # Recursively sample using the current generator.
        return self.generator.sample(
            cond_list,
            num_iter_list[fractal_level],
            cfg,
            cfg_schedule,
            temperature,
            filter_threshold,
            next_level_sample_function,
            visualize,
        )


def fractalar_in64(**kwargs):
    model = FractalGen(
        img_size_list=(64, 4, 1),
        embed_dim_list=(1024, 512, 128),
        num_blocks_list=(32, 8, 3),
        num_heads_list=(16, 8, 4),
        generator_type_list=("ar", "ar", "ar"),
        fractal_level=0,
        **kwargs
    )
    return model


# FractalGen (Level 0, img_size=64x64)
# │
# ├── AR Generator:
# │    - Embed_dim: 1024
# │    - Num_blocks: 32
# │    - Num_heads: 16
# │    └── Generates (16x16) patches of 4x4 pixels each
# │
# └── next_fractal → FractalGen (Level 1, img_size=4x4)
#                   │
#                   ├── AR Generator:
#                   │    - Embed_dim: 512
#                   │    - Num_blocks: 8
#                   │    - Num_heads: 8
#                   │    └── Generates (4x4) patches of 1x1 pixels each
#                   │
#                   └── next_fractal → PixelLoss (Level 2, img_size=1x1)
#                                     - Embed_dim: 128
#                                     - Num_blocks: 3
#                                     - Num_heads: 4

# t1 -> 1 
# t2 -> 2
# t3 -> 3




def fractalmar_in64(**kwargs):
    model = FractalGen(
        img_size_list=(64, 4, 1),
        embed_dim_list=(1024, 512, 128),
        num_blocks_list=(32, 8, 3),
        num_heads_list=(16, 8, 4),
        generator_type_list=("mar", "mar", "ar"),
        fractal_level=0,
        **kwargs
    )
    return model


def fractalmar_base_in256(**kwargs):
    model = FractalGen(
        img_size_list=(256, 16, 4, 1),
        embed_dim_list=(768, 384, 192, 64),
        num_blocks_list=(24, 6, 3, 1),
        num_heads_list=(12, 6, 3, 4),
        generator_type_list=("mar", "mar", "mar", "ar"),
        fractal_level=0,
        **kwargs
    )
    return model


def fractalmar_large_in256(**kwargs):
    model = FractalGen(
        img_size_list=(256, 16, 4, 1),
        embed_dim_list=(1024, 512, 256, 64),
        num_blocks_list=(32, 8, 4, 1),
        num_heads_list=(16, 8, 4, 4),
        generator_type_list=("mar", "mar", "mar", "ar"),
        fractal_level=0,
        **kwargs
    )
    return model


def fractalmar_huge_in256(**kwargs):
    model = FractalGen(
        img_size_list=(256, 16, 4, 1),
        embed_dim_list=(1280, 640, 320, 64),
        num_blocks_list=(40, 10, 5, 1),
        num_heads_list=(16, 8, 4, 4),
        generator_type_list=("mar", "mar", "mar", "ar"),
        fractal_level=0,
        **kwargs
    )
    return model
