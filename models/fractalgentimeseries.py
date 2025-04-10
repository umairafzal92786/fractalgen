from functools import partial

import torch
import torch.nn as nn

from models.ar import ARTimeSeries
from models.mar import MARTimeSeries
from models.pixelloss import MultiVariateTimeStepLoss  # whichever fits best


class FractalGenTimeSeries(nn.Module):
    """Fractal Generative Model"""

    def __init__(
        self,
        series_size_list,
        input_feat_dim,
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

        # --------------------------------------------------------------------------
        # fractal specifics
        self.fractal_level = fractal_level
        self.num_fractal_levels = len(series_size_list)

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
            generator = ARTimeSeries
        elif generator_type_list[fractal_level] == "mar":
            generator = MARTimeSeries
        else:
            raise NotImplementedError
        self.generator = generator(
            seq_len=(series_size_list[fractal_level] // series_size_list[fractal_level + 1]),
            patch_size=series_size_list[fractal_level + 1],
            input_feat_dim=input_feat_dim,       
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
            num_conds=num_conds,
            grad_checkpointing=grad_checkpointing,
        )

        # --------------------------------------------------------------------------
        # Build the next fractal level recursively
        if self.fractal_level < self.num_fractal_levels - 2:
            self.next_fractal = FractalGenTimeSeries(
                series_size_list=series_size_list,
                input_feat_dim=input_feat_dim,
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
            # The final fractal level uses PixelLoss.
            self.next_fractal = MultiVariateTimeStepLoss(
                c_channels=embed_dim_list[fractal_level],
                depth=num_blocks_list[fractal_level + 1],
                width=embed_dim_list[fractal_level + 1],
                num_heads=num_heads_list[fractal_level + 1],
                num_features=input_feat_dim,
            )
            #print each argument of MultiVariateTimeStepLoss class
            # print("c_channels: ", embed_dim_list[fractal_level])
            # print("depth: ", num_blocks_list[fractal_level + 1])
            # print("width: ", embed_dim_list[fractal_level + 1])
            # print("num_heads: ", num_heads_list[fractal_level + 1])
            # print("num_features: ", input_feat_dim)

    def forward(self, imgs, cond_list):
        """
        Forward pass to get loss recursively.
        """

        # print("Fractal level: ", self.fractal_level)
        # print("Input shape: ", imgs.shape)
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
            # cond_list = [class_embedding for _ in range(5)]
            cond_list = [class_embedding]

        # print('here')
        # for item in cond_list:
        #     print("Condition shape: ", item.shape)

        # exit()

        # Get image patches and conditions for the next level
        imgs, cond_list, guiding_pixel_loss = self.generator(imgs, cond_list)
        # print("output shape: ", imgs.shape)
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




def fractaltimeseriesar_in64(**kwargs):
    model = FractalGenTimeSeries(
        series_size_list=(1024, 4, 1),
        input_feat_dim=6,
        embed_dim_list=(1024, 512, 128),
        num_blocks_list=(32, 8, 3),
        num_heads_list=(16, 8, 4),
        generator_type_list=("ar", "ar", "ar"),
        fractal_level=0,
        **kwargs
    )
    return model

def fractaltimeseriesmar_in64(**kwargs):
    model = FractalGenTimeSeries(
        series_size_list=(1024, 4, 1),
        input_feat_dim=6,
        embed_dim_list=(1024, 512, 128),
        num_blocks_list=(32, 8, 3),
        num_heads_list=(16, 8, 4),
        generator_type_list=("mar", "mar", "ar"),
        fractal_level=0,
        **kwargs
    )
    return model

# jiayu proposed using small at first level and bigger at middle levels then again small
# update higher level after few steps and lower levels frequently