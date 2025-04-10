import torch

from models.mar import MAR, MARTimeSeries


def main_img():

    mar_model = MAR(
        seq_len=16,
        patch_size=4,
        cond_embed_dim=128,
        embed_dim=256,
        num_blocks=4,
        num_heads=8,
        attn_dropout=0.1,
        proj_dropout=0.1
    ).cuda()
    mar_model.train()  # use model.eval() if you want inference-like behavior

    # --- Dummy image input ---
    imgs = torch.randn(2, 3, 16, 16).cuda()        # [B, C, H, W]

    # --- Dummy condition vector ---
    cond_list = [torch.randn(2, 128).cuda()] 

    # --- Forward pass ---
    patches, cond_list_next, guiding_pixel_loss = mar_model(imgs, cond_list)
    print("MAR forward output patch shape:", patches.shape)

    def dummy_sample_function(cond_list, cfg, temperature, filter_threshold):
        return torch.randn(cond_list[0].size(0), 3 * mar_model.patch_size ** 2).cuda()
    
    # --- Sample image from cond_list ---
    sampled_imgs = mar_model.sample(
        cond_list=cond_list,
        num_iter=16,  # equal to seq_len
        cfg=1.0,
        cfg_schedule="linear",
        temperature=1.0,
        filter_threshold=0.9,
        next_level_sample_function=dummy_sample_function,
        visualize=False
    )
    print("Sampled image shape:", sampled_imgs.shape)  # Expect [B, 3, H, W]


def main_sr():
    print("Testing MARTimeSeries model on dummy time-series data...")

    # Initialize MARTimeSeries model
    mar_time_series_model = MARTimeSeries(
        seq_len=16,           # number of patches = T / patch_size
        patch_size=4,
        input_feat_dim=6,
        cond_embed_dim=1024,
        embed_dim=1024,
        num_blocks=32,
        num_heads=16,
        attn_dropout=0.1,
        proj_dropout=0.1,
        num_conds=1,
        grad_checkpointing=False
    ).cuda()

    # Create dummy input
    dummy_series = torch.randn(4, 64, 6).cuda()  # [B, T, F]
    dummy_cond = [torch.randn(4, 1024).cuda(), torch.randn(4, 1024).cuda()]   # [B, cond_dim]

    # Run forward pass
    mar_time_series_model.train()
    patches, cond_list_next, loss = mar_time_series_model(dummy_series, dummy_cond)
    print("MAR forward output patch shape:", patches.shape)
 
    def dummy_sample_function2(cond_list, cfg, temperature, filter_threshold):
        bsz = cond_list[0].size(0)
        patch_dim = mar_time_series_model.input_feat_dim * mar_time_series_model.patch_size
        return torch.randn(bsz, patch_dim).cuda()

    # Run sampling
    mar_time_series_model.eval()
    with torch.no_grad():
        sampled_series = mar_time_series_model.sample(
            cond_list=dummy_cond,
            num_iter=mar_time_series_model.seq_len,
            cfg=1.0,
            cfg_schedule="linear",
            temperature=1.0,
            filter_threshold=0.9,
            next_level_sample_function=dummy_sample_function2,
            visualize=False
        )
    print("MAR TimeSeries sample output shape:", sampled_series.shape)


if __name__ == "__main__":
    main_img()
    main_sr()
