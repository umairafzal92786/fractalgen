import torch

from models.fractalgen import fractalar_in64


def main():
    # Dummy inputs
    batch_size = 4
    img_size = 64
    num_classes = 10
    
    # Create dummy image input: (B, C, H, W)
    dummy_imgs = torch.randn(batch_size, 3, img_size, img_size).cuda()
    
    # Dummy class labels
    dummy_labels = torch.randint(0, num_classes, (batch_size,)).cuda()

    # Instantiate the model
    model = fractalar_in64(class_num=num_classes).cuda()
    model.train()  # Set to train mode to enable label dropping

    # Forward pass
    loss = model(dummy_imgs, dummy_labels)

    print("Forward loss:", loss.item())


if __name__ == "__main__":
    main()



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

###########################
### All the time series are not equal and within time series the splits are not equal

