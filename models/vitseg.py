import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_ch=3, embed_dim=256):  # in_ch=3, embed_dim适当减小
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_ch=3, embed_dim=256, depth=4, num_heads=4):  # embed_dim/heads适当减小
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, embed_dim)
        self.n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads) for _ in range(depth)
        ])
    def forward(self, x):
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        x = x + self.pos_embed
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None and skip.shape[2:] == x.shape[2:]:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ViTseg(nn.Module):
    def __init__(self, img_size=256, num_classes=1, patch_size=16, embed_dim=256, depth=4, num_heads=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.encoder = ViTEncoder(img_size, patch_size, 3, embed_dim, depth, num_heads)  # in_ch=3
        self.encoder_out_ch = embed_dim
        self.decoder_chs = [128, 64, 32, 16]

        self.up1 = UpBlock(embed_dim, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = UpBlock(32, 16)
        self.final_conv = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        b = x.shape[0]
        x_patch = self.encoder(x)
        x = x_patch.transpose(1, 2).reshape(b, self.encoder_out_ch, self.img_size // self.patch_size, self.img_size // self.patch_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x