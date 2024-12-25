import torch
import torch.nn as nn
from einops import rearrange

class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=768):
        super().__init__()
        # 使用类似ResNet的CNN结构，但简化了层数
        self.conv_layers = nn.Sequential(
            # 224x224x3 -> 112x112x48
            nn.Conv2d(in_channels, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 112x112x48 -> 56x56x96
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            # 56x56x96 -> 28x28x192
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            # 28x28x192 -> 14x14x384
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # 14x14x384 -> 14x14x768
            nn.Conv2d(384, hidden_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(hidden_dim),
        )

    def forward(self, x):
        return self.conv_layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CNNTransformer(nn.Module):
    def __init__(self,
                 img_size=256,
                 in_channels=3,
                 num_classes=14,
                 hidden_dim=768,
                 num_heads=12,
                 num_layers=12,
                 mlp_ratio=4.,
                 drop_rate=0.1,
                 attn_drop_rate=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(in_channels, hidden_dim)

        # Calculate sequence length after CNN
        self.seq_length = (224 // 16) ** 2  # 14x14 = 196

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_length + 1, hidden_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer Encoder
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # MLP Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]

        # CNN feature extraction
        x = self.cnn(x)  # B, C, H, W

        # # Add shape assertion
        H, W = x.shape[2:]
        assert H * W + 1 == self.pos_embed.shape[
            1], f"Got sequence length {H * W + 1}, expected {self.pos_embed.shape[1]}"

        # Reshape to sequence
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder blocks
        x = self.blocks(x)
        x = self.norm(x)

        # Get class token output
        x = x[:, 0]

        # MLP head
        x = self.head(x)

        return x


# Example usage
if __name__ == "__main__":
    # Create model
    model = CNNTransformer(
        img_size=256,
        in_channels=3,
        num_classes=14,
        hidden_dim=768,
        num_heads=12,
        num_layers=12
    )
