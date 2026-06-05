from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim_embedding: int, dim_feed_forward: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_embedding, dim_feed_forward),
            nn.GELU(),
            nn.Linear(dim_feed_forward, dim_embedding),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
