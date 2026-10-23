import torch
from torch import nn

class CorrectionNetwork(nn.Module):
    """cost_hat(q | x) = c_q - sum_i x_i * gamma_iq(x) * v_iq,  gamma in (0, 2), starting at exactly 1."""

    def __init__(self, benefits, baseline, hidden=(64,)):
        super().__init__()
        V = torch.as_tensor(benefits, dtype=torch.float32)          # (n_cand, n_tmpl)
        self.register_buffer('V', V)
        self.register_buffer('c', torch.as_tensor(baseline, dtype=torch.float32))  # (n_tmpl,)
        self.n_cand, self.n_tmpl = V.shape
        self.theta = nn.Parameter(torch.zeros_like(V))
        self.mlp = None
        if hidden:
            layers, d = [], self.n_cand
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU()]
                d = h
            self.mlp = nn.Sequential(*layers)
            self.head = nn.Linear(d, V.numel())
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def gamma(self, x):                                  # (B, n_cand) -> (B, n_cand, n_tmpl)
        z = self.theta.expand(x.shape[0], -1, -1)
        if self.mlp is not None:
            z = z + self.head(self.mlp(x)).view(-1, self.n_cand, self.n_tmpl)
        return 2 * torch.sigmoid(z)

    def forward(self, x):                                # predicted per-template cost, (B, n_tmpl)
        benefit = (x.unsqueeze(-1) * self.gamma(x) * self.V).sum(dim=1)
        return self.c - benefit