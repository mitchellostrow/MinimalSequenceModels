import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, repeat


class LRUBlock(nn.Module):
    # this is only implementing the state_space part
    # for the sake of speed, as in s4, we have both a convolutional and an rnn approach
    def __init__(self, input_dim, d_model, output_dim=None, rmin=0.8, rmax=0.99):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        self.init_A(d_model, rmin, rmax)
        # initialize complex B,C matrices with glorot, normalized by variance
        B_re = torch.rand(d_model, input_dim) / math.sqrt(2 * input_dim)
        B_im = torch.rand(d_model, input_dim) / math.sqrt(2 * input_dim)
        self.B = nn.Parameter(B_re + 1j * B_im)

        C_re = torch.rand(output_dim, d_model) / math.sqrt(d_model)
        C_im = torch.rand(output_dim, d_model) / math.sqrt(d_model)
        self.C = nn.Parameter(C_re + 1j * C_im)

        # initialize D as real, the skip connection
        # this may also need to have a omplex component to it for type issues
        D = torch.rand(output_dim, input_dim) / math.sqrt(input_dim)
        self.D = nn.Parameter(D)

    def init_A(self, d_model, rmin, rmax):
        # Lemma 3.2. Let 𝑢1, 𝑢2 be independent uniform random variables on the interval [0, 1].
        # Let 0 ≤ rmin ≤ rmax ≤ 1. Compute 𝜈 =−1/2 log(u1*(r_max^2-r_min^2)+r_min)
        # and 𝜃=2𝜋 * u2.
        # Then exp(−𝜈+𝑖𝜃) is uniformly distributed on the ring
        # in C between circles of radii 𝑟min and 𝑟max.
        # randomly initialize d_model numbers of u1 and u2
        u1 = torch.rand(d_model)
        u2 = torch.rand(d_model)

        nu = torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin**2))
        theta = torch.log(2 * math.pi * u2)
        # make nu and theta torch parameters

        self.nu_log = nn.Parameter(nu)
        self.theta_log = nn.Parameter(theta)

        gamma = torch.sqrt(1 - torch.exp(-torch.exp(nu)) ** 2)
        self.gamma_log = nn.Parameter(gamma)

    def forward(self, inputs, rnn=False):
        # input is a tensor of shape (batch_size, input_dim, L)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)

        skip = einsum(inputs, self.D, "b l h, d h -> b l d")
        # cast inputs to complex float
        inputs = inputs.to(torch.complex64)

        device = inputs.device
        (b, L, d) = inputs.shape

        # first, form A matrix to be diagonal
        A = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))

        B = (
            repeat(torch.exp(self.gamma_log), "h -> h w", w=self.B.shape[-1]) * self.B
        )  # scale by gamma (normalization)
        # initialize the input as gamma circ self.B @ input
        u = einsum(inputs, B, "b l d, h d -> b l h")

        if rnn:
            # TODO: add option for state input for autoregression implementation
            x = torch.zeros((b, L, self.d_model), dtype=torch.complex64).to(
                device
            )  # num_dimensions, num_states, time
            for i in range(L):
                x[:, i] = (
                    A * x[:, i - 1] + u[:, i]
                )  # einsum(A, xs, "d n, b d n -> b d n") + u[:, i]

        else:
            # Vandermonde multiplication
            H = A[:, None] ** torch.arange(L, device=device)
            h_f = torch.fft.fft(H, n=2 * L, dim=1)

            u_f = torch.fft.fft(u, n=2 * L, dim=1)  # (B H L)
            x = torch.fft.ifft(u_f * h_f.T, n=2 * L, dim=1)[:, :L]  # (B H L)

        y = einsum(x, self.C, "b l h, n h -> b l n").real + skip

        return x, y


class SISOLRUBlock(nn.Module):
    # runs the LRU in a single input single output fashion, 1 per dimension
    def __init__(self, input_dim, d_state, rmin=0.8, rmax=0.99):
        super().__init__()
        self.input_dim = input_dim

        self.lrus = nn.ModuleList(
            [LRUBlock(1, d_state, rmin=rmin, rmax=rmax) for _ in range(input_dim)]
        )

    def forward(self, inputs, rnn=False):
        hiddens = []
        outputs = []
        for i in range(self.input_dim):
            hidden, output = self.lrus[i](inputs[..., i : i + 1], rnn=rnn)
            hiddens.append(hidden)
            outputs.append(output)

        return torch.cat(hiddens, -1), torch.cat(outputs, -1)


class LRUMinimal(nn.Module):
    # a 1-layer LRU!
    def __init__(
        self,
        input_dim,
        d_model,  # dimensionality of the embedding
        d_state=None,  # dimensionality of the state space
        mlp_hidden=None,
        siso=False,
        rmin=0.8,
        rmax=0.99,
        seed=10,
        dropout=0.0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        if d_state is None:
            d_state = d_model
        if mlp_hidden is None:
            mlp_hidden = 4 * d_model

        self.encoder = nn.Linear(input_dim, d_model)

        self.ln = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        if not siso:
            self.lru = LRUBlock(
                d_model, d_state, output_dim=d_model, rmin=rmin, rmax=rmax
            )
        else:
            self.lru = SISOLRUBlock(d_model, d_state, rmin=rmin, rmax=rmax)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, input_dim),
        )
        self.rnn = False

    def eval(self):
        super().eval()  # do everything in the superclass, we love inheritance!
        self.rnn = True  # want to svve hidden states so do this

    def train(self, mode=True):  # super requires the mode argument
        super().train(mode)
        self.rnn = False

    def forward(self, inputs):
        # make sure shape is correct here!
        x = self.encoder(inputs)
        x = self.ln(x)
        hiddens, x = self.lru(x, rnn=self.rnn)
        x = self.ln2(x)
        x = self.mlp(x)
        return x, hiddens
