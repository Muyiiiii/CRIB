import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from .CRIB_embedding import (
    PositionalEmbedding,
    TokenEmbedding,
    FixedEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
    DataEmbedding,
    DataEmbedding_STT,
    DataEmbedding_inverted,
)

from .CRIB_module import (
    Chomp1d,
    TemporalBlock,
    TCNBlock,
    TransformerEncoder,
    EncoderLayer,
    AttentionLayer,
    Attention,
    FlashAttention,
    CRIB_Encoder,
    CRIB_PredHead,
    RevIN,
)

from .CRIB_utils import (
    TriangularCausalMask,
)


class CRIB(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_num = args.patch_num
        self.enc_pos_emded = PositionalEmbedding(
            d_model=(args.patch_len + 1) // 2 * 2, max_len=5000
        )
        self.dec_pos_emded = PositionalEmbedding(d_model=args.model_dim, max_len=5000)

        self.ini_embedding = nn.Linear(
            in_features=args.patch_len, out_features=args.model_dim
        )

        self.encoder = CRIB_Encoder(args=args, patch_num=args.patch_num)

        self.predictor = CRIB_PredHead(args=args, patch_num=args.patch_num)

        self.revinlayer = RevIN(num_features=1)

        self.sample_size = args.batch_size  # only used for the first sample call.
        self.old_sample_size = 0

    def get_prior(self, prior_type="norm"):
        # get prior distribution
        # Gaussain uniform prior: N(0, I)
        assert prior_type == "norm"
        if not self.sample_size == self.old_sample_size:
            # For the corner case of last batch that changes the batch size.
            prior_loc = torch.zeros(self.args.model_dim).to(self.args.device)
            prior_cov = torch.eye(self.args.model_dim).to(self.args.device)
            self.prior = MultivariateNormal(loc=prior_loc, covariance_matrix=prior_cov)
            self.old_sample_size = self.sample_size
        return self.prior

    def get_nll(self, x, px_z, m_mask, m_exist):
        ## Calculate the nll for observed features (not m_mask, m_exist)
        # nll = -px_z.log_prob(x.reshape(self.args.batch_size, -1, self.args.var_num)) # shape = (M*K*BS, TL, D)
        nll = -px_z.log_prob(x)  # shape = (M*K*BS, TL, D)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        # if self.model_type not in ["vae"]: # m_mask is not None: hivae, gpvae, vmfvae, mis
        # nll = torch.where(m_mask, torch.zeros_like(nll), nll) # [M*K*BS, TL, D]
        nll = torch.sum(nll, dim=2)  # [M*K*BS, TL]
        # nll = torch.where(m_exist.bool(), nll, torch.zeros_like(nll))
        nll = torch.sum(nll)
        return nll

    def forward(self, x, x_mark=None, test_flag=False):
        B, P, N, L = x.shape  # [batch_size, patch_num, var_num, patch_len]

        x_1 = x

        # RevIN
        x_1 = x_1.permute(0, 1, 3, 2).reshape(B, P * L, N)
        x_1 = self.revinlayer(x_1, mode="norm")

        x_1 = x_1.reshape(B, P, L, N).permute(
            0, 1, 3, 2
        )  # [batch_size, patch_num, var_num, patch_len] --seq_len

        x_1 = x_1.reshape(B, P * N, L)

        x_1 = x_1 + self.enc_pos_emded(x_1)
        x_1 = x_1.reshape(B, P, N, L)

        noise = 0.01 * torch.normal(x_1.mean().item(), x_1.std().item(), x_1.shape).to(
            self.args.device
        )

        x_2 = x_1 + noise

        # prior distribution (Gaussian)
        pz = self.get_prior(prior_type="norm")

        # Encoder-Decoder
        enc_out_1, enc_attns_1, qz_x_1 = self.encoder(x_enc=x_1, x_mark=None)
        if test_flag:
            enc_out_2, enc_attns_2, qz_x_2 = None, None, None
        else:
            enc_out_2, enc_attns_2, qz_x_2 = self.encoder(x_enc=x_2, x_mark=None)

        if test_flag:
            z = qz_x_1.mean
        else:
            z = qz_x_1.rsample()
        # py_z=self.predictor_2(z)
        py_z = None

        ### ELBO with KL
        kl = torch.distributions.kl.kl_divergence(qz_x_1, pz)  # [M*K*BS, TL or d]
        kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
        # kl = torch.where(mask_1.bool(), kl, torch.zeros_like(kl))
        kl = torch.sum(kl)

        # metirc
        z_mean = qz_x_1.mean
        preds = self.predictor(enc_out_1)

        # RevIN
        preds = self.revinlayer(preds, mode="denorm")
        # enc_out_1=self.revinlayer(enc_out_1, mode='denorm')
        # enc_out_2=self.revinlayer(enc_out_2, mode='denorm')

        return enc_out_1, enc_attns_1, enc_out_2, enc_attns_2, preds, py_z, kl
