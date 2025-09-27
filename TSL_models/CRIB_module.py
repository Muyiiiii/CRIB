import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
from torch.nn.utils import weight_norm
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

import muyi.utils as mu

from .CRIB_utils import TriangularCausalMask
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


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """Reversible Instance Normalization for Accurate Time-Series Forecasting
               against Distribution Shift, ICLR2021.

        Parameters
        ----------
        num_features: int, the number of features or channels.
        eps: float, a value added for numerical stability, default 1e-5.
        affine: bool, if True(default), RevIN has learnable affine parameters.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError("Only modes norm and denorm are supported.")
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride, dilation, padding, dropout
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv2d(in_channel, out_channel, 1) if in_channel != out_channel else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # return self.relu(out + res)
        return out + res


class TCNBlock(nn.Module):
    def __init__(self, in_channel, out_channel_list, kernel_size, dropout):
        super().__init__()
        layers = []

        for i in range(len(out_channel_list)):
            dilation_size = 2**i
            in_channel = in_channel if i == 0 else out_channel_list[i - 1]
            out_channel = out_channel_list[i]
            layers += [
                TemporalBlock(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm_layer = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attns


class EncoderLayer(nn.Module):
    def __init__(self, attention, model_dim, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = 4 * model_dim
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=model_dim, out_channels=d_ff, kernel_size=1
        )  # equal to MLP
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=model_dim, kernel_size=1)
        # self.conv3 = nn.Conv2d(in_channels=model_dim, out_channels=d_ff, kernel_size=(3,1))
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)

        x = x + self.dropout(new_x)  # residual

        y = self.norm1(x)  # norm

        # MLP
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn  # residual


class AttentionLayer(nn.Module):
    def __init__(self, attention, model_dim, heads_num, keys_dim=None, values_dim=None):
        super().__init__()
        keys_dim = keys_dim or (model_dim // heads_num)
        values_dim = values_dim or (model_dim // heads_num)

        self.inner_attention = attention
        self.query_projection = nn.Linear(model_dim, keys_dim * heads_num)
        self.key_projection = nn.Linear(model_dim, keys_dim * heads_num)
        self.value_projection = nn.Linear(model_dim, values_dim * heads_num)
        self.out_projection = nn.Linear(values_dim * heads_num, model_dim)
        self.heads_num = heads_num

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.heads_num

        # queries = self.query_projection(queries).view(B, L, H, -1).permute(0,2,1,3)
        # keys = self.key_projection(keys).view(B, S, H, -1).permute(0,2,1,3)
        # values = self.value_projection(values).view(B, S, H, -1).permute(0,2,1,3)

        queries = self.query_projection(queries).view(B, L, H, -1)
        # keys = self.key_projection(keys).view(B, S, H, -1)
        # values = self.value_projection(values).view(B, S, H, -1)
        keys = self.query_projection(keys).view(B, S, H, -1)
        values = self.query_projection(values).view(B, S, H, -1)

        # print(f'Q: {queries.shape}')

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )

        # ts_scores = torch.zeros((L,S), dtype=queries.dtype).to(queries.device)
        # for i in range(1, ts_scores.shape[0]):
        #     ts_scores[i, i-1] += 10000

        # torch.backends.cuda.enable_flash_sdp(enabled=True)
        # out = F.scaled_dot_product_attention(queries,keys,values,attn_mask=ts_scores)
        # # print(f'{}')
        # attn = None

        # out = out.reshape(B, L, -1)
        out = out.permute(0, 2, 1, 3).reshape(B, L, -1)
        out = self.out_projection(out)

        return out, attn


class Attention(nn.Module):
    def __init__(
        self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape  # [batch_size, seq_len, hidden_size, embed_size]
        _, S, _, D = values.shape  # [batch_size, pred_len, hidden_size, embed_size]
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum(
            "blhe,bshe->bhls", queries, keys
        )  # head和head之间不用计算，因为并行也是浪费，但transpose之后head和head的计算能并行了

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FlashAttention(nn.Module):
    def __init__(
        self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.dropout_p = attention_dropout

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # B, L, H, E = queries.shape # [batch_size, seq_len, hidden_size, embed_size]
        # _, S, _, D = values.shape # [batch_size, pred_len, hidden_size, embed_size]
        # scale = self.scale or 1. / math.sqrt(E)

        # scores = torch.einsum("blhe,bshe->bhls", queries, keys) # head and head之间不用计算，因为并行也是浪费，但transpose之后head和head的计算能并行了

        # A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # V = torch.einsum("bhls,bshd->blhd", A, values)

        # if self.output_attention:
        #     return (V.contiguous(), A)
        # else:
        #     return (V.contiguous(), None)

        qkv = torch.stack((queries, keys, values), dim=2)

        # set alibi slopes
        alibi_slopes = torch.randn(queries.shape[2]).to(qkv.device)

        output = flash_attn_qkvpacked_func(
            qkv=qkv,
            dropout_p=self.dropout_p,
            softmax_scale=self.scale,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )

        return (output, None)


class CRIB_Encoder(nn.Module):
    def __init__(self, args, patch_num):
        super().__init__()
        self.args = args
        self.patch_num = patch_num

        self.softplus = nn.Softplus()

        # self.enc_embedding_1=DataEmbedding_inverted(input_size=args.patch_len, hidden_size=args.model_dim, dropout=args.dropout) # out: [batch_size, patch_num, var_num, model_dim] # Linear-embedding
        self.enc_embedding_1 = DataEmbedding_inverted(
            input_size=args.model_dim, hidden_size=args.model_dim, dropout=args.dropout
        )  # out: [batch_size, patch_num, var_num, model_dim] # Linear-embedding

        self.enc_embedding_2 = TCNBlock(
            in_channel=args.patch_len,
            out_channel_list=[64, args.model_dim],
            kernel_size=3,
            dropout=args.dropout,
        )

        self.encoder = TransformerEncoder(
            attn_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention=Attention(
                            mask_flag=False,
                            scale=None,
                            attention_dropout=args.dropout,
                            output_attention=args.output_attention,
                        ),
                        # attention=FlashAttention(mask_flag=False,
                        #                     scale=None,
                        #                     attention_dropout=args.dropout,
                        #                     output_attention=args.output_attention),
                        model_dim=args.model_dim,
                        heads_num=args.heads_num,
                    ),
                    model_dim=args.model_dim,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for _ in range(args.enc_num)
            ],
            norm_layer=nn.LayerNorm(args.model_dim),
        )

        # self.projector=nn.Linear(in_features=self.args.model_dim, out_features=self.args.patch_len)
        self.projector = nn.Sequential(
            nn.Linear(
                in_features=self.patch_num * self.args.model_dim,
                out_features=self.args.model_dim,
            ),
            nn.ReLU(),  # 激活函数: ReLU
            nn.Linear(
                self.args.model_dim, self.args.model_dim * 2
            ),  # 第二层: 隐藏层到输出层
        )

    def forward(self, x_enc, x_mark=None):
        B, P, N, L = x_enc.shape  # [batch_size, patch_num, var_num, patch_len]

        # mask_tokens=nn.Parameter(torch.zeros_like(x_enc))
        # x_enc = mask_enc * x_enc + (1 - mask_enc) * mask_tokens

        # enc_out=self.enc_embedding_1(x=x_enc, x_mark=None) # [batch_size, patch_num, var_num, model_dim]

        x_enc = x_enc.permute(0, 3, 2, 1)
        enc_out = self.enc_embedding_2(
            x=x_enc
        )  # [batch_size, model_dim, var_num, patch_num ]
        # print(f'enc_out embedding: {enc_out.shape}')
        enc_out = enc_out.permute(
            0, 3, 2, 1
        )  # [batch_size, patch_num, var_num, model_dim]

        enc_out = enc_out.reshape(
            B, -1, self.args.model_dim
        )  # [batch_size, patch_num * var_num, model_dim]
        # print(f'enc_out reshape: {enc_out.shape}')

        enc_out, attns = self.encoder(
            x=enc_out
        )  # [batch_size, patch_num * var_num, model_dim]
        # print(f'enc_out attention: {attns[1].shape}')

        enc_out_tmp = enc_out.reshape(B, P, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)
        mapped = self.projector(enc_out_tmp)

        # convert to distribution
        eps = (
            torch.ones_like(self.softplus(mapped[:, :, self.args.model_dim :])) * 1e-9
        )  # For the numerical stability.
        loc = mapped[:, :, : self.args.model_dim]
        scale = self.softplus(mapped[:, :, self.args.model_dim :]) + eps

        distribution = MultivariateNormal(
            loc=loc, covariance_matrix=torch.diag_embed(scale)
        )

        return enc_out, attns, distribution


class CRIB_PredHead(nn.Module):
    def __init__(self, args, patch_num):
        super().__init__()
        self.args = args
        self.patch_num = patch_num

        # self.prediction=nn.Linear(in_features=self.patch_num*args.model_dim, out_features=args.pred_len)
        self.prediction_1 = nn.Linear(
            in_features=self.patch_num * args.model_dim, out_features=args.model_dim
        )
        self.act_1 = nn.ReLU()
        self.prediction_2 = nn.Linear(
            in_features=args.model_dim, out_features=args.pred_len
        )
        # self.prediction=nn.Linear(in_features=args.model_dim, out_features=args.patch_len)

        self.prediction_3 = TCNBlock(
            in_channel=args.model_dim,
            out_channel_list=[int(args.model_dim / 2), int(args.model_dim / 2)],
            kernel_size=3,
            dropout=args.dropout,
        )
        self.prediction_4 = nn.Linear(
            in_features=self.patch_num * int(args.model_dim / 2),
            out_features=args.model_dim,
        )
        self.act_2 = nn.ReLU()
        self.act_3 = nn.ReLU()
        self.prediction_5 = nn.Linear(
            in_features=args.model_dim, out_features=args.pred_len
        )

    def forward(self, x_pred, x_mark=None):
        B, _, _ = x_pred.shape  # [batch_size, patch_num*var_num, model_dim]

        x_pred = x_pred.reshape(
            B, -1, self.args.var_num, self.args.model_dim
        )  # [batch_size, patch_num, var_num, model_dim]

        x_pred = x_pred.permute(0, 2, 1, 3).reshape(
            B, self.args.var_num, -1
        )  # [batch_size, var_num, patch_num*model_dim]
        x_pred = self.act_1(
            self.prediction_1(x_pred)
        )  # [batch_size, var_num, model_dim]
        x_pred = self.prediction_2(x_pred)  # [batch_size, var_num, pred_len]
        x_pred = x_pred.permute(0, 2, 1)  # [batch_size, pred_len, var_num]

        # x_pred=x_pred.permute(0,3,2,1) # [batch_size, model_dim, var_num, patch_num]
        # x_pred=self.act_2(self.prediction_3(x_pred)) # [batch_size, model_dim/2, var_num, patch_num]
        # x_pred=self.prediction_5(self.act_3(self.prediction_4(x_pred.permute(0,2,3,1).reshape(B, self.args.var_num, -1)))) # [batch_size, var_num, pred_len]
        # x_pred=x_pred.permute(0,2,1) # [batch_size, pred_len, var_num]

        return x_pred
