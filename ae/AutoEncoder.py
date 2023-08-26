import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import normalize
from torch.nn.init import xavier_normal_
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import orthogonal


class AE_single_layer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["hidden_dim"]
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=kwargs["hidden_dim"], out_features=512
        )
        self.decoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        code = self.encode(features)
        reconstructed = self.decode(code)
        return reconstructed, code

    def encode(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.tanh(activation)
        code = self.encoder_output_layer(activation)
        return code

    def decode(self, code):
        activation = self.decoder_hidden_layer(code)
        activation = torch.tanh(activation)
        reconstructed = self.decoder_output_layer(activation)
        return reconstructed


class Symmetric(nn.Module):
    def forward(self, X):
        # X = X.clamp(0, 10.0)
        if torch.isnan(X).any():
            print(X)
            raise ValueError("NAN! Stop Training")
        S, _, _ = torch.svd(X)
        return S  # Return a symmetric matrix


class InvertibleAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder_prelu_1 = nn.PReLU(init=.9)
        self.encoder_prelu_2 = nn.PReLU(init=.9)
        self.encoder_prelu_3 = nn.PReLU(init=.9)
        self.decoder_prelu_1 = nn.PReLU(init=1.11)
        self.decoder_prelu_2 = nn.PReLU(init=1.11)
        self.decoder_prelu_3 = nn.PReLU(init=1.11)

        self.encoder_hidden_layer_0 = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        parametrize.register_parametrization(self.encoder_hidden_layer_0, "weight", Symmetric())

        self.encoder_hidden_layer_1 = nn.Linear(
            in_features=512, out_features=512
        )
        parametrize.register_parametrization(self.encoder_hidden_layer_1, "weight", Symmetric())

        self.encoder_hidden_layer_2 = nn.Linear(
            in_features=512, out_features=512
        )
        parametrize.register_parametrization(self.encoder_hidden_layer_2, "weight", Symmetric())

        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["hidden_dim"]
        )
        parametrize.register_parametrization(self.encoder_output_layer, "weight", Symmetric())

        self.clamp = 5.
        self.scaling = nn.Parameter(0.02 * torch.randn(1, 512))

    def _log_e(self, s):
        return self.clamp * 0.693 * torch.atan(s / self.clamp)

    def forward(self, features):
        code = self.encode(features)
        reconstructed = self.decode(code)
        return reconstructed, code

    def encode(self, features):
        log_scaling = self._log_e(self.scaling)
        activation = self.encoder_hidden_layer_0(features * torch.exp(log_scaling))
        activation = self.encoder_prelu_1(activation)
        activation = self.encoder_hidden_layer_1(activation * torch.exp(log_scaling))
        activation = self.encoder_prelu_2(activation)
        activation = self.encoder_hidden_layer_2(activation * torch.exp(log_scaling))
        activation = self.encoder_prelu_3(activation)
        code = self.encoder_output_layer(activation * torch.exp(log_scaling))
        return code

    def decode(self, code):
        log_scaling = self._log_e(self.scaling)
        activation = F.linear(code * torch.exp(-log_scaling) - self.encoder_output_layer.bias,
                              self.encoder_output_layer.weight.t())
        activation = self.decoder_prelu_1(activation)
        activation = F.linear(activation * torch.exp(-log_scaling) - self.encoder_hidden_layer_2.bias,
                              self.encoder_hidden_layer_2.weight.t())
        activation = self.decoder_prelu_2(activation)
        activation = F.linear(activation * torch.exp(-log_scaling) - self.encoder_hidden_layer_1.bias,
                              self.encoder_hidden_layer_1.weight.t())
        activation = self.decoder_prelu_3(activation)
        reconstructed = F.linear(activation * torch.exp(-log_scaling) - self.encoder_hidden_layer_0.bias,
                                 self.encoder_hidden_layer_0.weight.t())
        return reconstructed


class CholeskyAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder_prelu_1 = nn.PReLU(init=.9)
        self.encoder_prelu_2 = nn.PReLU(init=.9)
        self.encoder_prelu_3 = nn.PReLU(init=.9)
        self.decoder_prelu_1 = nn.PReLU(init=1.1)
        self.decoder_prelu_2 = nn.PReLU(init=1.1)
        self.decoder_prelu_3 = nn.PReLU(init=1.1)

        self.encoder_hidden_layer_0 = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        cholesky(self.encoder_hidden_layer_0.weight)

        self.encoder_hidden_layer_1 = nn.Linear(
            in_features=512, out_features=512
        )
        cholesky(self.encoder_hidden_layer_1.weight)

        self.encoder_hidden_layer_2 = nn.Linear(
            in_features=512, out_features=512
        )
        cholesky(self.encoder_hidden_layer_2.weight)

        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["hidden_dim"]
        )
        cholesky(self.encoder_output_layer.weight)

    def forward(self, features):
        code = self.encode(features)
        reconstructed = self.decode(code)
        return reconstructed, code

    def encode(self, features):
        # log_scaling = self._log_e(self.scaling)
        activation = self.encoder_hidden_layer_0(features)
        activation = self.encoder_prelu_1(activation)
        activation = self.encoder_hidden_layer_1(activation)
        activation = self.encoder_prelu_2(activation)
        activation = self.encoder_hidden_layer_2(activation)
        activation = self.encoder_prelu_3(activation)
        code = self.encoder_output_layer(activation)
        return code

    def decode(self, code):
        # log_scaling = self._log_e(self.scaling)
        activation = F.linear(code - self.encoder_output_layer.bias,
                              self.encoder_output_layer.weight.t())
        activation = self.decoder_prelu_1(activation)
        activation = F.linear(activation - self.encoder_hidden_layer_2.bias,
                              self.encoder_hidden_layer_2.weight.t())
        activation = self.decoder_prelu_2(activation)
        activation = F.linear(activation - self.encoder_hidden_layer_1.bias,
                              self.encoder_hidden_layer_1.weight.t())
        activation = self.decoder_prelu_3(activation)
        reconstructed = F.linear(activation - self.encoder_hidden_layer_0.bias,
                                 self.encoder_hidden_layer_0.weight.t())
        return reconstructed


class InvOrthogonalAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder_prelu_0 = nn.PReLU(init=.9)
        self.encoder_prelu_1 = nn.PReLU(init=.9)
        self.encoder_prelu_2 = nn.PReLU(init=.9)
        self.encoder_prelu_3 = nn.PReLU(init=.9)
        self.encoder_prelu_4 = nn.PReLU(init=.9)
        self.encoder_prelu_5 = nn.PReLU(init=.9)

        self.decoder_prelu_0 = nn.PReLU(init=1.1)
        self.decoder_prelu_1 = nn.PReLU(init=1.1)
        self.decoder_prelu_2 = nn.PReLU(init=1.1)
        self.decoder_prelu_3 = nn.PReLU(init=1.1)
        self.decoder_prelu_4 = nn.PReLU(init=1.1)
        self.decoder_prelu_5 = nn.PReLU(init=1.1)

        self.encoder_hidden_layer_0 = orthogonal(nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        ))

        self.encoder_hidden_layer_1 = orthogonal(nn.Linear(
            in_features=512, out_features=512
        ))

        self.encoder_hidden_layer_2 = orthogonal(nn.Linear(
            in_features=512, out_features=512
        ))

        self.encoder_hidden_layer_3 = orthogonal(nn.Linear(
            in_features=512, out_features=512
        ))

        self.encoder_hidden_layer_4 = orthogonal(nn.Linear(
            in_features=512, out_features=512
        ))

        self.encoder_hidden_layer_5 = orthogonal(nn.Linear(
            in_features=512, out_features=512
        ))

        self.encoder_output_layer = orthogonal(nn.Linear(
            in_features=512, out_features=kwargs["hidden_dim"]
        ))

    def forward(self, features):
        code = self.encode(features)
        reconstructed = self.decode(code)
        return reconstructed, code

    def encode(self, features):
        activation = self.encoder_hidden_layer_0(features)
        activation = self.encoder_prelu_0(activation)

        activation = self.encoder_hidden_layer_1(activation)
        activation = self.encoder_prelu_1(activation)

        activation = self.encoder_hidden_layer_2(activation)
        activation = self.encoder_prelu_2(activation)

        activation = self.encoder_hidden_layer_3(activation)
        activation = self.encoder_prelu_3(activation)

        activation = self.encoder_hidden_layer_4(activation)
        activation = self.encoder_prelu_4(activation)

        activation = self.encoder_hidden_layer_5(activation)
        activation = self.encoder_prelu_5(activation)

        code = self.encoder_output_layer(activation)
        return code

    def decode(self, code):
        activation = F.linear(code - self.encoder_output_layer.bias,
                              self.encoder_output_layer.weight.t())

        activation = self.decoder_prelu_5(activation)
        activation = F.linear(activation - self.encoder_hidden_layer_5.bias,
                              self.encoder_hidden_layer_5.weight.t())

        activation = self.decoder_prelu_4(activation)
        activation = F.linear(activation - self.encoder_hidden_layer_4.bias,
                              self.encoder_hidden_layer_4.weight.t())

        activation = self.decoder_prelu_3(activation)
        activation = F.linear(activation - self.encoder_hidden_layer_3.bias,
                              self.encoder_hidden_layer_3.weight.t())

        activation = self.decoder_prelu_2(activation)
        activation = F.linear(activation - self.encoder_hidden_layer_2.bias,
                              self.encoder_hidden_layer_2.weight.t())

        activation = self.decoder_prelu_1(activation)
        activation = F.linear(activation - self.encoder_hidden_layer_1.bias,
                              self.encoder_hidden_layer_1.weight.t())

        activation = self.decoder_prelu_0(activation)
        reconstructed = F.linear(activation - self.encoder_hidden_layer_0.bias,
                                 self.encoder_hidden_layer_0.weight.t())
        return reconstructed


def init_enc_weights(m):
    if type(m) == nn.Linear:
        xavier_normal_(m.weight)
        m.weight = (m.weight + m.weight.t()) / 2
        m.weight, V, D = torch.svd(m.weight)
        m.weight = nn.Parameter(m.weight)
        return m


def cholesky(m):
    if type(m) == nn.Linear:
        xavier_normal_(m.weight)
        m.weight = (m.weight + m.weight.t()) / 2
        m.weight = torch.cholesky(m.weight, upper=False)
        m.weight = nn.Parameter(m.weight)
        return m


class AE_single_layer_prelu(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )

        self.encoder_prelu = nn.PReLU(init=0.9)

        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["hidden_dim"]
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=kwargs["hidden_dim"], out_features=512
        )

        self.decoder_prelu = nn.PReLU(init=1.1)

        self.decoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        code = self.encode(features)
        reconstructed = self.decode(code)
        return reconstructed, code

    def encode(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = self.encoder_prelu(activation)
        code = self.encoder_output_layer(activation)
        return code

    def decode(self, code):
        activation = self.decoder_hidden_layer(code)
        activation = self.decoder_prelu(activation)
        reconstructed = self.decoder_output_layer(activation)
        return reconstructed


class AE_multiple_layers(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.nonlinear_enc_1 = nn.PReLU(init=.9)
        self.nonlinear_enc_2 = nn.PReLU(init=.9)
        self.nonlinear_enc_3 = nn.PReLU(init=.9)
        self.nonlinear_dec_1 = nn.PReLU(init=1. / .9)
        self.nonlinear_dec_2 = nn.PReLU(init=1. / .9)
        self.nonlinear_dec_3 = nn.PReLU(init=1. / .9)

        self.encoder_hidden_layer_1 = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        xavier_normal_(self.encoder_hidden_layer_1.weight)

        self.encoder_hidden_layer_2 = nn.Linear(
            in_features=512, out_features=512
        )
        xavier_normal_(self.encoder_hidden_layer_2.weight)

        self.encoder_hidden_layer_3 = nn.Linear(
            in_features=512, out_features=512
        )
        xavier_normal_(self.encoder_hidden_layer_3.weight)

        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["hidden_dim"]
        )
        xavier_normal_(self.encoder_output_layer.weight)

        self.decoder_hidden_layer_1 = nn.Linear(
            in_features=kwargs["hidden_dim"], out_features=512
        )
        self.decoder_hidden_layer_1.weight = nn.Parameter(self.encoder_output_layer.weight.t())

        self.decoder_hidden_layer_2 = nn.Linear(
            in_features=512, out_features=512
        )
        self.decoder_hidden_layer_2.weight = nn.Parameter(self.encoder_hidden_layer_3.weight.t())

        self.decoder_hidden_layer_3 = nn.Linear(
            in_features=512, out_features=512
        )
        self.decoder_hidden_layer_3.weight = nn.Parameter(self.encoder_hidden_layer_2.weight.t())

        self.decoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["input_shape"]
        )
        self.decoder_output_layer.weight = nn.Parameter(self.encoder_hidden_layer_1.weight.t())

    def forward(self, features):
        code = self.encode(features)
        reconstructed = self.decode(code)
        return reconstructed, code

    def encode(self, features):
        activation = self.encoder_hidden_layer_1(features)
        activation = self.nonlinear_enc_1(activation)
        activation = self.encoder_hidden_layer_2(activation)
        activation = self.nonlinear_enc_2(activation)
        activation = self.encoder_hidden_layer_3(activation)
        activation = self.nonlinear_enc_3(activation)
        code = self.encoder_output_layer(activation)
        return code

    def decode(self, code):
        activation = self.decoder_hidden_layer_1(code)
        activation = self.nonlinear_dec_1(activation)
        activation = self.decoder_hidden_layer_2(activation)
        activation = self.nonlinear_dec_2(activation)
        activation = self.decoder_hidden_layer_3(activation)
        activation = self.nonlinear_dec_3(activation)
        reconstructed = self.decoder_output_layer(activation)
        return reconstructed


class DeepAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder_prelu_0 = nn.PReLU(init=.9)
        self.encoder_prelu_1 = nn.PReLU(init=.9)
        self.encoder_prelu_2 = nn.PReLU(init=.9)
        self.encoder_prelu_3 = nn.PReLU(init=.9)
        self.encoder_prelu_4 = nn.PReLU(init=.9)
        self.encoder_prelu_5 = nn.PReLU(init=.9)

        self.decoder_prelu_0 = nn.PReLU(init=1. / .9)
        self.decoder_prelu_1 = nn.PReLU(init=1. / .9)
        self.decoder_prelu_2 = nn.PReLU(init=1. / .9)
        self.decoder_prelu_3 = nn.PReLU(init=1. / .9)
        self.decoder_prelu_4 = nn.PReLU(init=1. / .9)
        self.decoder_prelu_5 = nn.PReLU(init=1. / .9)

        self.encoder_input_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )

        self.encoder_hidden_layer_1 = nn.Linear(
            in_features=512, out_features=512
        )

        self.encoder_hidden_layer_2 = nn.Linear(
            in_features=512, out_features=512
        )

        self.encoder_hidden_layer_3 = nn.Linear(
            in_features=512, out_features=512
        )

        self.encoder_hidden_layer_4 = nn.Linear(
            in_features=512, out_features=512
        )

        self.encoder_hidden_layer_5 = nn.Linear(
            in_features=512, out_features=512
        )

        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["hidden_dim"]
        )

        self.decoder_input_layer = nn.Linear(
            in_features=kwargs["hidden_dim"], out_features=512
        )

        self.decoder_hidden_layer_5 = nn.Linear(
            in_features=512, out_features=512
        )

        self.decoder_hidden_layer_4 = nn.Linear(
            in_features=512, out_features=512
        )

        self.decoder_hidden_layer_3 = nn.Linear(
            in_features=512, out_features=512
        )

        self.decoder_hidden_layer_2 = nn.Linear(
            in_features=512, out_features=512
        )

        self.decoder_hidden_layer_1 = nn.Linear(
            in_features=512, out_features=512
        )

        self.decoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        code = self.encode(features)
        reconstructed = self.decode(code)
        return reconstructed, code

    def encode(self, features):
        activation = self.encoder_input_layer(features)
        activation = self.encoder_prelu_0(activation)
        activation = self.encoder_hidden_layer_1(activation)
        activation = self.encoder_prelu_1(activation)
        activation = self.encoder_hidden_layer_2(activation)
        activation = self.encoder_prelu_2(activation)
        activation = self.encoder_hidden_layer_3(activation)
        activation = self.encoder_prelu_3(activation)
        activation = self.encoder_hidden_layer_4(activation)
        activation = self.encoder_prelu_4(activation)
        activation = self.encoder_hidden_layer_5(activation)
        activation = self.encoder_prelu_5(activation)
        code = self.encoder_output_layer(activation)
        return code

    def decode(self, code):
        activation = self.decoder_input_layer(code)
        activation = self.decoder_prelu_5(activation)
        activation = self.decoder_hidden_layer_5(activation)
        activation = self.decoder_prelu_4(activation)
        activation = self.decoder_hidden_layer_4(activation)
        activation = self.decoder_prelu_3(activation)
        activation = self.decoder_hidden_layer_3(activation)
        activation = self.decoder_prelu_2(activation)
        activation = self.decoder_hidden_layer_2(activation)
        activation = self.decoder_prelu_1(activation)
        activation = self.decoder_hidden_layer_1(activation)
        activation = self.decoder_prelu_0(activation)
        reconstructed = self.decoder_output_layer(activation)
        return reconstructed


def relu_loss(output, target):
    loss = torch.abs(output - target) - 0.04  # 4% of the range of the data color
    return torch.sum(torch.relu(loss))


def loss_on_animal_faces_1(output, target):
    '''
    We want to punish when label is coded on <1,
    and we are OK when it is between [1, +inf);
    other labels - coding different attributes - we punish normally
    '''
    # loss = torch.abs(output - target) - (1.0 * target)
    loss = output - target - (1.0 * target)
    return torch.sum(torch.relu(-loss))


def loss_on_animal_faces_2(output, target, begin=1.0, end=2.0):
    '''
    We are OK when label is between [1, 2];
    other labels - coding different attributes - we punish normally
    '''
    loss_1 = output - target - ((begin - 1) * target)
    loss_2 = output - target - ((end - 1) * target)
    return torch.sum(torch.relu(-loss_1)) + torch.sum(torch.relu(loss_2))


def loss_on_animal_faces_3(output, target):
    '''
    We want to punish when label is coded on <1,
    and we are OK when it is between [1, +inf);
    other labels - coding different attributes - we punish normally
    '''
    loss = output - target
    return torch.sum(torch.relu(-loss) ** 2)
