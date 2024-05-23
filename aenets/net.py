import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    

class Swish(torch.nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)



class AE(nn.Module):
    def __init__(self, ipt_size, opt_size):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(ipt_size, opt_size),
            nn.ReLU(),
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(opt_size, ipt_size),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
class AET(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_heads=4, dim_feedforward=2048):
        super().__init__()
        output_dim = input_dim
        self.encoder = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.encoder_layers = nn.ModuleList([self.encoder for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([self.decoder for _ in range(num_layers)])
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, src):
        # src: (seq_len, batch_size)
        
        # Encoding
        x = self.fc_in(src)  # Project to hidden_dim
        for layer in self.encoder_layers:
            x = layer(x)
        # x: (seq_len, batch_size, hidden_dim) - This is the low-dimensional representation
        
        # Decoding
        for layer in self.decoder_layers:
            x = layer(x, x)  # In autoencoder, we use the same sequence for self-attention
        x = self.fc_out(x)  # Project back to output_dim
        
        return x



class AE2(nn.Module):
    def __init__(self, ipt_size, opt_size):
        super(AE2, self).__init__()

        # 编码器层
        self.encoder = nn.Linear(ipt_size, opt_size)
        self.decoder = nn.Linear(opt_size, ipt_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.mish = Mish()
        self.swish = Swish()
       

    def forward(self, x):
        x = nn.init.normal_(x, mean=0.0, std=1.0)
        x = self.encoder(x)
        x= self.relu(x)
        x = self.decoder(x)
        x = self.mish(x)
        return x



class AE_test(nn.Module):
    def __init__(self, ipt_size, opt_size):
        super(AE_test, self).__init__()

        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(ipt_size, opt_size),
            nn.ReLU(),
            nn.Linear(opt_size, opt_size),
            nn.ReLU(),
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(opt_size, ipt_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class AE_sw(nn.Module):
    def __init__(self, ipt_size, opt_size):
        super(AE_sw, self).__init__()

        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(ipt_size, opt_size),
            nn.ReLU(),
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(opt_size, ipt_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        self.encoder[0].weight = nn.Parameter(torch.transpose(self.decoder[0].weight, 0, 1))
        return x


class AE2layers(nn.Module):
    def __init__(self, ipt_size, opt_size):
        super(AE2layers, self).__init__()

        # 编码器层
        self.encoder = nn.Sequential(
            nn.Linear(ipt_size, int((ipt_size + opt_size) / 2.0)),
            nn.ReLU(),
            nn.Linear(int((ipt_size + opt_size) / 2.0), opt_size),
            nn.ReLU(),
        )

        # 解码器层
        self.decoder = nn.Sequential(
            nn.Linear(opt_size, int((ipt_size + opt_size) / 2.0)),
            nn.ReLU(),
            nn.Linear(int((ipt_size + opt_size) / 2.0), ipt_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
