import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, num_latents, Encoder, EncoderToLatents, LatentsToDecoder, Decoder):
        super().__init__()

        self.num_latents = num_latents

        self.encoder = Encoder()
        self.enc_to_lat = EncoderToLatents(num_latents)
        self.lat_to_dec = LatentsToDecoder(num_latents)
        self.decoder = Decoder()

    def forward(self, x, *, reparameterise=True):
        x_enc = self.encoder(x)
        mu, logvar = self.enc_to_lat(x_enc)
        z = self.reparameterise(mu, logvar) if reparameterise else mu
        z_dec = self.lat_to_dec(z)
        logits = self.decoder(z_dec)

        return {
            "x_enc": x_enc,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_dec": z_dec,
            "logits": logits
        }

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + (std * eps)

    def x_to_mu_logvar(self, x):
        x_enc = self.encoder(x)
        mu, logvar = self.enc_to_lat(x_enc)

        return x_enc, mu, logvar

    def z_to_logits(self, z):
        z_dec = self.lat_to_dec(z)
        logits = self.decoder(z_dec)

        return z_dec, logits
