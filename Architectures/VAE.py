import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self,
                 num_latents,
                 num_latents_group,
                 Encoder,
                 EncoderToLatents,
                 LatentsToDecoder,
                 Decoder,
                 *,
                 size=28,
                 channels=1,
                 out_channels=None):
        super().__init__()

        self.num_latents = num_latents
        self.num_latents_group = num_latents if num_latents_group is None else num_latents_group
        self.num_groups = num_latents // self.num_latents_group

        self.encoder = Encoder(num_latents, size, channels, out_channels)
        self.enc_to_lat = [EncoderToLatents(num_latents, self.num_latents_group) for _ in range(self.num_groups)]
        self.lat_to_dec = [LatentsToDecoder(num_latents, self.num_latents_group) for _ in range(self.num_groups)]
        self.decoder = Decoder(num_latents, size, channels, out_channels)

    def forward(self, x, *, reparameterise=True):
        # x to x_enc, mu, logvar
        x_enc, mu, logvar = self.x_to_mu_logvar(x)

        # mu, logvar to z
        z = self.reparameterise(mu, logvar) if reparameterise else mu

        # z to z_dec, logits
        z_dec, logits = self.z_to_logits(z)

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
        mu, logvar = self.x_enc_to_mu_logvar(x_enc)

        return x_enc, mu, logvar

    def x_enc_to_mu_logvar(self, x_enc):
        mus = []
        logvars = []

        for group in range(self.num_groups):
            mu, logvar = self.enc_to_lat[group](x_enc)
            mus.append(mu)
            logvars.append(logvar)

        mu, logvar = torch.hstack(mus), torch.hstack(logvars)

        return mu, logvar

    def z_to_logits(self, z):
        z_dec = None

        for group in range(self.num_groups):
            start = group * self.num_latents_group
            end = (group * self.num_latents_group) + self.num_latents_group

            if z_dec is None:
                z_dec = self.z_to_z_dec(z[:, start:end], 0)
            else:
                z_dec = z_dec + self.z_to_z_dec(z[:, start:end], group)

        logits = self.z_dec_to_logits(z_dec)

        return z_dec, logits

    def z_to_z_dec(self, z, group):
        return self.lat_to_dec[group](z)

    def z_dec_to_logits(self, z_dec):
        return self.decoder(z_dec)
