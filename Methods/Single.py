from os.path import exists
import torch
import torch.optim as optim

from VAE import Encoder, EncoderToLatents, LatentsToDecoder, Decoder
from Method import Method


class Single(Method):
    def __init__(self, num_latents, num_latents_group):
        super().__init__(num_latents=num_latents)

        self.num_latents_group = num_latents_group
        self.num_groups = num_latents // num_latents_group

        self.encoder = Encoder()
        self.enc_to_lats = [EncoderToLatents(num_latents_group) for _ in range(self.num_groups)]
        self.lats_to_dec = [LatentsToDecoder(num_latents_group) for _ in range(self.num_groups)]
        self.decoder = Decoder()

        self.optimiser_encoder = optim.Adam(self.encoder.parameters(), lr=1e-3) # 0.001
        self.optimiser_enc_to_lats = [optim.Adam(x.parameters(), lr=1e-3) for x in self.enc_to_lats] # 0.001
        self.optimiser_lats_to_dec = [optim.Adam(x.parameters(), lr=1e-3) for x in self.lats_to_dec] # 0.001
        self.optimiser_decoder = optim.Adam(self.decoder.parameters(), lr=1e-3) # 0.001

    def train(self, i, data):
        total_loss = torch.tensor(0.0, requires_grad=False)
        total_log_prob = torch.tensor(0.0, requires_grad=False)
        total_KLD = torch.tensor(0.0, requires_grad=False)

        # Get the input images
        images, _ = data

        # Zero the parameter's gradients
        self.optimiser_encoder.zero_grad()
        for x in self.optimiser_enc_to_lats:
            x.zero_grad()
        for x in self.optimiser_lats_to_dec:
            x.zero_grad()
        self.optimiser_decoder.zero_grad()

        # Forward
        # Get the encoder output
        x_enc = self.encoder(images)

        for i in range(self.num_groups):
            # Forward
            mu = []
            logvar = []

            for j in range(i):
                with torch.no_grad():
                    mu_2, logvar_2 = self.enc_to_lats[j](x_enc)
                    mu.append(mu_2)
                    logvar.append(logvar_2)

            mu_3, logvar_3 = self.enc_to_lats[i](x_enc)
            mu.append(mu_3)
            logvar.append(logvar_3)

            mu = torch.cat(mu, dim=1)
            logvar = torch.cat(logvar, dim=1)

            # Reparameterise
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + (std * eps)

            batch_size = images.shape[0]
            z_dec = torch.zeros(batch_size, 48)

            for j in range(i):
                with torch.no_grad():
                    z_dec = z_dec + self.lats_to_dec[j](z[:, j, None])

            z_dec = self.lats_to_dec[i](z[:, i, None])

            logits = self.decoder(z_dec)

            # Loss, backwards
            loss, log_prob, KLD = self.ELBO(logits, images.view(-1, 28 * 28), mu, logvar)
            # Because optimisers minimise, and we want to maximise the ELBO, we multiply it by -1
            loss = -loss
            loss.backward(retain_graph=True)
            # Total
            total_loss = total_loss + loss
            total_log_prob = total_log_prob + log_prob
            total_KLD = total_KLD + KLD

        # Step
        self.optimiser_encoder.step()
        for x in self.optimiser_enc_to_lats:
            x.step()
        for x in self.optimiser_lats_to_dec:
            x.step()
        self.optimiser_decoder.step()

        return total_loss.item(), total_log_prob.item(), total_KLD.item()

    @torch.no_grad()
    def test(self, i, data):
        # Get the input images
        images, _ = data

        x_enc, mu, logvar = self.x_to_mu_logvar(images)

        # Should we reparameterise?
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + (std * eps)

        z_dec, logits = self.z_to_logits(z)

        output = {
            "x_enc": x_enc,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_dec": z_dec,
            "logits": logits
        }

        # Calculate loss
        loss, log_prob, KLD = self.ELBO(logits, images.view(-1, 28 * 28), mu, logvar)

        return output, -loss.item(), log_prob.item(), KLD.item()

    def save(self, path):
        torch.save(self.encoder.state_dict(), f"{path}_enc.pth")
        for i, x in enumerate(self.enc_to_lats):
            torch.save(x.state_dict(), f"{path}_enc_to_lat_{i}.pth")
        for i, x in enumerate(self.lats_to_dec):
            torch.save(x.state_dict(), f"{path}_lat_to_dec_{i}.pth")
        torch.save(self.decoder.state_dict(), f"{path}_dec.pth")

    def load(self, path):
        self.encoder.load_state_dict(torch.load(f"{path}_enc.pth"))
        for i, x in enumerate(self.enc_to_lats):
            x.load_state_dict(torch.load(f"{path}_enc_to_lat_{i}.pth"))
        for i, x in enumerate(self.lats_to_dec):
            x.load_state_dict(torch.load(f"{path}_lat_to_dec_{i}.pth"))
        self.decoder.load_state_dict(torch.load(f"{path}_dec.pth"))

    @torch.no_grad()
    def x_to_mu_logvar(self, x):
        # Get the encoder output
        x_enc = self.encoder(x)

        mu = []
        logvar = []

        for i in range(self.num_groups):
            mu_2, logvar_2 = self.enc_to_lats[i](x_enc)
            mu.append(mu_2)
            logvar.append(logvar_2)

        mu = torch.cat(mu, dim=1)
        logvar = torch.cat(logvar, dim=1)

        return x_enc, mu, logvar

    @torch.no_grad()
    def z_to_logits(self, z):
        z_dec = self.lats_to_dec[0](z[:, 0, None])

        for i in range(1, self.num_groups):
            z_dec = z_dec + self.lats_to_dec[i](z[:, i, None])

        logits = self.decoder(z_dec)

        return z_dec, logits
