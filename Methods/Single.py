import torch
import torch.optim as optim
from torchinfo import summary

from Method import Method


class Single(Method):
    def __init__(self,
                 architecture,
                 num_latents,
                 num_latents_group,
                 *,
                 step="Multiple",
                 size=28,
                 channels=1,
                 out_channels=None,
                 log_prob_fn="CB",
                 std=0.05):
        super().__init__(num_latents=num_latents, type="Single")

        self.num_latents_group = num_latents_group
        self.num_groups = num_latents // num_latents_group
        self.step = step
        self.size = size
        self.channels = channels
        self.log_prob_fn = log_prob_fn
        self.std = std

        self.encoder = architecture["Encoder"](num_latents, size, channels, out_channels)
        self.enc_to_lats = [architecture["EncoderToLatents"](num_latents, num_latents_group)
                            for _ in range(self.num_groups)]
        self.lats_to_dec = [architecture["LatentsToDecoder"](num_latents, num_latents_group)
                            for _ in range(self.num_groups)]
        self.decoder = architecture["Decoder"](num_latents, size, channels, out_channels)

        self.optimiser_encoder = optim.Adam(self.encoder.parameters(), lr=1e-3) # 0.001
        self.optimiser_enc_to_lats = [optim.Adam(x.parameters(), lr=1e-3) for x in self.enc_to_lats] # 0.001
        self.optimiser_lats_to_dec = [optim.Adam(x.parameters(), lr=1e-3) for x in self.lats_to_dec] # 0.001
        self.optimiser_decoder = optim.Adam(self.decoder.parameters(), lr=1e-3) # 0.001

    def train(self, i, data, *, get_grad=False):
        losses = []
        log_probs = []
        KLDs = []
        grads = []

        # Get the input images
        images, _ = data

        if self.step == "Single":
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

        for group in range(self.num_groups):
            if self.step == "Multiple":
                # Zero the parameter's gradients
                self.optimiser_encoder.zero_grad()
                self.optimiser_enc_to_lats[group].zero_grad()
                self.optimiser_lats_to_dec[group].zero_grad()
                self.optimiser_decoder.zero_grad()

                # Forward
                # Get the encoder output
                x_enc = self.encoder(images)

            # Forward
            mu = []
            logvar = []

            for j in range(group):
                with torch.no_grad():
                    mu_2, logvar_2 = self.enc_to_lats[j](x_enc)
                    mu.append(mu_2)
                    logvar.append(logvar_2)

            mu_3, logvar_3 = self.enc_to_lats[group](x_enc)
            mu.append(mu_3)
            logvar.append(logvar_3)

            mu = torch.cat(mu, dim=1)
            logvar = torch.cat(logvar, dim=1)

            # Reparameterise
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + (std * eps)

            batch_size = images.shape[0]
            z_dec = torch.zeros(batch_size, self.num_latents * 4)

            for j in range(group):
                with torch.no_grad():
                    start = j * self.num_latents_group
                    end = (j * self.num_latents_group) + self.num_latents_group
                    z_dec = z_dec + self.lats_to_dec[j](z[:, start:end])

            start = group * self.num_latents_group
            end = (group * self.num_latents_group) + self.num_latents_group
            z_dec = z_dec + self.lats_to_dec[group](z[:, start:end])

            logits = self.decoder(z_dec)

            # Loss, backward
            loss, log_prob, KLD = self.ELBO(
                logits,
                images.view(-1, self.channels * (self.size ** 2)),
                log_prob_fn=self.log_prob_fn,
                KLD_fn="N",
                mu=mu_3,
                logvar=logvar_3,
                std=self.std)
            # Because optimisers minimise, and we want to maximise the ELBO, we multiply it by -1
            loss = -loss
            if self.step == "Single":
                loss.backward(retain_graph=True)
            elif self.step == "Multiple":
                loss.backward()

            # Keep track of losses, log probs, and KLDs
            losses.append(loss.detach())
            log_probs.append(log_prob.detach())
            KLDs.append(KLD.detach())

            # Get the gradients
            if get_grad:
                grad = []

                for x in [self.encoder, self.enc_to_lats[group], self.lats_to_dec[group], self.decoder]:
                    for name, param in x.named_parameters():
                        grad.append(param.grad.abs().flatten())

                grads.append(torch.concat(grad).mean().item())

            if self.step == "Multiple":
                # Step
                self.optimiser_encoder.step()
                self.optimiser_enc_to_lats[group].step()
                self.optimiser_lats_to_dec[group].step()
                self.optimiser_decoder.step()

        if self.step == "Single":
            # Step
            self.optimiser_encoder.step()
            for x in self.optimiser_enc_to_lats:
                x.step()
            for x in self.optimiser_lats_to_dec:
                x.step()
            self.optimiser_decoder.step()

        return losses, log_probs, KLDs, grads

    @torch.no_grad()
    def test(self, i, data):
        # Get the input images
        images, _ = data

        x_enc, mu, logvar = self.x_to_mu_logvar(images)

        # The ELBO might be higher when testing because z is set to its expected value
        z = mu

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
        loss, log_prob, KLD = self.ELBO(
            logits,
            images.view(-1, self.channels * (self.size ** 2)),
            log_prob_fn=self.log_prob_fn,
            KLD_fn="N",
            mu=mu,
            logvar=logvar,
            std=self.std)

        return output, [-loss.item()], [log_prob.item()], [KLD.item()]

    def save(self, path):
        torch.save(self.encoder.state_dict(), f"{path}.pth")
        for i, x in enumerate(self.enc_to_lats):
            torch.save(x.state_dict(), f"{path}_enc_to_lat_{i}.pth")
        for i, x in enumerate(self.lats_to_dec):
            torch.save(x.state_dict(), f"{path}_lat_to_dec_{i}.pth")
        torch.save(self.decoder.state_dict(), f"{path}_dec.pth")

    def load(self, path):
        self.encoder.load_state_dict(torch.load(f"{path}.pth"))
        for i, x in enumerate(self.enc_to_lats):
            x.load_state_dict(torch.load(f"{path}_enc_to_lat_{i}.pth"))
        for i, x in enumerate(self.lats_to_dec):
            x.load_state_dict(torch.load(f"{path}_lat_to_dec_{i}.pth"))
        self.decoder.load_state_dict(torch.load(f"{path}_dec.pth"))

    def summary(self):
        summaries = []
        summaries.append("Encoder")
        summaries.append(str(summary(self.encoder)))
        for i, x in enumerate(self.enc_to_lats):
            summaries.append(f"Encoder to Latents {i}")
            summaries.append(str(summary(x)))
        for i, x in enumerate(self.lats_to_dec):
            summaries.append(f"Latents to Decoder {i}")
            summaries.append(str(summary(x)))
        summaries.append("Decoder")
        summaries.append(str(summary(self.decoder)))

        return summaries

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
        z_dec = self.z_to_z_dec(z[:, 0:self.num_latents_group], 0)

        for i in range(1, self.num_groups):
            start = i * self.num_latents_group
            end = (i * self.num_latents_group) + self.num_latents_group
            z_dec = z_dec + self.z_to_z_dec(z[:, start:end], i)

        logits = self.z_dec_to_logits(z_dec)

        return z_dec, logits

    @torch.no_grad()
    def z_to_z_dec(self, z, group):
        return self.lats_to_dec[group](z)

    @torch.no_grad()
    def z_dec_to_logits(self, z_dec):
        return self.decoder(z_dec)
