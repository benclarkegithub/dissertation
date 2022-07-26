import torch
import torch.optim as optim
from torch.distributions import Normal
from torch.functional import F
from torchinfo import summary

from Method import Method


class Quotient(Method):
    def __init__(self,
                 architecture,
                 num_latents,
                 num_latents_group,
                 *,
                 size=28,
                 channels=1,
                 out_channels=None,
                 log_prob_fn="CB",
                 std=0.05):
        super().__init__(num_latents=num_latents, type="Single")

        self.num_latents_group = num_latents_group
        self.num_groups = num_latents // num_latents_group
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

        zs = None
        log_ps = None
        log_qs = None
        outputs = []

        for group in range(self.num_groups):
            # Forward
            mu_1, logvar_1 = self.enc_to_lats[group](x_enc)

            # For Z>1 get the encoder output of the reconstruction
            if len(outputs) > 0:
                output_images = torch.sigmoid(outputs[-1])
                x_enc_rec = self.encoder(output_images)
                mu_2, logvar_2 = self.enc_to_lats[group](x_enc_rec)
                z, q = self.mu_logvar_to_z_and_quotient(mu_1, logvar_1, mu_2, logvar_2)
                zs = torch.cat([zs, z], dim=1)
                log_p = Normal(0, 1).log_prob(z)
                log_ps = torch.cat([log_ps, log_p], dim=1)
                log_qs = torch.cat([log_qs, q.log()], dim=1)
            else:
                mus = mu_1
                logvars = logvar_1
                std = torch.exp(0.5 * logvars)
                eps = torch.randn_like(std)
                z = mus + (std * eps)
                zs = z
                log_p = Normal(0, 1).log_prob(z)
                log_ps = log_p
                log_qs = Normal(mu_1, logvar_1.exp().sqrt()).log_prob(z)

            # Get z_dec
            z_dec = self.lats_to_dec[0](zs[:, 0:self.num_latents_group])

            for group_i in range(1, group+1):
                start = group_i * self.num_latents_group
                end = (group_i * self.num_latents_group) + self.num_latents_group
                z_dec = z_dec + self.lats_to_dec[group_i](zs[:, start:end])

            logits = self.decoder(z_dec)

            # Reshape logits and add to outputs
            logits_reshaped = logits.reshape(images.shape)
            outputs.append(logits_reshaped)

            # Loss, backward
            loss, log_prob, KLD = self.ELBO(
                logits,
                images.view(-1, self.channels * (self.size ** 2)),
                log_prob_fn=self.log_prob_fn,
                KLD_fn="Custom",
                log_p=log_ps,
                log_q=log_qs,
                std=self.std)
            # Because optimisers minimise, and we want to maximise the ELBO, we multiply it by -1
            loss = -loss
            loss.backward(retain_graph=True)

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

        # Forward
        # Get the encoder output
        x_enc = self.encoder(images)

        zs = None
        log_ps = None
        log_qs = None
        outputs = []

        for group in range(self.num_groups):
            # Forward
            mu_1, logvar_1 = self.enc_to_lats[group](x_enc)

            # For Z>1 get the encoder output of the reconstruction
            if len(outputs) > 0:
                output_images = torch.sigmoid(outputs[-1])
                x_enc_rec = self.encoder(output_images)
                mu_2, logvar_2 = self.enc_to_lats[group](x_enc_rec)
                z, q = self.mu_logvar_to_z_and_quotient(mu_1, logvar_1, mu_2, logvar_2)
                zs = torch.cat([zs, z], dim=1)
                log_p = Normal(0, 1).log_prob(z)
                log_ps = torch.cat([log_ps, log_p], dim=1)
                log_qs = torch.cat([log_qs, q.log()], dim=1)
            else:
                mus = mu_1
                logvars = logvar_1
                std = torch.exp(0.5 * logvars)
                eps = torch.randn_like(std)
                z = mus + (std * eps)
                zs = z
                log_p = Normal(0, 1).log_prob(z)
                log_ps = log_p
                log_qs = Normal(mu_1, logvar_1.exp().sqrt()).log_prob(z)

            # Can't use `z_to_logits` here because it assumes a fully-dimensional z
            z_dec = self.z_to_z_dec(zs[:, 0:self.num_latents_group], 0)

            for i in range(1, group+1):
                start = i * self.num_latents_group
                end = (i * self.num_latents_group) + self.num_latents_group
                z_dec = z_dec + self.z_to_z_dec(zs[:, start:end], i)

            logits = self.z_dec_to_logits(z_dec)

            # Reshape logits and add to outputs
            logits_reshaped = logits.reshape(images.shape)
            outputs.append(logits_reshaped)

        # Reshape the final output
        logits_reshaped = outputs[-1].reshape(-1, 28 * 28)

        output = {
            "x_enc": x_enc,
            "mu": mus,
            "logvar": logvars,
            # "z": z,
            "z_dec": z_dec,
            "logits": logits_reshaped
        }

        # Calculate loss
        loss, log_prob, KLD = self.ELBO(
            logits_reshaped,
            images.view(-1, self.channels * (self.size ** 2)),
            log_prob_fn=self.log_prob_fn,
            KLD_fn="Custom",
            log_p=log_ps,
            log_q=log_qs,
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
        pass

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

    # https://rstudio-pubs-static.s3.amazonaws.com/287838_7c982110ffe44d1eb5184739c5724926.html
    def mu_logvar_to_z_and_quotient(self, mu_1, logvar_1, mu_2, logvar_2):
        std_1, std_2 = logvar_1.exp().sqrt(), logvar_2.exp().sqrt()

        N_1, N_2 = Normal(mu_1, std_1), Normal(mu_2, std_2)
        z = N_1.rsample() / N_2.rsample()

        b = mu_1 / mu_2
        p = std_2 / std_1
        d_y = std_2 / mu_2

        return z, self.quotient_distribution(z, b, p, d_y)

    def quotient_distribution(self, z, b, p, d_y):
        frac_1 = p / (torch.pi * (1 + ((p ** 2) * (z ** 2))))

        frac_2_sum_1 = torch.exp(-(((p ** 2) * (b ** 2)) + 1) / (2 * (d_y ** 2)))

        q = (1 + (b * (p ** 2) * z)) / (d_y * torch.sqrt(1 + ((p ** 2) * (z ** 2))))
        frac_2_sum_2_1 = torch.sqrt(torch.tensor(torch.pi / 2)) * q * torch.erf(q / torch.sqrt(torch.tensor(2)))
        frac_2_sum_2_2 = torch.exp(-((p ** 2) * ((z - b) ** 2)) / (2 * (d_y ** 2) * (1 + ((p ** 2) * (z ** 2)))))
        frac_2_sum_2 = frac_2_sum_2_1 * frac_2_sum_2_2

        frac_2 = frac_2_sum_1 + frac_2_sum_2

        frac = frac_1 * frac_2

        return frac
