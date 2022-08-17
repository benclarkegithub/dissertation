import torch
import torch.optim as optim
from torch.distributions import Normal
from torchinfo import summary

from Method import Method
from Architectures.Quotient import Quotient as Quotient2


"""
Options

encoder_to_latents
    True:   Many sets of parameters for encoding to latent variable(s)
    False:  One set of parameters for encoding to latent variable(s)
    
resample:
    True:   Resample z at each step
    False:  Do not resample z at each step (reuse previous zs)
    
learnable_KL:
    True:   Learnable KL parameters N(a, b) / N(mu_rec, std_rec)
    False:  Standard Normal KL parameters N(0, 1) / N(mu_rec, std_rec)
"""
class Quotient(Method):
    def __init__(self,
                 architecture,
                 num_latents,
                 num_latents_group,
                 # Options
                 encoder_to_latents,
                 resample,
                 learnable_KL,
                 *,
                 learning_rate=1e-4,
                 size=28,
                 channels=1,
                 out_channels=None,
                 log_prob_fn="CB",
                 beta=1,
                 std=0.05,
                 hidden_size=None,
                 clip=None):
        super().__init__(
            num_latents=num_latents,
            type="Single",
            learning_rate=learning_rate,
            size=size,
            channels=channels,
            out_channels=out_channels,
            log_prob_fn=log_prob_fn,
            beta=beta,
            std=std,
            hidden_size=hidden_size)

        self.num_latents_group = num_latents_group
        self.num_groups = num_latents // num_latents_group
        self.clip = clip

        # Options
        self.encoder_to_latents = encoder_to_latents
        self.resample = resample
        self.learnable_KL = learnable_KL

        self.canvas = architecture["Canvas"](size, channels)
        self.KL = None
        if self.learnable_KL:
            self.KL = architecture["KL"]()
        self.encoder = architecture["Encoder"](size, channels, self.hidden_size, out_channels)
        self.enc_to_lat = None
        self.enc_to_lats = None
        if not self.encoder_to_latents:
            self.enc_to_lat = architecture["EncoderToLatents"](self.hidden_size, num_latents_group)
        else:
            self.enc_to_lats = [
                architecture["EncoderToLatents"](self.hidden_size, num_latents_group) for _ in range(self.num_groups)]
        self.lats_to_dec = [
            architecture["LatentsToDecoder"](self.hidden_size, num_latents_group) for _ in range(self.num_groups)]
        self.decoder = architecture["Decoder"](self.hidden_size, size, channels, out_channels)

        self.model = Quotient2(
            self.canvas,
            self.encoder,
            self.lats_to_dec,
            self.decoder,
            KL=self.KL,
            enc_to_lat=self.enc_to_lat,
            enc_to_lats=self.enc_to_lats)
        self.optimiser_model = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

    def train(self, i, data, *, get_grad=True):
        losses = []
        log_probs = []
        KLDs = []
        grads = []

        # Get the input images
        images, _ = data

        # Zero the parameter's gradients
        self.model.zero_grad()

        # Forward
        # Get encoder output of images
        x_enc = self.encoder(images)
        if not self.encoder_to_latents:
            mu_images, logvar_images = self.enc_to_lat(x_enc)
        else:
            mu_images, logvar_images = [], []

            for enc_to_lat in self.enc_to_lats:
                mu_images_1, logvar_images_1 = enc_to_lat(x_enc)
                mu_images.append(mu_images_1)
                logvar_images.append(logvar_images_1)

            mu_images, logvar_images = torch.cat(mu_images, dim=1), torch.cat(logvar_images, dim=1)

        # Get the blank canvas
        ones = torch.ones(images.shape[0], 1)
        canvas = self.canvas(ones)
        canvas_reshaped = canvas.reshape(images.shape)

        mu = None
        logvar = None
        outputs = [canvas_reshaped]
        # Used for "NoResample" and "Both" variants
        zs = None
        log_qs = None
        log_ps = None

        for group in range(self.num_groups):
            # Get encoder output of reconstruction
            output_images = torch.sigmoid(outputs[-1])
            x_enc_rec = self.encoder(output_images)
            if not self.encoder_to_latents:
                mu_rec, logvar_rec = self.enc_to_lat(x_enc_rec)
            else:
                mu_rec, logvar_rec = self.enc_to_lats[group](x_enc_rec)
            mu = torch.cat([mu, mu_rec], dim=1) if mu is not None else mu_rec
            logvar = torch.cat([logvar, logvar_rec], dim=1) if logvar is not None else logvar_rec

            # Reparameterise, get log_q and log_p
            if not self.encoder_to_latents:
                mu_images_temp = mu_images
                logvar_images_temp = logvar_images
            else:
                end = (group * self.num_latents_group) + self.num_latents_group
                mu_images_temp = mu_images[:, :end]
                logvar_images_temp = logvar_images[:, :end]
            z, log_q = self.mu_logvar_to_z_and_log_q(mu_images_temp, logvar_images_temp, mu, logvar)
            log_p = self.z_mu_logvar_to_log_p(z, mu, logvar)
            if not self.resample:
                start = group * self.num_latents_group
                end = (group * self.num_latents_group) + self.num_latents_group
                zs = torch.cat([zs, z[:, start:end]], dim=1) if zs is not None else z
                log_qs = torch.cat([log_qs, log_q[:, start:end]], dim=1) if log_qs is not None else log_q
                log_ps = torch.cat([log_ps, log_p[:, start:end]], dim=1) if log_ps is not None else log_p

            # Get z_dec
            z_temp = z if self.resample else zs
            z_dec = self.lats_to_dec[0](z_temp[:, 0:self.num_latents_group])

            for group_i in range(1, group+1):
                start = group_i * self.num_latents_group
                end = (group_i * self.num_latents_group) + self.num_latents_group
                z_dec = z_dec + self.lats_to_dec[group_i](z_temp[:, start:end])

            logits = self.decoder(z_dec)

            # Reshape logits and add to outputs
            logits_reshaped = logits.reshape(images.shape)
            outputs.append(logits_reshaped.detach())

            # Loss, backward
            log_p_temp = log_p if self.resample else log_ps
            log_q_temp = log_q if self.resample else log_qs
            loss, log_prob, KLD = self.ELBO(
                logits,
                images.view(-1, self.channels * (self.size ** 2)),
                log_prob_fn=self.log_prob_fn,
                KLD_fn="Custom",
                log_p=log_p_temp,
                log_q=log_q_temp,
                beta=self.beta,
                std=self.std)
            # Because optimisers minimise, and we want to maximise the ELBO, we multiply it by -1
            loss = -loss
            loss.backward(retain_graph=True)

            # Keep track of losses, log probs, and KLDs
            losses.append(loss.detach())
            log_probs.append(log_prob.detach())
            KLDs.append(KLD.detach())

        # Clip the gradients
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        # Get the gradients
        if get_grad:
            grad_temp = []

            # Components that are the same for each group
            components = [self.canvas, self.encoder, self.decoder]
            components += self.lats_to_dec
            components += [self.KL] if self.KL is not None else []
            components += [self.enc_to_lat] if self.enc_to_lat is not None else []
            # Components that are the unique for each group
            components += [
                self.enc_to_lats[j] for j in range(self.num_groups)] if self.enc_to_lats is not None else []

            for x in components:
                for name, param in x.named_parameters():
                    if param.grad is not None:
                        grad_temp.append(param.grad.flatten())

            grads.append(torch.linalg.norm(torch.cat(grad_temp)))

        # Step
        self.optimiser_model.step()

        # Fix KLDs
        KLDs_temp = KLDs.copy()
        KLDs_temp.insert(0, 0)
        KLDs = [KLDs[i] - KLDs_temp[i] for i in range(len(KLDs))]

        return losses, log_probs, KLDs, grads

    @torch.no_grad()
    def test(self, i, data):
        # Get the input images
        images, _ = data

        # Forward
        # Get encoder output of images
        x_enc = self.encoder(images)
        if not self.encoder_to_latents:
            mu_images, logvar_images = self.enc_to_lat(x_enc)
        else:
            mu_images, logvar_images = [], []

            for enc_to_lat in self.enc_to_lats:
                mu_images_1, logvar_images_1 = enc_to_lat(x_enc)
                mu_images.append(mu_images_1)
                logvar_images.append(logvar_images_1)

            mu_images, logvar_images = torch.cat(mu_images, dim=1), torch.cat(logvar_images, dim=1)

        # Get the blank canvas
        ones = torch.ones(images.shape[0], 1)
        canvas = self.canvas(ones)
        canvas_reshaped = canvas.reshape(images.shape)

        mu = None
        logvar = None
        outputs = [canvas_reshaped]
        z_dec = None
        # Used for "NoResample" and "Both" variants
        zs = None
        log_q = None
        log_p = None
        log_qs = None
        log_ps = None

        for group in range(self.num_groups):
            # Get encoder output of reconstruction
            output_images = torch.sigmoid(outputs[-1])
            x_enc_rec = self.encoder(output_images)
            if not self.encoder_to_latents:
                mu_rec, logvar_rec = self.enc_to_lat(x_enc_rec)
            else:
                mu_rec, logvar_rec = self.enc_to_lats[group](x_enc_rec)
            mu = torch.cat([mu, mu_rec], dim=1) if mu is not None else mu_rec
            logvar = torch.cat([logvar, logvar_rec], dim=1) if logvar is not None else logvar_rec

            # Reparameterise, get log_q and log_p
            if not self.encoder_to_latents:
                mu_images_temp = mu_images
                logvar_images_temp = logvar_images
            else:
                end = (group * self.num_latents_group) + self.num_latents_group
                mu_images_temp = mu_images[:, :end]
                logvar_images_temp = logvar_images[:, :end]
            z, log_q = self.mu_logvar_to_z_and_log_q(mu_images_temp, logvar_images_temp, mu, logvar)
            log_p = self.z_mu_logvar_to_log_p(z, mu, logvar)
            if not self.resample:
                start = group * self.num_latents_group
                end = (group * self.num_latents_group) + self.num_latents_group
                zs = torch.cat([zs, z[:, start:end]], dim=1) if zs is not None else z
                log_qs = torch.cat([log_qs, log_q[:, start:end]], dim=1) if log_qs is not None else log_q
                log_ps = torch.cat([log_ps, log_p[:, start:end]], dim=1) if log_ps is not None else log_p
            else:
                zs = z

            # Can't use `z_to_logits` here because it assumes a fully-dimensional z
            z_temp = z if self.resample else zs
            z_dec = self.z_to_z_dec(z_temp[:, 0:self.num_latents_group], 0)

            for i in range(1, group+1):
                start = i * self.num_latents_group
                end = (i * self.num_latents_group) + self.num_latents_group
                z_dec = z_dec + self.z_to_z_dec(z_temp[:, start:end], i)

            logits = self.z_dec_to_logits(z_dec)

            # Reshape logits and add to outputs
            logits_reshaped = logits.reshape(images.shape)
            outputs.append(logits_reshaped)

        # Reshape the final output
        logits_reshaped = outputs[-1].reshape(-1, self.channels * (self.size ** 2))

        # THIS IS WRONG, WON'T WORK AS EXPECTED!
        output = {
            "x_enc": x_enc,
            "mu": mu,
            "logvar": logvar,
            "z": zs,
            "z_dec": z_dec,
            "logits": logits_reshaped
        }

        # Calculate loss
        log_p_temp = log_p if self.resample else log_ps
        log_q_temp = log_q if self.resample else log_qs
        loss, log_prob, KLD = self.ELBO(
            logits_reshaped,
            images.view(-1, self.channels * (self.size ** 2)),
            log_prob_fn=self.log_prob_fn,
            KLD_fn="Custom",
            log_p=log_p_temp,
            log_q=log_q_temp,
            beta=self.beta,
            std=self.std)

        return output, [-loss.item()], [log_prob.item()], [KLD.item()]

    def save(self, path):
        torch.save(self.canvas.state_dict(), f"{path}_canvas.pth")
        torch.save(self.encoder.state_dict(), f"{path}.pth")
        if not self.encoder_to_latents:
            torch.save(self.enc_to_lat.state_dict(), f"{path}_enc_to_lat.pth")
        else:
            for i, x in enumerate(self.enc_to_lats):
                torch.save(x.state_dict(), f"{path}_enc_to_lats_{i}.pth")
        for i, x in enumerate(self.lats_to_dec):
            torch.save(x.state_dict(), f"{path}_lat_to_dec_{i}.pth")
        torch.save(self.decoder.state_dict(), f"{path}_dec.pth")

    def load(self, path):
        self.canvas.load_state_dict(torch.load(f"{path}_canvas.pth"))
        self.encoder.load_state_dict(torch.load(f"{path}.pth"))
        if not self.encoder_to_latents:
            self.enc_to_lat.load_state_dict(torch.load(f"{path}_enc_to_lat.pth"))
        else:
            for i, x in enumerate(self.enc_to_lats):
                x.load_state_dict(torch.load(f"{path}_enc_to_lats_{i}.pth"))
        for i, x in enumerate(self.lats_to_dec):
            x.load_state_dict(torch.load(f"{path}_lat_to_dec_{i}.pth"))
        self.decoder.load_state_dict(torch.load(f"{path}_dec.pth"))

    def summary(self):
        return str(summary(self.model, verbose=False))

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

    def mu_logvar_to_z_and_log_q(self, mu_1, logvar_1, mu_2, logvar_2):
        # N(mu_1, std_1) / N(mu_2, std_2)
        samples = mu_2.shape[1] // mu_1.shape[1]
        mu_1_repeated, logvar_1_repeated = mu_1.repeat(1, samples), logvar_1.repeat(1, samples)
        std_1_repeated, std_2 = logvar_1_repeated.exp().sqrt(), logvar_2.exp().sqrt()

        N_1, N_2 = Normal(mu_1_repeated, std_1_repeated), Normal(mu_2, std_2)
        z = N_1.rsample() / N_2.rsample()

        q = self.z_mu_std_to_quotient_prob(z, mu_1_repeated, std_1_repeated, mu_2, std_2)
        log_q = q.log()

        return z, log_q

    def z_mu_logvar_to_log_p(self, z, mu, logvar):
        # For KL divergence
        mu_temp, std_temp = None, None
        if not self.learnable_KL:
            # N(0, 1) / N(mu_rec, std_rec)
            mu_temp, std_temp = torch.zeros(mu.shape), torch.ones(logvar.shape)
        else:
            # N(a, b) / N(mu_rec, std_rec)
            # Get the mu and logvar for KL divergence
            one = torch.tensor([1.0])
            mu, logvar = self.KL(one)
            mu_temp, std_temp = mu, logvar.exp().sqrt()

        std = logvar.exp().sqrt()

        p = self.z_mu_std_to_quotient_prob(z, mu_temp, std_temp, mu, std)
        log_p = p.log()

        return log_p

    def z_mu_std_to_quotient_prob(self, z, mu_1, std_1, mu_2, std_2):
        # N(mu_1, std_1) / N(mu_2, std_2)
        b = mu_1 / mu_2
        p = std_2 / std_1
        d_y = std_2 / mu_2

        quotient_prob = self.quotient_distribution(z, b, p, d_y)

        return quotient_prob

    # https://rstudio-pubs-static.s3.amazonaws.com/287838_7c982110ffe44d1eb5184739c5724926.html
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
