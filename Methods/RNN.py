import torch
import torch.optim as optim
from torchinfo import summary

from Method import Method
from Architectures.RNN import RNN as RNN2


"""
Options

reconstruction
    True:   Use the reconstruction x̂
    False:  Do not use the reconstruction x̂

encoders
    True:   Use two encoders, one for the images and one for the reconstruction
    False:  Use one encoder, for both the images and the reconstruction
    
to_latents
    "Encoder":          p(z | x) if reconstruction is False else p(z | x, x̂)
    "EncoderLatents":   p(z | x, z_1, …, z_n-1) if reconstruction is False else p(z | x, x̂, z_1, …, z_n-1)
    "Latents":          p(z | z_a, z_b) where z_a = z | x, z_b = z | x̂
            
encoder_to_latents
    True:   Many sets of parameters for encoding to latent variable(s)
    False:  One set of parameters for encoding to latent variable(s)
    
backprop
    True:   Backpropagate through Z_<=n at each step
    False:  Only backpropagate through Z_n at each step
    
resample:
    True:   Resample z at each step
    False:  Do not resample z at each step (reuse previous zs)
"""
class RNN(Method):
    def __init__(self,
                 architecture,
                 num_latents,
                 num_latents_group,
                 # Options
                 reconstruction,
                 encoders,
                 to_latents,
                 encoder_to_latents,
                 backprop,
                 resample,
                 *,
                 type="Multiple",
                 learning_rate=1e-3,
                 size=28,
                 channels=1,
                 out_channels=None,
                 log_prob_fn="CB",
                 std=0.05,
                 hidden_size=None,
                 clip=None):
        super().__init__(num_latents=num_latents, type=type)

        self.num_latents_group = num_latents_group
        self.num_groups = num_latents // num_latents_group
        self.type = type
        self.size = size
        self.channels = channels
        self.log_prob_fn = log_prob_fn
        self.std = std
        self.hidden_size = hidden_size if hidden_size is not None else channels * (size ** 2) // 8
        self.clip = clip
        # Options
        self.reconstruction = reconstruction
        self.encoders = encoders
        self.to_latents = to_latents
        self.encoder_to_latents = encoder_to_latents
        self.backprop = backprop
        self.resample = resample

        # Canvas
        self.canvas = architecture["Canvas"](size, channels).to(self.device)

        # Encoder
        self.encoder = architecture["Encoder"](size, channels, self.hidden_size, out_channels).to(self.device)

        # Encoder 2
        self.encoder_2 = None
        if self.encoders:
            self.encoder_2 = architecture["Encoder"](size, channels, self.hidden_size, out_channels).to(self.device)

        # Encoder Encoder to Encoder
        self.enc_enc_to_enc = None
        if self.reconstruction and (self.to_latents == "Encoder"):
            self.enc_enc_to_enc = architecture["EncoderEncoderToEncoder"](self.hidden_size).to(self.device)

        # Encoder to Latents
        self.enc_to_lat = None
        self.enc_to_lats = None
        self.enc_lats_to_lats = None
        if (self.to_latents == "Encoder") or (self.to_latents == "Latents"):
            if not self.encoder_to_latents:
                self.enc_to_lat = architecture["EncoderToLatents"](self.hidden_size, num_latents_group).to(self.device)
            else:
                self.enc_to_lats = [
                    architecture["EncoderToLatents"](self.hidden_size, num_latents_group).to(self.device)
                    for _ in range(self.num_groups)]

        # Encoder Latents to Latents
        elif self.to_latents == "EncoderLatents":
            hidden_size_temp = self.hidden_size if not self.reconstruction else self.hidden_size * 2

            self.enc_lats_to_lats = [
                architecture["EncoderLatentsToLatents"](hidden_size_temp, group, num_latents_group).to(self.device)
                for group in range(self.num_groups)]

        # Latents to Latents
        self.lats_to_lats = None
        if self.to_latents == "Latents":
            self.lats_to_lats = architecture["LatentsToLatents"](num_latents_group).to(self.device)

        # Latents to Decoder
        self.lats_to_dec = [
            architecture["LatentsToDecoder"](self.hidden_size, num_latents_group).to(self.device)
            for _ in range(self.num_groups)]

        # Decoder
        self.decoder = architecture["Decoder"](self.hidden_size, size, channels, out_channels).to(self.device)

        # Model
        self.model = RNN2(
            self.canvas,
            self.encoder,
            self.lats_to_dec,
            self.decoder,
            encoder_2=self.encoder_2,
            enc_enc_to_enc=self.enc_enc_to_enc,
            enc_to_lat=self.enc_to_lat,
            enc_to_lats=self.enc_to_lats,
            enc_lats_to_lats=self.enc_lats_to_lats,
            lats_to_lats=self.lats_to_lats)
        self.optimiser_model = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

    def train(self, i, data, *, get_grad=False, model=None):
        losses = []
        log_probs = []
        KLDs = []
        grads = []

        # Get the input images
        images, _ = data
        images = images.to(self.device)

        # Zero the parameter's gradients
        self.optimiser_model.zero_grad()

        # Forward
        # Get encoder output of images
        x_enc_images = self.encoder(images)

        mu_images = None
        logvar_images = None
        if (not self.reconstruction and (self.to_latents == "Encoder")) or (self.to_latents == "Latents"):
            # Get the parameters of the distribution of the images for either the EncoderToEncoder or the
            # LatentsToLatents component.
            if not self.encoder_to_latents:
                mu_images, logvar_images = self.enc_to_lat(x_enc_images)
            else:
                mu_images, logvar_images = [], []

                for enc_to_lat in self.enc_to_lats:
                    mu_images_1, logvar_images_1 = enc_to_lat(x_enc_images)
                    mu_images.append(mu_images_1)
                    logvar_images.append(logvar_images_1)

                mu_images, logvar_images = torch.cat(mu_images, dim=1), torch.cat(logvar_images, dim=1)

        outputs = None
        if self.reconstruction:
            # Get the blank canvas
            ones = torch.ones(images.shape[0], 1, device=self.device)
            canvas = self.canvas(ones)
            canvas_reshaped = canvas.reshape(images.shape)
            outputs = [canvas_reshaped]

        mu = None
        logvar = None
        # Used when resample is False
        zs = None

        if self.type == "Single":
            model = self.num_groups - 1

        for group in range(model + 1):
            mu_2, logvar_2 = None, None
            start = group * self.num_latents_group
            end = (group * self.num_latents_group) + self.num_latents_group

            if not self.reconstruction:
                if self.to_latents == "Encoder":
                    if not self.encoder_to_latents:
                        mu_2, logvar_2 = mu_images, logvar_images
                    else:
                        mu_2, logvar_2 = mu_images[:, start:end], logvar_images[:, start:end]
                elif self.to_latents == "EncoderLatents":
                    mu_temp = mu.detach() if mu is not None else None
                    logvar_temp = logvar.detach() if logvar is not None else None
                    mu_2, logvar_2 = self.enc_lats_to_lats[group](x_enc_images, mu_temp, logvar_temp)
            else:
                # Get the encoder output of the reconstruction
                output_images = torch.sigmoid(outputs[-1])
                if not self.encoders:
                    x_enc_rec = self.encoder(output_images)
                else:
                    x_enc_rec = self.encoder_2(output_images)

                mu_1, logvar_1 = None, None
                if (self.to_latents == "Encoder") or (self.to_latents == "Latents"):
                    x_enc = None
                    if (self.to_latents == "Encoder"):
                        x_enc = self.enc_enc_to_enc(x_enc_images, x_enc_rec)
                    elif (self.to_latents == "Latents"):
                        x_enc = x_enc_rec

                    if not self.encoder_to_latents:
                        mu_1, logvar_1 = self.enc_to_lat(x_enc)
                    else:
                        mu_1, logvar_1 = self.enc_to_lats[group](x_enc)

                if self.to_latents == "Encoder":
                    mu_2, logvar_2 = mu_1, logvar_1
                elif self.to_latents == "EncoderLatents":
                    x = torch.cat([x_enc_images, x_enc_rec], dim=1)
                    mu_temp = mu.detach() if mu is not None else None
                    logvar_temp = logvar.detach() if logvar is not None else None
                    mu_2, logvar_2 = self.enc_lats_to_lats[group](x, mu_temp, logvar_temp)
                elif self.to_latents == "Latents":
                    if not self.encoder_to_latents:
                        mu_2, logvar_2 = self.lats_to_lats(mu_images, logvar_images, mu_1, logvar_1)
                    else:
                        mu_2, logvar_2 = self.lats_to_lats(
                            mu_images[:, start:end], logvar_images[:, start:end], mu_1, logvar_1)

            # Detach mu, logvar for Z_<n if backprop == False
            if self.backprop:
                mu = torch.cat([mu, mu_2], dim=1) if mu is not None else mu_2
                logvar = torch.cat([logvar, logvar_2], dim=1) if logvar is not None else logvar_2
            else:
                mu = torch.cat([mu.detach(), mu_2], dim=1) if mu is not None else mu_2
                logvar = torch.cat([logvar.detach(), logvar_2], dim=1) if logvar is not None else logvar_2

            # Reparameterise
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + (std * eps)

            if not self.resample:
                zs = torch.cat([zs, z[:, start:end]], dim=1) if zs is not None else z

            # Get z_dec
            z_temp = z if self.resample else zs
            z_dec = self.lats_to_dec[0](z_temp[:, 0:self.num_latents_group])

            for group_i in range(1, group + 1):
                start = group_i * self.num_latents_group
                end = (group_i * self.num_latents_group) + self.num_latents_group
                z_dec = z_dec + self.lats_to_dec[group_i](z_temp[:, start:end])

            logits = self.decoder(z_dec)

            if self.reconstruction:
                # Reshape logits and add to outputs
                logits_reshaped = logits.reshape(images.shape)
                outputs.append(logits_reshaped.detach())

            # Loss, backward
            loss, log_prob, KLD = self.ELBO(
                logits,
                images.view(-1, self.channels * (self.size ** 2)),
                log_prob_fn=self.log_prob_fn,
                KLD_fn="N",
                mu=mu,
                logvar=logvar,
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
            components += [self.encoder_2] if self.encoder_2 is not None else []
            components += [self.enc_enc_to_enc] if self.enc_enc_to_enc is not None else []
            components += [self.enc_to_lat] if self.enc_to_lat is not None else []
            components += [self.lats_to_lats] if self.lats_to_lats is not None else []
            # Components that are the unique for each group
            components += [
                self.enc_to_lats[j] for j in range(self.num_groups)] if self.enc_to_lats is not None else []
            components += [
                self.enc_lats_to_lats[j] for j in range(self.num_groups)] if self.enc_lats_to_lats is not None else []

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
        KLDs = [KLDs[j] - KLDs_temp[j] for j in range(len(KLDs))]

        if self.type == "Single":
            return losses, log_probs, KLDs, grads
        else:
            return [losses[-1]], [log_probs[-1]], [KLDs[-1]], grads

    @torch.no_grad()
    def test(self, i, data, *, model=None):
        # Get the input images
        images, _ = data
        images = images.to(self.device)

        # Forward
        # Get encoder output of images
        x_enc_images = self.encoder(images)

        mu_images = None
        logvar_images = None
        if (not self.reconstruction and (self.to_latents == "Encoder")) or (self.to_latents == "Latents"):
            # Get the parameters of the distribution of the images for either the EncoderToEncoder or the
            # LatentsToLatents component.
            if not self.encoder_to_latents:
                mu_images, logvar_images = self.enc_to_lat(x_enc_images)
            else:
                mu_images, logvar_images = [], []

                for enc_to_lat in self.enc_to_lats:
                    mu_images_1, logvar_images_1 = enc_to_lat(x_enc_images)
                    mu_images.append(mu_images_1)
                    logvar_images.append(logvar_images_1)

                mu_images, logvar_images = torch.cat(mu_images, dim=1), torch.cat(logvar_images, dim=1)

        outputs = None
        if self.reconstruction:
            # Get the blank canvas
            ones = torch.ones(images.shape[0], 1, device=self.device)
            canvas = self.canvas(ones)
            canvas_reshaped = canvas.reshape(images.shape)
            outputs = [canvas_reshaped]

        mu = None
        logvar = None
        logits_output = None
        # Used when resample == False
        zs = None

        model_temp = model if model is not None else (self.num_groups - 1)

        for group in range(model_temp + 1):
            mu_2, logvar_2 = None, None
            start = group * self.num_latents_group
            end = (group * self.num_latents_group) + self.num_latents_group

            if not self.reconstruction:
                if self.to_latents == "Encoder":
                    if not self.encoder_to_latents:
                        mu_2, logvar_2 = mu_images, logvar_images
                    else:
                        mu_2, logvar_2 = mu_images[:, start:end], logvar_images[:, start:end]
                elif self.to_latents == "EncoderLatents":
                    mu_temp = mu if mu is not None else None
                    logvar_temp = logvar if logvar is not None else None
                    mu_2, logvar_2 = self.enc_lats_to_lats[group](x_enc_images, mu_temp, logvar_temp)
            else:
                # Get the encoder output of the reconstruction
                output_images = torch.sigmoid(outputs[-1])
                if not self.encoders:
                    x_enc_rec = self.encoder(output_images)
                else:
                    x_enc_rec = self.encoder_2(output_images)

                mu_1, logvar_1 = None, None
                if (self.to_latents == "Encoder") or (self.to_latents == "Latents"):
                    x_enc = None
                    if (self.to_latents == "Encoder"):
                        x_enc = self.enc_enc_to_enc(x_enc_images, x_enc_rec)
                    elif (self.to_latents == "Latents"):
                        x_enc = x_enc_rec

                    if not self.encoder_to_latents:
                        mu_1, logvar_1 = self.enc_to_lat(x_enc)
                    else:
                        mu_1, logvar_1 = self.enc_to_lats[group](x_enc)

                if self.to_latents == "Encoder":
                    mu_2, logvar_2 = mu_1, logvar_1
                elif self.to_latents == "EncoderLatents":
                    x = torch.cat([x_enc_images, x_enc_rec], dim=1)
                    mu_temp = mu if mu is not None else None
                    logvar_temp = logvar if logvar is not None else None
                    mu_2, logvar_2 = self.enc_lats_to_lats[group](x, mu_temp, logvar_temp)
                elif self.to_latents == "Latents":
                    if not self.encoder_to_latents:
                        mu_2, logvar_2 = self.lats_to_lats(mu_images, logvar_images, mu_1, logvar_1)
                    else:
                        mu_2, logvar_2 = self.lats_to_lats(
                            mu_images[:, start:end], logvar_images[:, start:end], mu_1, logvar_1)

            mu = torch.cat([mu, mu_2], dim=1) if mu is not None else mu_2
            logvar = torch.cat([logvar, logvar_2], dim=1) if logvar is not None else logvar_2

            # The ELBO might be higher when testing because z is set to its expected value
            z = mu

            if not self.resample:
                zs = torch.cat([zs, z[:, start:end]], dim=1) if zs is not None else z

            # Can't use `z_to_logits` here because it assumes a fully-dimensional z
            z_temp = z if self.resample else zs
            z_dec = self.z_to_z_dec(z_temp[:, 0:self.num_latents_group], 0)

            for group_i in range(1, group + 1):
                start = group_i * self.num_latents_group
                end = (group_i * self.num_latents_group) + self.num_latents_group
                z_dec = z_dec + self.z_to_z_dec(z_temp[:, start:end], group_i)

            logits = self.z_dec_to_logits(z_dec)

            if self.reconstruction:
                # Reshape logits and add to outputs
                logits_reshaped = logits.reshape(images.shape)
                outputs.append(logits_reshaped)

            if group == model_temp:
                # Get the final output
                logits_output = logits.reshape(-1, self.channels * (self.size ** 2))

        # THIS IS WRONG, WON'T WORK AS EXPECTED!
        output = {
            # "x_enc": x_enc,
            "mu": mu,
            "logvar": logvar,
            # "z": z,
            # "z_dec": z_dec,
            "logits": logits_output
        }

        # Calculate loss
        loss, log_prob, KLD = self.ELBO(
            logits_output,
            images.view(-1, self.channels * (self.size ** 2)),
            log_prob_fn=self.log_prob_fn,
            KLD_fn="N",
            mu=mu,
            logvar=logvar,
            std=self.std)

        return output, [-loss.item()], [log_prob.item()], [KLD.item()]

    def save(self, path):
        torch.save(self.canvas.state_dict(), f"{path}_canvas.pth")
        torch.save(self.encoder.state_dict(), f"{path}.pth")
        if self.encoders:
            torch.save(self.encoder_2.state_dict(), f"{path}_enc_2.pth")
        if self.reconstruction and (self.to_latents == "Encoder"):
            torch.save(self.enc_enc_to_enc.state_dict(), f"{path}_enc_enc_to_enc.pth")
        if (self.to_latents == "Encoder") or (self.to_latents == "Latents"):
            if not self.encoder_to_latents:
                torch.save(self.enc_to_lat.state_dict(), f"{path}_enc_to_lat.pth")
            else:
                for i, x in enumerate(self.enc_to_lats):
                    torch.save(x.state_dict(), f"{path}_enc_to_lats_{i}.pth")
        elif self.to_latents == "EncoderLatents":
            for i, x in enumerate(self.enc_lats_to_lats):
                torch.save(x.state_dict(), f"{path}_enc_lats_to_lats_{i}.pth")
        if self.to_latents == "Latents":
            torch.save(self.lats_to_lats.state_dict(), f"{path}_lats_to_lats.pth")
        for i, x in enumerate(self.lats_to_dec):
            torch.save(x.state_dict(), f"{path}_lat_to_dec_{i}.pth")
        torch.save(self.decoder.state_dict(), f"{path}_dec.pth")

    def load(self, path):
        self.canvas.load_state_dict(torch.load(f"{path}_canvas.pth"))
        self.encoder.load_state_dict(torch.load(f"{path}.pth"))
        if self.encoders:
            self.encoder_2.load_state_dict(torch.load(f"{path}_enc_2.pth"))
        if self.reconstruction and (self.to_latents == "Encoder"):
            self.enc_enc_to_enc.load_state_dict(torch.load(f"{path}_enc_enc_to_enc.pth"))
        if (self.to_latents == "Encoder") or (self.to_latents == "Latents"):
            if not self.encoder_to_latents:
                self.enc_to_lat.load_state_dict(torch.load(f"{path}_enc_to_lat.pth"))
            else:
                for i, x in enumerate(self.enc_to_lats):
                    x.load_state_dict(torch.load(f"{path}_enc_to_lats_{i}.pth"))
        elif self.to_latents == "EncoderLatents":
            for i, x in enumerate(self.enc_lats_to_lats):
                x.load_state_dict(torch.load(f"{path}_enc_lats_to_lats_{i}.pth"))
        if self.to_latents == "Latents":
            self.lats_to_lats.load_state_dict(torch.load(f"{path}_lats_to_lats.pth"))
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

    @torch.no_grad()
    def z_decs_to_logits(self, z_decs, groups):
        z_dec = None

        for i in range(len(z_decs)):
            if z_dec == None:
                z_dec = z_decs[i]
            else:
                z_dec = z_dec + z_decs[i]

        return self.z_dec_to_logits(z_dec)
