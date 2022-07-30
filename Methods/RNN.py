import torch
import torch.optim as optim
from torchinfo import summary

from Method import Method


"""
encoders
    True:   Use two encoders, one for the images and one for the reconstruction
    False:  Use one encoder, for both the images and the reconstruction
    
encoder_encoder_to_encoder
    True:   Combine the image and reconstruction encodings into one encoding
    False:  Do not combine the encodings, get a separate distribution for the images and reconstruction and combine them
            together with LatentsToLatents
            
encoder_to_latents
    "Many":     Many sets of parameters for encoding to latent variable(s)
    "One":      One set of parameters for encoding to latent variable(s)
    "Latents":  Many sets of parameters and condition Z_n on the combined image and reconstruction encoding and Z_<n
    
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
                 encoders,
                 encoder_encoder_to_encoder,
                 encoder_to_latents,
                 backprop,
                 resample,
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
        # Options
        self.encoders = encoders
        self.encoder_encoder_to_encoder = encoder_encoder_to_encoder
        if (encoder_to_latents == "Latents") and not encoder_encoder_to_encoder:
            raise Exception("The EncoderLatentsToLatents component needs the EncoderEncoderToEncoder component.")
        self.encoder_to_latents = encoder_to_latents
        self.backprop = backprop
        self.resample = resample

        # Canvas
        self.canvas = architecture["Canvas"](size, channels)
        self.optimiser_canvas = optim.Adam(self.canvas.parameters(), lr=1e-3, weight_decay=1e-5)

        # Encoder
        self.encoder = architecture["Encoder"](num_latents, size, channels, out_channels)
        self.optimiser_encoder = optim.Adam(self.encoder.parameters(), lr=1e-3, weight_decay=1e-5)

        # Encoder 2
        if self.encoders:
            self.encoder_2 = architecture["Encoder"](num_latents, size, channels, out_channels)
            self.optimiser_encoder_2 = optim.Adam(self.encoder_2.parameters(), lr=1e-3, weight_decay=1e-5)

        # Encoder Encoder to Encoder
        if self.encoder_encoder_to_encoder:
            self.enc_enc_to_enc = architecture["EncoderEncoderToEncoder"](num_latents)
            self.optimiser_enc_enc_to_enc = optim.Adam(self.enc_enc_to_enc.parameters(), lr=1e-3, weight_decay=1e-5)

        # Encoder to Latents
        if self.encoder_to_latents == "One":
            self.enc_to_lat = architecture["EncoderToLatents"](num_latents, num_latents_group)
            self.optimiser_enc_to_lat = optim.Adam(self.enc_to_lat.parameters(), lr=1e-3, weight_decay=1e-5)
        elif self.encoder_to_latents == "Many":
            self.enc_to_lats = [architecture["EncoderToLatents"](num_latents, num_latents_group)
                                for _ in range(self.num_groups)]
            self.optimiser_enc_to_lats = [optim.Adam(x.parameters(), lr=1e-3, weight_decay=1e-5)
                                          for x in self.enc_to_lats]
        elif self.encoder_to_latents == "Latents":
            self.enc_lats_to_lats = [architecture["EncoderLatentsToLatents"](num_latents, group, num_latents_group)
                                     for group in range(self.num_groups)]
            self.optimiser_enc_lats_to_lats = [optim.Adam(x.parameters(), lr=1e-3, weight_decay=1e-5)
                                               for x in self.enc_lats_to_lats]

        # Latents to Latents
        if not self.encoder_encoder_to_encoder:
            self.lats_to_lats = architecture["LatentsToLatents"](num_latents_group)
            self.optimiser_lats_to_lats = optim.Adam(self.lats_to_lats.parameters(), lr=1e-3, weight_decay=1e-5)

        # Latents to Decoder
        self.lats_to_dec = [architecture["LatentsToDecoder"](num_latents, num_latents_group)
                            for _ in range(self.num_groups)]
        self.optimiser_lats_to_dec = [optim.Adam(x.parameters(), lr=1e-3, weight_decay=1e-5) for x in self.lats_to_dec]

        # Decoder
        self.decoder = architecture["Decoder"](num_latents, size, channels, out_channels)
        self.optimiser_decoder = optim.Adam(self.decoder.parameters(), lr=1e-3, weight_decay=1e-5)


    def train(self, i, data, *, get_grad=False):
        losses = []
        log_probs = []
        KLDs = []
        grads = []
        grads_2 = []

        # Get the input images
        images, _ = data

        # Zero the parameter's gradients
        self.optimiser_canvas.zero_grad()
        self.optimiser_encoder.zero_grad()
        if self.encoders:
            self.optimiser_encoder_2.zero_grad()
        if self.encoder_encoder_to_encoder:
            self.optimiser_enc_enc_to_enc.zero_grad()
        if self.encoder_to_latents == "One":
            self.optimiser_enc_to_lat.zero_grad()
        elif self.encoder_to_latents == "Many":
            for x in self.optimiser_enc_to_lats:
                x.zero_grad()
        elif self.encoder_to_latents == "Latents":
            for x in self.optimiser_enc_lats_to_lats:
                x.zero_grad()
        for x in self.optimiser_lats_to_dec:
            x.zero_grad()
        if not self.encoder_encoder_to_encoder:
            self.optimiser_lats_to_lats.zero_grad()
        self.optimiser_decoder.zero_grad()

        # Forward
        # Get encoder output of images
        x_enc_images = self.encoder(images)

        mu_images = None
        logvar_images = None
        if not self.encoder_encoder_to_encoder:
            # Get the image distribution to combine it later with the LatentsToLatents component
            if self.encoder_to_latents == "One":
                mu_images, logvar_images = self.enc_to_lat(x_enc_images)
            elif self.encoder_to_latents == "Many":
                mu_images, logvar_images = [], []

                for enc_to_lat in self.enc_to_lats:
                    mu_images_1, logvar_images_1 = enc_to_lat(x_enc_images)
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
        # Used when resample == False
        zs = None

        for group in range(self.num_groups):
            # Get encoder output of reconstruction
            output_images = torch.sigmoid(outputs[-1])
            if self.encoders:
                x_enc_rec = self.encoder_2(output_images)
            else:
                x_enc_rec = self.encoder(output_images)

            # Combine the image and reconstruction encodings or use the reconstruction encoding
            x_enc_temp = self.enc_enc_to_enc(x_enc_images, x_enc_rec) if self.encoder_encoder_to_encoder else x_enc_rec
            if self.encoder_to_latents == "One":
                mu_1, logvar_1 = self.enc_to_lat(x_enc_temp)
            elif self.encoder_to_latents == "Many":
                mu_1, logvar_1 = self.enc_to_lats[group](x_enc_temp)
            elif self.encoder_to_latents == "Latents":
                mu_1, logvar_1 = self.enc_lats_to_lats[group](x_enc_temp, mu, logvar)

            if not self.encoder_encoder_to_encoder:
                # Use the LatentsToLatents component to get a combined images and reconstruction distribution
                if self.encoder_to_latents == "One":
                    mu_2, logvar_2 = self.lats_to_lats(mu_images, logvar_images, mu_1, logvar_1)
                elif self.encoder_to_latents == "Many":
                    start = group * self.num_latents_group
                    end = (group * self.num_latents_group) + self.num_latents_group
                    mu_2, logvar_2 = self.lats_to_lats(
                        mu_images[:, start:end], logvar_images[:, start:end], mu_1, logvar_1)
            else:
                mu_2, logvar_2 = mu_1, logvar_1

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
                start = group * self.num_latents_group
                end = (group * self.num_latents_group) + self.num_latents_group
                zs = torch.cat([zs, z[:, start:end]], dim=1) if zs is not None else z

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

            # Get the gradients
            # Not 100% correct when encoder_to_latents == "One"
            if get_grad:
                grad = []
                grad_2 = []

                components = [self.encoder, self.decoder] + ([self.encoder_2] if self.encoders else [])
                for x in components:
                    for name, param in x.named_parameters():
                        grad.append(param.grad.abs().flatten())

                enc_to_lat = None
                if self.encoder_to_latents == "One":
                    enc_to_lat = self.enc_to_lat
                elif self.encoder_to_latents == "Many":
                    enc_to_lat = self.enc_to_lats[group]
                elif self.encoder_to_latents == "Latents":
                    enc_to_lat = self.enc_lats_to_lats[group]

                for x in [enc_to_lat, self.lats_to_dec[group]]:
                    for name, param in x.named_parameters():
                        grad_2.append(param.grad.abs().flatten())

                grads.append(torch.concat(grad).mean().item())
                grads_2.append(torch.concat(grad_2).mean().item())

        # Step
        self.optimiser_canvas.step()
        self.optimiser_encoder.step()
        if self.encoders:
            self.optimiser_encoder_2.step()
        if self.encoder_encoder_to_encoder:
            self.optimiser_enc_enc_to_enc.step()
        if self.encoder_to_latents == "One":
            self.optimiser_enc_to_lat.step()
        elif self.encoder_to_latents == "Many":
            for x in self.optimiser_enc_to_lats:
                x.step()
        elif self.encoder_to_latents == "Latents":
            for x in self.optimiser_enc_lats_to_lats:
                x.step()
        for x in self.optimiser_lats_to_dec:
            x.step()
        if not self.encoder_encoder_to_encoder:
            self.optimiser_lats_to_lats.step()
        self.optimiser_decoder.step()

        # Fix KLDs and gradients
        KLDs_temp = KLDs.copy()
        KLDs_temp.insert(0, 0)
        KLDs = [KLDs[i] - KLDs_temp[i] for i in range(len(KLDs))]
        grads_temp = grads.copy()
        grads_temp.insert(0, 0)
        grads = [grads[i] - grads_temp[i] + grads_2[i] for i in range(len(grads))]

        return losses, log_probs, KLDs, grads

    @torch.no_grad()
    def test(self, i, data):
        # Get the input images
        images, _ = data

        # Forward
        # Get encoder output of images
        x_enc_images = self.encoder(images)

        mu_images = None
        logvar_images = None
        if not self.encoder_encoder_to_encoder:
            # Get the image distribution to combine it later with the LatentsToLatents component
            if self.encoder_to_latents == "One":
                mu_images, logvar_images = self.enc_to_lat(x_enc_images)
            elif self.encoder_to_latents == "Many":
                mu_images, logvar_images = [], []

                for enc_to_lat in self.enc_to_lats:
                    mu_images_1, logvar_images_1 = enc_to_lat(x_enc_images)
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
        # Used when resample == False
        zs = None

        for group in range(self.num_groups):
            # Get encoder output of reconstruction
            output_images = torch.sigmoid(outputs[-1])
            if self.encoders:
                x_enc_rec = self.encoder_2(output_images)
            else:
                x_enc_rec = self.encoder(output_images)

            # Combine the image and reconstruction encodings or use the reconstruction encoding
            x_enc_temp = self.enc_enc_to_enc(x_enc_images, x_enc_rec) if self.encoder_encoder_to_encoder else x_enc_rec
            if self.encoder_to_latents == "One":
                mu_1, logvar_1 = self.enc_to_lat(x_enc_temp)
            elif self.encoder_to_latents == "Many":
                mu_1, logvar_1 = self.enc_to_lats[group](x_enc_temp)
            elif self.encoder_to_latents == "Latents":
                mu_1, logvar_1 = self.enc_lats_to_lats[group](x_enc_temp, mu, logvar)

            if not self.encoder_encoder_to_encoder:
                # Use the LatentsToLatents component to get a combined images and reconstruction distribution
                if self.encoder_to_latents == "One":
                    mu_2, logvar_2 = self.lats_to_lats(mu_images, logvar_images, mu_1, logvar_1)
                elif self.encoder_to_latents == "Many":
                    start = group * self.num_latents_group
                    end = (group * self.num_latents_group) + self.num_latents_group
                    mu_2, logvar_2 = self.lats_to_lats(
                        mu_images[:, start:end], logvar_images[:, start:end], mu_1, logvar_1)
            else:
                mu_2, logvar_2 = mu_1, logvar_1

            mu = torch.cat([mu, mu_2], dim=1) if mu is not None else mu_2
            logvar = torch.cat([logvar, logvar_2], dim=1) if logvar is not None else logvar_2

            # The ELBO might be higher when testing because z is set to its expected value
            z = mu

            if not self.resample:
                start = group * self.num_latents_group
                end = (group * self.num_latents_group) + self.num_latents_group
                zs = torch.cat([zs, z[:, start:end]], dim=1) if zs is not None else z

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
            "x_enc": x_enc_temp,
            "mu": mu,
            "logvar": logvar,
            # "z": z,
            "z_dec": z_dec,
            "logits": logits_reshaped
        }

        # Calculate loss
        loss, log_prob, KLD = self.ELBO(
            logits_reshaped,
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
        if self.encoder_encoder_to_encoder:
            torch.save(self.enc_enc_to_enc.state_dict(), f"{path}_enc_enc_to_enc.pth")
        if self.encoder_to_latents == "One":
            torch.save(self.enc_to_lat.state_dict(), f"{path}_enc_to_lat.pth")
        elif self.encoder_to_latents == "Many":
            for i, x in enumerate(self.enc_to_lats):
                torch.save(x.state_dict(), f"{path}_enc_to_lats_{i}.pth")
        elif self.encoder_to_latents == "Latents":
            for i, x in enumerate(self.enc_lats_to_lats):
                torch.save(x.state_dict(), f"{path}_enc_lats_to_lats_{i}.pth")
        if not self.encoder_encoder_to_encoder:
            torch.save(self.lats_to_lats.state_dict(), f"{path}_lats_to_lats.pth")
        for i, x in enumerate(self.lats_to_dec):
            torch.save(x.state_dict(), f"{path}_lat_to_dec_{i}.pth")
        torch.save(self.decoder.state_dict(), f"{path}_dec.pth")

    def load(self, path):
        self.canvas.load_state_dict(torch.load(f"{path}_canvas.pth"))
        self.encoder.load_state_dict(torch.load(f"{path}.pth"))
        if self.encoders:
            self.encoder_2.load_state_dict(torch.load(f"{path}_enc_2.pth"))
        if self.encoder_encoder_to_encoder:
            self.enc_enc_to_enc.load_state_dict(torch.load(f"{path}_enc_enc_to_enc.pth"))
        if self.encoder_to_latents == "One":
            self.enc_to_lat.load_state_dict(torch.load(f"{path}_enc_to_lat.pth"))
        elif self.encoder_to_latents == "Many":
            for i, x in enumerate(self.enc_to_lats):
                x.load_state_dict(torch.load(f"{path}_enc_to_lats_{i}.pth"))
        elif self.encoder_to_latents == "Latents":
            for i, x in enumerate(self.enc_lats_to_lats):
                x.load_state_dict(torch.load(f"{path}_enc_lats_to_lats_{i}.pth"))
        if not self.encoder_encoder_to_encoder:
            self.lats_to_lats.load_state_dict(torch.load(f"{path}_lats_to_lats.pth"))
        for i, x in enumerate(self.lats_to_dec):
            x.load_state_dict(torch.load(f"{path}_lat_to_dec_{i}.pth"))
        self.decoder.load_state_dict(torch.load(f"{path}_dec.pth"))

    def summary(self):
        summaries = []
        summaries.append("Encoder")
        summaries.append(str(summary(self.encoder)))
        if self.encoders:
            summaries.append("Encoder 2")
            summaries.append(str(summary(self.encoder_2)))
        if self.encoder_encoder_to_encoder:
            summaries.append("Encoder Encoder to Encoder")
            summaries.append(str(summary(self.enc_enc_to_enc)))
        if self.encoder_to_latents == "One":
            summaries.append(f"Encoder to Latent")
            summaries.append(str(summary(self.enc_to_lat)))
        elif self.encoder_to_latents == "Many":
            for i, x in enumerate(self.enc_to_lats):
                summaries.append(f"Encoder to Latents {i}")
                summaries.append(str(summary(x)))
        elif self.encoder_to_latents == "Latents":
            for i, x in enumerate(self.enc_lats_to_lats):
                summaries.append(f"Encoder Latents to Latents {i}")
                summaries.append(str(summary(x)))
        if not self.encoder_encoder_to_encoder:
            summaries.append("Latents to Latents")
            summaries.append(str(summary(self.lats_to_lats)))
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
