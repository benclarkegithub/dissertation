import torch
import torch.optim as optim
from torchinfo import summary

from Method import Method

"""
Variants:
    Combined:
        x -> x̂_1
        x - x̂_1 -> x̂_2
        x - x̂_1 - x̂_2 -> x̂_3
        
    Original:
        x -> x̂_1
        x -> x̂_2
        x -> x̂_3
        
    Reconstruction:
        x -> x̂_1
        x̂_1 -> x̂_2
        x̂_2 -> x̂_3
"""
class Multiple(Method):
    def __init__(self,
                 VAE,
                 num_latents,
                 num_latents_group,
                 variant,
                 *,
                 learning_rate=1e-3,
                 size=28,
                 channels=1,
                 out_channels=None,
                 log_prob_fn="CB",
                 std=0.05,
                 hidden_size=None):
        super().__init__(num_latents=num_latents, type="Multiple")

        self.num_latents_group = num_latents_group
        self.num_groups = num_latents // num_latents_group
        self.variant = variant
        self.size = size
        self.channels = channels
        self.log_prob_fn = log_prob_fn
        self.std = std
        self.hidden_size = hidden_size if hidden_size is not None else channels * (size ** 2) // 8

        self.VAEs = [VAE(
            num_latents,
            self.num_latents_group,
            size=size,
            channels=channels,
            out_channels=out_channels,
            hidden_size=self.hidden_size)
                     for _ in range(self.num_groups)]
        self.optimisers = [optim.Adam(x.parameters(), lr=learning_rate) for x in self.VAEs]

    def train(self, i, data, *, get_grad=None, model=None):
        # Get the input images
        images, _ = data
        images_logits = torch.logit(images)
        images_logits_clipped = self.clip_logits(images_logits)

        # For each of the variants, the loss is as follows:
        # x
        # x - x̂_1
        # x - x̂_1 - x̂_2
        # etc...
        # First get the target output, as the processing done here can be reused for the combined method
        inputs = [images_logits_clipped]
        outputs = []

        for VAE_i in range(model):
            with torch.no_grad():
                output = self.VAEs[VAE_i](inputs[-1], reparameterise=False)["logits"]
                output_reshaped = output.reshape(images.shape)
                output_reshaped_clipped = self.clip_logits(output_reshaped)
                outputs.append(output_reshaped_clipped)

            # If self.variant is Original, nothing necessary
            if self.variant == "Reconstruction":
                inputs.append(output_reshaped_clipped)
            elif self.variant == "Combined":
                with torch.no_grad():
                    next_input = inputs[-1] - output_reshaped_clipped
                    next_input_clipped = self.clip_logits(next_input)
                    inputs.append(next_input_clipped)

        # Get the target output
        target_output = images_logits_clipped.detach().clone()
        for output in outputs:
            target_output = target_output - output
        # target_output needs to be transformed to the range [0, 1] to be used in the ELBO equation
        target_output_images = torch.sigmoid(target_output)

        # Zero the parameter's gradients
        self.optimisers[model].zero_grad()

        # Forward
        if self.variant == "Original":
            output = self.VAEs[model](images_logits_clipped)
        elif self.variant == "Reconstruction":
            input = outputs[-1] if len(outputs) > 0 else images_logits_clipped
            output = self.VAEs[model](input)
        else:
            output = self.VAEs[model](target_output)

        # Backward
        loss, log_prob, KLD = self.ELBO(
            output["logits"],
            target_output_images.view(-1, self.channels * (self.size ** 2)),
            log_prob_fn=self.log_prob_fn,
            KLD_fn="N",
            mu=output["mu"],
            logvar=output["logvar"],
            std=self.std)
        # Because optimisers minimise, and we want to maximise the ELBO, we multiply it by -1
        loss = -loss
        loss.backward()

        # Get the gradients
        grad = None
        if get_grad:
            temp_grad = []

            for name, param in self.VAEs[model].named_parameters():
                temp_grad.append(param.grad.abs().flatten())

            grad = torch.concat(temp_grad).mean().item()

        # Step
        self.optimisers[model].step()

        return [loss.item()], [log_prob.item()], [KLD.item()], [grad]

    @torch.no_grad()
    def test(self, i, data, *, model=None):
        # Get the input images
        images, _ = data
        images_logits = torch.logit(images)
        images_logits_clipped = self.clip_logits(images_logits)
        # images_logits_clipped needs to be transformed to the range [0, 1] to be used in the ELBO equation
        images_clipped = torch.sigmoid(images_logits_clipped)

        mu = []
        logvar = []
        inputs = [images_logits_clipped]
        outputs = []

        # The current model should be included in the test
        models = model + 1 if model is not None else len(self.VAEs)

        for VAE_i in range(models):
            output = self.VAEs[VAE_i](inputs[-1], reparameterise=False)
            mu.append(output["mu"])
            logvar.append(output["logvar"])
            output_reshaped = output["logits"].reshape(images.shape)
            # In training the last VAE's output is not clipped
            output_reshaped_clipped = self.clip_logits(output_reshaped) if VAE_i != (models - 1) else output_reshaped
            outputs.append(output_reshaped_clipped)

            # If self.variant is Original, nothing necessary
            if self.variant == "Reconstruction":
                inputs.append(output_reshaped_clipped)
            elif self.variant == "Combined":
                next_input = inputs[-1] - output_reshaped_clipped
                next_input_clipped = self.clip_logits(next_input)
                inputs.append(next_input_clipped)

        mu = torch.cat(mu, dim=1)
        logvar = torch.cat(logvar, dim=1)

        # Get the final output
        # In training the target output is not clipped
        final_output = torch.zeros_like(images.view(-1, self.channels * (self.size ** 2)))
        for output in outputs:
            final_output = final_output + output.view(-1, self.channels * (self.size ** 2))

        # Calculate loss
        loss, log_prob, KLD = self.ELBO(
            final_output,
            images_clipped.view(-1, self.channels * (self.size ** 2)),
            log_prob_fn=self.log_prob_fn,
            KLD_fn="N",
            mu=mu,
            logvar=logvar,
            std=self.std)

        # Commented out for now
        output = {
            # "x_enc": x_enc,
            "mu": mu,
            "logvar": logvar,
            # "z": z,
            # "z_dec": z_dec,
            "logits": final_output
        }

        return output, [-loss.item()], [log_prob.item()], [KLD.item()]

    def save(self, path):
        for i, x in enumerate(self.VAEs):
            if i == 0:
                torch.save(x.state_dict(), f"{path}.pth")
            else:
                torch.save(x.state_dict(), f"{path}_{i}.pth")

    def load(self, path):
        for i, x in enumerate(self.VAEs):
            if i == 0:
                x.load_state_dict(torch.load(f"{path}.pth"))
            else:
                x.load_state_dict(torch.load(f"{path}_{i}.pth"))

    def summary(self):
        return [str(summary(x)) for x in self.VAEs]

    @torch.no_grad()
    def x_to_mu_logvar(self, x):
        pass

    @torch.no_grad()
    def z_to_logits(self, z):
        z_decs = []
        groups = range(len(self.VAEs))

        for VAE_i in groups:
            start = VAE_i * self.num_latents_group
            end = (VAE_i * self.num_latents_group) + self.num_latents_group
            z_dec = self.z_to_z_dec(z[:, start:end], VAE_i)
            z_decs.append(z_dec)

        final_output = self.z_decs_to_logits(z_decs, groups)

        return z_decs, final_output

    @torch.no_grad()
    def z_to_z_dec(self, z, group):
        return self.VAEs[group].z_to_z_dec(z, 0)

    # Because this method uses multiple encoders/decoders, the
    # groups the `z_decs` vectors came from needs to be specified.
    @torch.no_grad()
    def z_decs_to_logits(self, z_decs, groups):
        outputs = []

        for i, VAE_i in enumerate(groups):
            logits = self.VAEs[VAE_i].z_dec_to_logits(z_decs[i])
            # In training the last VAE's output is not clipped
            logits_clipped = self.clip_logits(logits) if VAE_i != (len(self.VAEs) - 1) else logits
            outputs.append(logits_clipped)

        # Get the final output
        # In training the target output is not clipped
        final_output = torch.zeros_like(outputs[0])
        for output in outputs:
            final_output = final_output + output

        return final_output

    def clip_logits(self, logits, *, range=6):
        # Get the upper and lower bounds
        upper = torch.tensor(range)
        lower = torch.tensor(-range)
        # Clip
        logits = torch.where(logits < lower, lower, logits)
        logits = torch.where(logits > upper, upper, logits)

        return logits
