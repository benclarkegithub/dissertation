import torch
import torch.optim as optim
from torchinfo import summary

from Method import Method


class Standard(Method):
    def __init__(self,
                 VAE,
                 num_latents,
                 *,
                 learning_rate=1e-3,
                 size=28,
                 channels=1,
                 out_channels=None,
                 log_prob_fn="CB",
                 beta=1,
                 std=0.05,
                 hidden_size=None):
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

        self.VAE = VAE(
            self.num_latents,
            self.num_latents_group,
            size=self.size,
            channels=self.channels,
            out_channels=self.out_channels,
            hidden_size=self.hidden_size)
        self.optimiser = optim.Adam(self.VAE.parameters(), lr=self.learning_rate)

    def train(self, i, data, *, get_grad=False):
        # Get the input images
        images, _ = data

        # Zero the parameter's gradients
        self.optimiser.zero_grad()

        # Forward, backward, loss
        output = self.VAE(images)
        loss, log_prob, KLDs = self.ELBO(
            output["logits"],
            images.view(-1, self.channels * (self.size ** 2)),
            log_prob_fn=self.log_prob_fn,
            KLD_fn="N",
            KLD_multiple=True,
            mu=output["mu"],
            logvar=output["logvar"],
            beta=self.beta,
            std=self.std)
        # Because optimisers minimise, and we want to maximise the ELBO, we multiply it by -1
        loss = -loss
        loss.backward()

        # Get the gradients
        grad = None
        if get_grad:
            temp_grad = []

            for name, param in self.VAE.named_parameters():
                temp_grad.append(param.grad.abs().flatten())

            grad = torch.concat(temp_grad).mean().item()

        # Step
        self.optimiser.step()

        return [loss.item()], [log_prob.item()], KLDs.tolist(), [grad]

    @torch.no_grad()
    def test(self, i, data):
        # Get the input images
        images, _ = data

        # Get the output
        output = self.VAE(images, reparameterise=False)

        # Calculate loss
        loss, log_prob, KLD = self.ELBO(
            output["logits"],
            images.view(-1, self.channels * (self.size ** 2)),
            log_prob_fn=self.log_prob_fn,
            KLD_fn="N",
            mu=output["mu"],
            logvar=output["logvar"],
            beta=self.beta,
            std=self.std)

        return output, [-loss.item()], [log_prob.item()], [KLD.item()]

    def save(self, path):
        torch.save(self.VAE.state_dict(), f"{path}.pth")

    def load(self, path):
        self.VAE.load_state_dict(torch.load(f"{path}.pth"))

    def summary(self):
        return str(summary(self.VAE))

    @torch.no_grad()
    def x_to_mu_logvar(self, x):
        return self.VAE.x_to_mu_logvar(x)

    @torch.no_grad()
    def z_to_logits(self, z):
        return self.VAE.z_to_logits(z)

    @torch.no_grad()
    def z_to_z_dec(self, z, group):
        return self.VAE.z_to_z_dec(z, group)

    @torch.no_grad()
    def z_dec_to_logits(self, z_dec):
        return self.VAE.z_dec_to_logits(z_dec)
