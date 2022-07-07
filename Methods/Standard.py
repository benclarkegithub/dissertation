import torch
import torch.optim as optim
from torchinfo import summary

from Method import Method


class Standard(Method):
    def __init__(self, VAE, num_latents):
        super().__init__(num_latents=num_latents)

        self.VAE = VAE(num_latents)
        self.optimiser = optim.Adam(self.VAE.parameters(), lr=1e-3) # 0.001

    def train(self, i, data, *, get_grad=False):
        # Get the input images
        images, _ = data

        # Zero the parameter's gradients
        self.optimiser.zero_grad()

        # Forward, backward, loss
        output = self.VAE(images)
        loss, log_prob, KLD = self.ELBO(output["logits"], images.view(-1, 28 * 28), output["mu"], output["logvar"])
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

        return [loss.item()], [log_prob.item()], [KLD.item()], [grad]

    @torch.no_grad()
    def test(self, i, data):
        # Get the input images
        images, _ = data
        # Get the output
        output = self.VAE(images)
        # Calculate loss
        loss, log_prob, KLD = self.ELBO(output["logits"], images.view(-1, 28 * 28), output["mu"], output["logvar"])

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
