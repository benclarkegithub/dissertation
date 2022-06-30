from abc import ABC, abstractmethod

import torch


class Method(ABC):
    def __init__(self, num_latents):
        self.num_latents = num_latents

    @abstractmethod
    def train(self, batch_i, data):
        pass

    @abstractmethod
    def test(self, batch_i, data):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def x_to_mu_logvar(self, x):
        pass

    @abstractmethod
    def z_to_logits(self, z):
        pass

    def get_num_latents(self):
        return self.num_latents

    def ELBO(self, logits, x, mu, logvar, *, beta=1):
        CB_log_prob = self.CB_log_prob_fn(logits, x)
        KLD = self.KLD_fn(mu, logvar)

        return (CB_log_prob - (beta * KLD)).mean(), CB_log_prob.mean(), KLD.mean()

    def CB_log_prob_fn(self, logits, x):
        # The continuous Bernoulli: fixing a pervasive error in variational autoencoders, Loaiza-Ganem G and Cunningham
        # JP, NeurIPS 2019. https://arxiv.org/abs/1907.06845.
        CB = torch.distributions.ContinuousBernoulli(logits=logits)
        CB_log_prob = CB.log_prob(x).sum(dim=-1)

        return CB_log_prob

    def KLD_fn(self, mu, logvar):
        return -0.5 * (1 + logvar - (mu ** 2) - logvar.exp()).sum(dim=-1)
