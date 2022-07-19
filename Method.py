from abc import ABC, abstractmethod

import torch


class Method(ABC):
    def __init__(self, num_latents, type):
        self.num_latents = num_latents
        self.num_latents_group = 1
        self.num_groups = num_latents // self.num_latents_group
        self.type = type

    @abstractmethod
    def train(self, i, data, *, get_grad=False):
        pass

    @abstractmethod
    def test(self, i, data):
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

    @abstractmethod
    def z_to_z_dec(self, z, group):
        pass

    # Type is "Single"
    # @abstractmethod
    # def z_dec_to_logits(self, z_dec):
    #     pass

    # Type is "Multiple"
    # @abstractmethod
    # def z_decs_to_logits(self, z_dec):
    #     pass

    def get_num_latents(self):
        return self.num_latents

    def get_num_latents_group(self):
        return self.num_latents_group

    def get_num_groups(self):
        return self.num_groups

    def get_type(self):
        return self.type

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
