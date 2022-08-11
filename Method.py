from abc import ABC, abstractmethod
import torch

class Method(ABC):
    def __init__(self,
                 num_latents,
                 type,
                 learning_rate=1e-3,
                 size=28,
                 channels=1,
                 out_channels=None,
                 log_prob_fn="CB",
                 std=0.05,
                 hidden_size=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_latents = num_latents
        self.num_latents_group = 1
        self.num_groups = num_latents // self.num_latents_group
        self.type = type
        self.learning_rate = learning_rate
        self.size = size
        self.channels = channels
        self.out_channels = out_channels
        self.log_prob_fn = log_prob_fn
        self.std = std
        self.hidden_size = hidden_size if hidden_size is not None else channels * (size ** 2) // 8

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

    def get_learning_rate(self):
        return self.learning_rate

    def get_size(self):
        return self.size

    def get_channels(self):
        return self.channels

    def get_out_channels(self):
        return self.out_channels

    def get_log_prob_fn(self):
        return self.log_prob_fn

    def get_std(self):
        return self.std

    def get_hidden_size(self):
        return self.hidden_size

    def ELBO(self, logits, x, *, log_prob_fn="CB", KLD_fn="N", mu=None, logvar=None, log_p=None, log_q=None, beta=1, std=0.05):
        if log_prob_fn == "CB":
            log_prob = self.CB_log_prob_fn(logits, x)
        elif log_prob_fn == "N":
            log_prob = self.N_log_prob_fn(logits, x, std=std)
        else: # log_prob_fn == "MSE"
            # MSE needs to be multipled by -1 because the methods use gradient ascent
            log_prob = -self.MSE_fn(logits, x)

        if KLD_fn == "N":
            KLD = self.KLD_N_fn(mu, logvar)
        else: # KLD_fn == "Custom"
            KLD = self.KLD_Custom_fn(log_p, log_q)

        return (log_prob - (beta * KLD)).mean(), log_prob.mean(), KLD.mean()

    def CB_log_prob_fn(self, logits, x):
        # The continuous Bernoulli: fixing a pervasive error in variational autoencoders, Loaiza-Ganem G and Cunningham
        # JP, NeurIPS 2019. https://arxiv.org/abs/1907.06845.
        CB = torch.distributions.ContinuousBernoulli(logits=logits)
        CB_log_prob = CB.log_prob(x).sum(dim=-1)

        return CB_log_prob

    def N_log_prob_fn(self, logits, x, *, std=0.05):
        images = torch.sigmoid(logits)
        N = torch.distributions.Normal(images, std)
        N_log_prob = N.log_prob(x).sum(dim=-1)

        return N_log_prob

    def MSE_fn(self, logits, x):
        images = torch.sigmoid(logits)
        MSE = (images - x) ** 2
        MSE_sum = MSE.sum(dim=-1)

        return MSE_sum

    def KLD_N_fn(self, mu, logvar):
        return -0.5 * (1 + logvar - (mu ** 2) - logvar.exp()).sum(dim=-1)

    def KLD_Custom_fn(self, log_p, log_q):
        return (log_q - log_p).sum(dim=-1)
