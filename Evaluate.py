from os import mkdir
from os.path import exists, isdir
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


class Evaluate:
    def __init__(self, method, experiment):
        self.method = method
        self.experiment = experiment
        self.path = f"{experiment}/{type(method).__name__}_{experiment}"

        # Try and load the model
        self.load(verbose=True)

    def train(self, train_loader, val_loader, max_epochs, max_no_improvement):
        avg_train_loss = []
        avg_train_log_prob = []
        avg_train_KLD = []
        avg_val_loss = []
        avg_val_log_prob = []
        avg_val_KLD = []
        best_avg_val_epoch = 1
        best_avg_val_loss = np.inf

        for epoch in range(1, max_epochs+1):
            # Train the model and get the average training loss
            avg_train_loss_2 = []
            avg_train_log_prob_2 = []
            avg_train_KLD_2 = []

            for i, data in enumerate(train_loader):
                loss, log_prob, KLD = self.method.train(i=i, data=data)
                avg_train_loss_2.append(loss)
                avg_train_log_prob_2.append(log_prob)
                avg_train_KLD_2.append(KLD)

            avg_train_loss.append(np.array(avg_train_loss_2).mean())
            avg_train_log_prob.append(np.array(avg_train_log_prob_2).mean())
            avg_train_KLD.append(np.array(avg_train_KLD_2).mean())

            # Get the trained model's average validation loss
            avg_val_loss_2 = []
            avg_val_log_prob_2 = []
            avg_val_KLD_2 = []

            for i, data in enumerate(val_loader):
                output, loss, log_prob, KLD = self.method.test(i=i, data=data)
                avg_val_loss_2.append(loss)
                avg_val_log_prob_2.append(log_prob)
                avg_val_KLD_2.append(KLD)

            avg_val_loss.append(np.array(avg_val_loss_2).mean())
            avg_val_log_prob.append(np.array(avg_val_log_prob_2).mean())
            avg_val_KLD.append(np.array(avg_val_KLD_2).mean())

            # Print epoch information
            self.print_epoch_info(
                epoch=epoch,
                avg_train_ELBO=-avg_train_loss[-1],
                avg_val_ELBO=-avg_val_loss[-1],
                avg_train_log_prob=avg_train_log_prob[-1],
                avg_val_log_prob=avg_val_log_prob[-1],
                avg_train_KLD=avg_train_KLD[-1],
                avg_val_KLD=avg_val_KLD[-1])

            # Check if the average validation loss is the best
            if avg_val_loss[-1] < best_avg_val_loss:
                best_avg_val_epoch = epoch
                best_avg_val_loss = avg_val_loss[-1]
                # Save the model
                self.save()
            elif (epoch - best_avg_val_epoch + 1) == max_no_improvement:
                print(f"No improvement after {max_no_improvement} epochs...")
                break

        # Plot average training/validation loss over epochs
        avg_train_ELBO = -np.array(avg_train_loss)
        avg_val_ELBO = -np.array(avg_val_loss)

        self.plot_train_val_ELBO(
            avg_train_ELBO=avg_train_ELBO,
            avg_val_ELBO=avg_val_ELBO,
            avg_train_log_prob=avg_train_log_prob,
            avg_val_log_prob=avg_val_log_prob)

        self.plot_train_val_KLD(avg_train_KLD=avg_train_KLD, avg_val_KLD=avg_val_KLD)

    def test(self, test_loader, *, avg_var=True, output_images=True, output_images_opt=None):
        if avg_var:
            # Calculate mu, var, z mean and variance
            mu = None
            logvar = None
            z = None

            for i, data in enumerate(test_loader):
                output, loss, log_prob, KLD = self.method.test(i=i, data=data)

                if (i == 0):
                    mu = output["mu"]
                    logvar = output["logvar"]
                    z = output["z"]
                else:
                    mu = torch.vstack([mu, output["mu"]])
                    logvar = torch.vstack([logvar, output["logvar"]])
                    z = torch.vstack([z, output["z"]])

            var = np.exp(logvar)
            avg_mu = mu.mean(dim=0)
            avg_var = var.mean(dim=0)
            avg_z = z.mean(dim=0)
            var_mu = mu.var(dim=0)
            var_var = var.var(dim=0)
            var_z = z.var(dim=0)

            print(f"Avg. mu: {avg_mu}")
            print(f"Avg. var: {avg_var}")
            print(f"Avg. z: {avg_z}")
            print(f"Var. mu: {var_mu}")
            print(f"Var. var: {var_var}")
            print(f"Var. z: {var_z}")

        if output_images:
            if not output_images_opt:
                # Set output images options if it doesn't exist
                output_images_opt = { "range": 2.5, "number": 11, "size": 28 }

            # Get output images options
            range = output_images_opt["range"]
            number = output_images_opt["number"]
            size = output_images_opt["size"]
            total_number = number ** 2
            total_size = number * size

            # Make a Z tensor in the range, e.g. [-2.5, 2.5], [-2.0, 2.5], ...
            X = np.linspace(start=-range, stop=range, num=number, dtype=np.single)
            Y = np.linspace(start=range, stop=-range, num=number, dtype=np.single)
            Z = [[x, y] for y in Y for x in X]
            Z_tensor = torch.tensor(Z)

            # Add additional latent variables if necessary
            if Z_tensor.shape[1] != self.method.get_num_latents():
                cols_needed = self.method.get_num_latents() - Z_tensor.shape[1]
                cols = torch.zeros(total_number, cols_needed)
                Z_tensor = torch.hstack([Z_tensor, cols])

            # Get output
            z_dec, logits = self.method.z_to_logits(Z_tensor)

            # Make grid of images
            images = torch.sigmoid(logits).unsqueeze(dim=1).reshape(total_number, 1, size, size)
            images_grid = torchvision.utils.make_grid(tensor=images, nrow=number)

            # Create and display the 2D graph
            plt.title(f"{type(self.method).__name__} {self.experiment} output images")
            plt.xlabel("Latent variable 1")
            plt.ylabel("Latent variable 2")
            x_ticks = np.linspace(start=0, stop=total_size, num=number) + (size // 2) - 1
            y_ticks = np.linspace(start=total_size, stop=0, num=number) + (size // 2) - 1
            plt.xticks(ticks=x_ticks, labels=X)
            plt.yticks(ticks=y_ticks, labels=X)
            plt.imshow(X=np.transpose(images_grid.numpy(), (1, 2, 0)))
            plt.show()

    def save(self, verbose=False):
        if verbose:
            print(f"Saving model... {self.path}")
        # Get the directory name and if it doesn't exist create it
        path_split = self.path.split("/")
        if not isdir(path_split[0]):
            mkdir(path_split[0])
        # Save
        self.method.save(self.path)

    def load(self, verbose=False):
        # Check that the path exists and load the model if it does
        if exists(self.path):
            if verbose:
                print(f"Loading model... {self.path}")
            self.method.load(self.path)
        else:
            if verbose:
                print(f"No model exists... {self.path}")

    def print_epoch_info(self,
                         epoch,
                         avg_train_ELBO,
                         avg_val_ELBO,
                         *,
                         avg_train_log_prob,
                         avg_val_log_prob,
                         avg_train_KLD,
                         avg_val_KLD):
        print(f"[Epoch {epoch:3}]\t"
              f"Avg. train ELBO: {avg_train_ELBO:.3f}\t"
              f"Avg. val. ELBO: {avg_val_ELBO:.3f}\t"
              f"Avg. train log prob: {avg_train_log_prob:.3f}\t"
              f"Avg. val. log prob: {avg_val_log_prob:.3f}\t"
              f"Avg. train KLD: {avg_train_KLD:.3f}\t"
              f"Avg. val. KLD: {avg_val_KLD:.3f}\t")

    def plot_train_val_ELBO(self,
            avg_train_ELBO,
            avg_val_ELBO,
            *,
            avg_train_log_prob,
            avg_val_log_prob):
        plt.title("Average training/validation ELBO over epochs")
        epochs = len(avg_train_ELBO)
        X = np.arange(1, epochs+1)
        plt.plot(X, avg_train_ELBO, label="Train ELBO")
        plt.plot(X, avg_val_ELBO, label="Validation ELBO")
        if avg_train_log_prob:
            plt.plot(X, avg_train_log_prob, label="Train log-likelihood")
        if avg_val_log_prob:
            plt.plot(X, avg_val_log_prob, label="Validation log-likelihood")
        plt.legend()
        plt.show()

    def plot_train_val_KLD(self, avg_train_KLD, avg_val_KLD):
        plt.title("Average training/validation KL divergence over epochs")
        epochs = len(avg_train_KLD)
        X = np.arange(1, epochs + 1)
        plt.plot(X, avg_train_KLD, label="Train KL divergence")
        plt.plot(X, avg_val_KLD, label="Validation KL divergence")
        plt.legend()
        plt.show()
