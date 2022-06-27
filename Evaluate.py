from os.path import exists
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from Utils import make_dir

class Evaluate:
    def __init__(self, method, experiment, *, log=True):
        self.method = method
        self.experiment = experiment
        self.path = f"{experiment}/{type(method).__name__}_{experiment}"
        self.log = log

    def train(self, train_loader, val_loader, max_epochs, max_no_improvement):
        # Try and load the model
        self.load(best=False, verbose=True)
        self.log_summary()

        # Load training progress
        training_progress = self.load_training()

        if training_progress is None:
            avg_train_loss = []
            avg_train_log_prob = []
            avg_train_KLD = []
            avg_val_loss = []
            avg_val_log_prob = []
            avg_val_KLD = []
            best_avg_val_epoch = 1
            best_avg_val_loss = np.inf
            current_epoch =  0
            epoch_time = []
        else:
            avg_train_loss = training_progress["avg_train_loss"]
            avg_train_log_prob = training_progress["avg_train_log_prob"]
            avg_train_KLD = training_progress["avg_train_KLD"]
            avg_val_loss = training_progress["avg_val_loss"]
            avg_val_log_prob = training_progress["avg_val_log_prob"]
            avg_val_KLD = training_progress["avg_val_KLD"]
            best_avg_val_epoch = training_progress["best_avg_val_epoch"]
            best_avg_val_loss = training_progress["best_avg_val_loss"]
            current_epoch = training_progress["current_epoch"]
            epoch_time = training_progress["epoch_time"]

        for epoch in range(current_epoch+1, max_epochs+1):
            # Train the model and get the average training loss
            avg_train_loss_2 = []
            avg_train_log_prob_2 = []
            avg_train_KLD_2 = []

            # Keep track of the training time
            start = time.time()

            for i, data in enumerate(train_loader):
                loss, log_prob, KLD = self.method.train(i=i, data=data)
                avg_train_loss_2.append(loss)
                avg_train_log_prob_2.append(log_prob)
                avg_train_KLD_2.append(KLD)

            end = time.time()
            epoch_time.append(end - start)

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

            # Log epoch information
            self.log_epoch(
                epoch=epoch,
                epoch_time=epoch_time[-1],
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
                self.save(best=True)
            elif (epoch - best_avg_val_epoch + 1) == max_no_improvement:
                print(f"No improvement after {max_no_improvement} epochs...")
                break

            # Save training progress
            self.save(best=False)
            self.save_training(avg_train_loss=avg_train_loss,
                               avg_train_log_prob=avg_train_log_prob,
                               avg_train_KLD=avg_train_KLD,
                               avg_val_loss=avg_val_loss,
                               avg_val_log_prob=avg_val_log_prob,
                               avg_val_KLD=avg_val_KLD,
                               best_avg_val_epoch=best_avg_val_epoch,
                               best_avg_val_loss=best_avg_val_loss,
                               current_epoch=epoch,
                               epoch_time=epoch_time)

        # Log training stats
        self.log_training(
            best_epoch=best_avg_val_epoch,
            training_time=sum(epoch_time),
            best_train_ELBO=-min(avg_train_loss),
            best_train_log_prob=max(avg_train_log_prob),
            best_val_ELBO=-min(avg_val_loss),
            best_val_log_prob=max(avg_val_log_prob)
        )

        # Plot average training/validation loss over epochs
        self.plot_train_val_ELBO(
            avg_train_ELBO=-np.array(avg_train_loss),
            avg_val_ELBO=-np.array(avg_val_loss),
            avg_train_log_prob=avg_train_log_prob,
            avg_val_log_prob=avg_val_log_prob)

        self.plot_train_val_KLD(avg_train_KLD=avg_train_KLD, avg_val_KLD=avg_val_KLD)

    def test(self, test_loader, *, avg_var=True, output_images=True, output_images_opt=None):
        # Try and load the best model
        self.load(best=True, verbose=True)

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
            max_mu = mu.max(dim=0)
            max_var = var.max(dim=0)
            max_z = z.max(dim=0)
            min_mu = mu.min(dim=0)
            min_var = var.min(dim=0)
            min_z = z.min(dim=0)
            cov_mu = np.cov(mu.T)
            cov_var = np.cov(var.T)
            cov_z = np.cov(z.T)

            message = f"Avg. mu: {avg_mu}\n"\
                      f"Avg. var: {avg_var}\n"\
                      f"Avg. z: {avg_z}\n"\
                      f"Max. mu: {max_mu.values}\n"\
                      f"Max. var: {max_var.values}\n"\
                      f"Max. z: {max_z.values}\n"\
                      f"Min. mu: {min_mu.values}\n"\
                      f"Min. var: {min_var.values}\n"\
                      f"Min. z: {min_z.values}\n"\
                      f"Cov. mu: {cov_mu}\n" \
                      f"Cov. var: {cov_var}\n" \
                      f"Cov. z: {cov_z}\n"

            print(message)

            if self.log:
                self.write_log(message)

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

    def save(self, best, verbose=False):
        path = self.path + "_Best" if best else self.path
        if verbose:
            print(f"Saving model... {path}")
        # Get the directory name and if it doesn't exist create it
        make_dir(path)
        # Save
        self.method.save(path)

    def load(self, best, verbose=False):
        path = self.path + "_Best" if best else self.path
        # Check that the path exists and load the model if it does
        if exists(f"{path}.pth"):
            if verbose:
                print(f"Loading model... {path}")
            self.method.load(path)
        else:
            if verbose:
                print(f"No model exists... {path}")

    def save_training(self,
                      avg_train_loss,
                      avg_train_log_prob,
                      avg_train_KLD,
                      avg_val_loss,
                      avg_val_log_prob,
                      avg_val_KLD,
                      best_avg_val_epoch,
                      best_avg_val_loss,
                      current_epoch,
                      epoch_time):
        # Get the directory name and if it doesn't exist create it
        make_dir(self.path)

        training_progress = {
            "avg_train_loss": avg_train_loss,
            "avg_train_log_prob": avg_train_log_prob,
            "avg_train_KLD": avg_train_KLD,
            "avg_val_loss": avg_val_loss,
            "avg_val_log_prob": avg_val_log_prob,
            "avg_val_KLD": avg_val_KLD,
            "best_avg_val_epoch": best_avg_val_epoch,
            "best_avg_val_loss": best_avg_val_loss,
            "current_epoch": current_epoch,
            "epoch_time": epoch_time,
        }

        with open(f"{self.path}.pkl", "wb") as file:
            pickle.dump(training_progress, file)

    def load_training(self):
        training_progress = None

        if exists(f"{self.path}.pkl"):
            with open(f"{self.path}.pkl", "rb") as file:
                training_progress = pickle.load(file)

        return training_progress

    def log_summary(self):
        summary = self.method.summary()

        if self.log:
            # Check if the summary is a string or a list of strings and log accordingly
            if type(summary) is str:
                self.write_log(summary)
            else:
                for s in summary:
                    self.write_log(s)

    def log_epoch(self,
                  epoch,
                  epoch_time,
                  avg_train_ELBO,
                  avg_val_ELBO,
                  *,
                  avg_train_log_prob,
                  avg_val_log_prob,
                  avg_train_KLD,
                  avg_val_KLD):
        message = f"[Epoch {epoch:3}] ({epoch_time:.2f}s)\t"\
                  f"ELBO: {avg_train_ELBO:.3f} ({avg_val_ELBO:.3f})\t"\
                  f"Log prob: {avg_train_log_prob:.3f} ({avg_val_log_prob:.3f})\t"\
                  f"KLD: {avg_train_KLD:.3f} ({avg_val_KLD:.3f})\t"

        # Print to console and log to file
        print(message)

        if self.log:
            self.write_log(message)

    def log_training(self,
                     best_epoch,
                     training_time,
                     best_train_ELBO,
                     best_train_log_prob,
                     best_val_ELBO,
                     best_val_log_prob):
        message = f"Best epoch: {best_epoch}\t"\
                  f"Training time: {training_time:.2f}s\t" \
                  f"Best ELBO: {best_train_ELBO:.3f} ({best_val_ELBO:.3f})\t" \
                  f"Best log prob: {best_train_log_prob:.3f} ({best_val_log_prob:.3f})"

        # Print to console and log to file
        print(message)

        if self.log:
            self.write_log(message)

    def write_log(self, message):
        # Get the directory name and if it doesn't exist create it
        make_dir(self.path)

        # Check to see if the log file exists
        if exists(f"{self.path}.log"):
            with open(f"{self.path}.log", "a") as file:
                file.write(f"{message}\n")
        else:
            with open(f"{self.path}.log", "w") as file:
                file.write(f"{message}\n")

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
