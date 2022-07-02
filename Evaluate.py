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
                losses, log_probs, KLDs = self.method.train(i=i, data=data)
                avg_train_loss_2.append(losses)
                avg_train_log_prob_2.append(log_probs)
                avg_train_KLD_2.append(KLDs)

            # Keep track of the training time
            end = time.time()
            epoch_time.append(end - start)

            avg_train_loss.append(np.array(avg_train_loss_2).mean(axis=0))
            avg_train_log_prob.append(np.array(avg_train_log_prob_2).mean(axis=0))
            avg_train_KLD.append(np.array(avg_train_KLD_2).mean(axis=0))

            # Get the trained model's average validation loss
            avg_val_loss_2 = []
            avg_val_log_prob_2 = []
            avg_val_KLD_2 = []

            for i, data in enumerate(val_loader):
                output, losses, log_probs, KLDs = self.method.test(i=i, data=data)
                avg_val_loss_2.append(losses)
                avg_val_log_prob_2.append(log_probs)
                avg_val_KLD_2.append(KLDs)

            avg_val_loss.append(np.array(avg_val_loss_2).mean(axis=0))
            avg_val_log_prob.append(np.array(avg_val_log_prob_2).mean(axis=0))
            avg_val_KLD.append(np.array(avg_val_KLD_2).mean(axis=0))

            # Log epoch information
            self.log_epoch(
                epoch=epoch,
                epoch_time=epoch_time[-1],
                avg_train_losses=avg_train_loss[-1],
                avg_val_losses=avg_val_loss[-1],
                avg_train_log_probs=avg_train_log_prob[-1],
                avg_val_log_probs=avg_val_log_prob[-1],
                avg_train_KLDs=avg_train_KLD[-1],
                avg_val_KLDs=avg_val_KLD[-1])

            # Check if the average validation loss is the best
            if avg_val_loss[-1][-1] < best_avg_val_loss:
                best_avg_val_epoch = epoch
                best_avg_val_loss = avg_val_loss[-1][-1]
                # Save the model
                self.save(best=True)
            elif (epoch - best_avg_val_epoch + 1) == max_no_improvement:
                self.write_log(f"No improvement after {max_no_improvement} epochs...")
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
            avg_train_loss=avg_train_loss,
            avg_val_loss=avg_val_loss,
            avg_train_log_prob=avg_train_log_prob,
            avg_val_log_prob=avg_val_log_prob
        )

        # Plot training/validation ELBO and KL divergence
        self.plot_training_ELBO(avg_train_loss=avg_train_loss, avg_val_loss=avg_val_loss)
        self.plot_training_KLD(avg_train_KLD=avg_train_KLD, avg_val_KLD=avg_val_KLD)

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

            var = torch.exp(logvar)
            avg_mu = mu.mean(dim=0)
            avg_var = var.mean(dim=0)
            max_mu = mu.max(dim=0)
            max_var = var.max(dim=0)
            min_mu = mu.min(dim=0)
            min_var = var.min(dim=0)
            cov_mu = np.cov(mu.T)

            torch.set_printoptions(precision=3, profile="short", sci_mode=False)
            np.set_printoptions(precision=3, formatter={"all": lambda x: f"{x:.3f}"})

            message = f"Avg. mu: {', '.join([f'{x:.3f}' for x in avg_mu.tolist()])}\n" \
                      f"Avg. var: {', '.join([f'{x:.3f}' for x in avg_var.tolist()])}\n" \
                      f"Max. mu: {', '.join([f'{x:.3f}' for x in max_mu.values.tolist()])}\n" \
                      f"Max. var: {', '.join([f'{x:.3f}' for x in max_var.values.tolist()])}\n" \
                      f"Min. mu: {', '.join([f'{x:.3f}' for x in min_mu.values.tolist()])}\n" \
                      f"Min. var: {', '.join([f'{x:.3f}' for x in min_var.values.tolist()])}\n" \
                      f"Cov. mu:\n{cov_mu}"

            torch.set_printoptions()
            np.set_printoptions()

            # Log message
            self.write_log(message)

        if output_images:
            if not output_images_opt:
                # Set output images options if it doesn't exist
                output_images_opt = { "range": 2.5, "number": 11, "size": 28 }

            # Get output images options
            range_opt = output_images_opt["range"]
            number = output_images_opt["number"]
            size = output_images_opt["size"]
            total_number = number ** 2
            total_size = number * size

            # Make a Z tensor in the range, e.g. [-2.5, 2.5], [-2.0, 2.5], ...
            X = np.linspace(start=-range_opt, stop=range_opt, num=number, dtype=np.single)
            Y = np.linspace(start=range_opt, stop=-range_opt, num=number, dtype=np.single)
            Z = torch.tensor([[x, y] for y in Y for x in X])

            # Output images for every pair of latent variables, i.e. (z1, z2), (z3, z4), ...
            for z_i in np.arange(0, self.method.get_num_latents()-1, 2):
                Z_input = torch.zeros(total_number, self.method.get_num_latents())
                Z_input[:, z_i:z_i+2] = Z

                # Get output
                z_dec, logits = self.method.z_to_logits(Z_input)

                # Make grid of images
                images = torch.sigmoid(logits).unsqueeze(dim=1).reshape(total_number, 1, size, size)
                images_grid = torchvision.utils.make_grid(tensor=images, nrow=number)

                # Create and display the 2D graph
                plt.title(f"{type(self.method).__name__} {self.experiment} output images (z{z_i+1} & z{z_i+2})")
                plt.xlabel(f"z{z_i+1}")
                plt.ylabel(f"z{z_i+2}")
                x_ticks = np.linspace(start=0, stop=total_size, num=number) + (size // 2) - 1
                y_ticks = np.linspace(start=total_size, stop=0, num=number) + (size // 2) - 1
                plt.xticks(ticks=x_ticks, labels=X)
                plt.yticks(ticks=y_ticks, labels=X)
                plt.imshow(X=np.transpose(images_grid.numpy(), (1, 2, 0)))
                plt.savefig(f"{self.path}_Output_Images_z{z_i+1}_z{z_i+2}.png")
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
                  avg_train_losses,
                  avg_val_losses,
                  *,
                  avg_train_log_probs,
                  avg_val_log_probs,
                  avg_train_KLDs,
                  avg_val_KLDs):
        message = f"[Epoch {epoch:3} ({epoch_time:.2f}s)]\t" \
                  f"ELBO: {', '.join([f'{-x:.3f}' for x in avg_train_losses])}" \
                  f" ({', '.join([f'{-x:.3f}' for x in avg_val_losses])})\t" \
                  f"Log prob: {', '.join([f'{x:.3f}' for x in avg_train_log_probs])}" \
                  f" ({', '.join([f'{x:.3f}' for x in avg_val_log_probs])})\t" \
                  f"KLD: {', '.join([f'{x:.3f}' for x in avg_train_KLDs])}" \
                  f" ({', '.join([f'{x:.3f}' for x in avg_val_KLDs])})"

        # Log message
        self.write_log(message)

    def log_training(self,
                     best_epoch,
                     training_time,
                     avg_train_loss,
                     avg_val_loss,
                     avg_train_log_prob,
                     avg_val_log_prob):
        avg_train_loss = np.stack(avg_train_loss)
        avg_val_loss = np.stack(avg_val_loss)
        avg_train_log_prob = np.stack(avg_train_log_prob)
        avg_val_log_prob = np.stack(avg_val_log_prob)

        best_train_ELBO = -avg_train_loss[:, -1].min()
        best_val_ELBO = -avg_val_loss[:, -1].min()
        best_train_log_prob = avg_train_log_prob[:, -1].max()
        best_val_log_prob = avg_val_log_prob[:, -1].max()

        message = f"Best epoch: {best_epoch}\t"\
                  f"Training time: {training_time:.2f}s\t" \
                  f"Best ELBO: {best_train_ELBO:.3f} ({best_val_ELBO:.3f})\t" \
                  f"Best log prob: {best_train_log_prob:.3f} ({best_val_log_prob:.3f})"

        # Log message
        self.write_log(message)

    def write_log(self, message):
        # Print the message to console
        print(message)

        if self.log:
            # Get the directory name and if it doesn't exist create it
            make_dir(self.path)

            # Check to see if the log file exists
            if exists(f"{self.path}.log"):
                with open(f"{self.path}.log", "a") as file:
                    file.write(f"{message}\n")
            else:
                with open(f"{self.path}.log", "w") as file:
                    file.write(f"{message}\n")

    def plot_training_ELBO(self,
            avg_train_loss,
            avg_val_loss,
            *,
            avg_train_log_prob=None,
            avg_val_log_prob=None):
        avg_train_loss = np.stack(avg_train_loss)
        avg_val_loss = np.stack(avg_val_loss)
        if avg_train_log_prob:
            avg_train_log_prob = np.stack(avg_train_log_prob)
        if avg_val_log_prob:
            avg_val_log_prob = np.stack(avg_val_log_prob)

        plt.title(f"{type(self.method).__name__} {self.experiment} avg. training/validation ELBO")
        epochs, train_groups = avg_train_loss.shape
        X = np.arange(1, epochs+1)

        for g_i in range(train_groups):
            plt.plot(X, -avg_train_loss[:, g_i], label=f"Train ELBO (Z<={g_i+1})")
            if avg_train_log_prob:
                plt.plot(X, avg_train_log_prob[:, g_i], label=f"Train log-likelihood (Z<={g_i+1})")

        plt.plot(X, -avg_val_loss[:, 0], label="Validation ELBO")
        if avg_val_log_prob:
            plt.plot(X, avg_val_log_prob[:, 0], label="Validation log-likelihood")

        plt.legend()
        plt.savefig(f"{self.path}_ELBO.png")
        plt.show()

    def plot_training_KLD(self, avg_train_KLD, avg_val_KLD):
        avg_train_KLD = np.stack(avg_train_KLD)
        avg_val_KLD = np.stack(avg_val_KLD)

        plt.title(f"{type(self.method).__name__} {self.experiment} avg. training/validation KL divergence")
        epochs, train_groups = avg_train_KLD.shape
        X = np.arange(1, epochs + 1)

        for g_i in range(train_groups):
            plt.plot(X, avg_train_KLD[:, g_i], label=f"Train (Z{g_i+1})")

        plt.plot(X, avg_val_KLD[:, 0], label="Validation")

        plt.legend()
        plt.savefig(f"{self.path}_KLD.png")
        plt.show()
