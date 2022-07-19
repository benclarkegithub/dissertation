from os.path import exists
import pickle
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from Utils import make_dir

class Evaluate:
    def __init__(self, method, experiment, *, log=True, seed=None):
        self.method = method
        self.experiment = experiment
        self.path = f"{experiment}/{type(method).__name__}_{experiment}"
        self.log = log
        if seed is not None:
            self.seed_everything(seed)

    def seed_everything(self, seed):
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def train(self, train_loader, val_loader, max_epochs, max_no_improvement, *, get_grad=False):
        # Try and load the model(s) and training progress
        self.load(best=False, verbose=True)
        training_progress = self.load_training()

        # Determine the number of models to train
        models = 1 if self.method.get_type() == "Single" else self.method.get_num_groups()

        if training_progress is None:
            # If this is the first time training, log a summary of the model(s)
            self.log_summary()

            avg_train_loss = [[] for _ in range(models)]
            avg_train_log_prob = [[] for _ in range(models)]
            avg_train_KLD = [[] for _ in range(models)]
            grad = [[] for _ in range(models)]
            avg_val_loss = [[] for _ in range(models)]
            avg_val_log_prob = [[] for _ in range(models)]
            avg_val_KLD = [[] for _ in range(models)]
            best_avg_val_epoch = [1 for _ in range(models)]
            best_avg_val_loss = [np.inf for _ in range(models)]
            current_model = 0
            current_epoch =  0
            epoch_time = [[] for _ in range(models)]
        else:
            avg_train_loss = training_progress["avg_train_loss"]
            avg_train_log_prob = training_progress["avg_train_log_prob"]
            avg_train_KLD = training_progress["avg_train_KLD"]
            grad = training_progress["grad"]
            avg_val_loss = training_progress["avg_val_loss"]
            avg_val_log_prob = training_progress["avg_val_log_prob"]
            avg_val_KLD = training_progress["avg_val_KLD"]
            best_avg_val_epoch = training_progress["best_avg_val_epoch"]
            best_avg_val_loss = training_progress["best_avg_val_loss"]
            current_model = training_progress["current_model"]
            current_epoch = training_progress["current_epoch"]
            epoch_time = training_progress["epoch_time"]

        for model in range(current_model, models):
            if self.method.get_type() == "Multiple":
                self.write_log(f"Training model {model+1}/{models}...")

            for epoch in range(current_epoch+1, max_epochs+1):
                # Train the model and get the average training loss
                avg_train_loss_2 = []
                avg_train_log_prob_2 = []
                avg_train_KLD_2 = []
                grad_2 = []

                # Keep track of the training time
                start = time.time()

                for i, data in enumerate(train_loader):
                    # If the method has multiple models, pass the model number
                    if self.method.get_type() == "Single":
                        losses, log_probs, KLDs, grads = self.method.train(i=i, data=data, get_grad=get_grad)
                    else:
                        losses, log_probs, KLDs, grads = self.method.train(
                            i=i, data=data, get_grad=get_grad, model=model)
                    avg_train_loss_2.append(losses)
                    avg_train_log_prob_2.append(log_probs)
                    avg_train_KLD_2.append(KLDs)
                    if get_grad:
                        grad_2.append(grads)

                # Keep track of the training time
                end = time.time()
                epoch_time[model].append(end - start)

                avg_train_loss[model].append(np.array(avg_train_loss_2).mean(axis=0))
                avg_train_log_prob[model].append(np.array(avg_train_log_prob_2).mean(axis=0))
                avg_train_KLD[model].append(np.array(avg_train_KLD_2).mean(axis=0))
                if get_grad:
                    grad[model].append(np.array(grad_2).mean(axis=0))

                # Get the trained model's average validation loss
                avg_val_loss_2 = []
                avg_val_log_prob_2 = []
                avg_val_KLD_2 = []

                for i, data in enumerate(val_loader):
                    # If the method has multiple models pass the model number
                    if self.method.get_type() == "Single":
                        output, losses, log_probs, KLDs = self.method.test(i=i, data=data)
                    else:
                        output, losses, log_probs, KLDs = self.method.test(i=i, data=data, model=model)
                    avg_val_loss_2.append(losses)
                    avg_val_log_prob_2.append(log_probs)
                    avg_val_KLD_2.append(KLDs)

                avg_val_loss[model].append(np.array(avg_val_loss_2).mean(axis=0))
                avg_val_log_prob[model].append(np.array(avg_val_log_prob_2).mean(axis=0))
                avg_val_KLD[model].append(np.array(avg_val_KLD_2).mean(axis=0))

                # Log epoch information
                grad_param = grad[model][-1] if get_grad else None
                self.log_epoch(
                    epoch=epoch,
                    epoch_time=epoch_time[model][-1],
                    avg_train_losses=avg_train_loss[model][-1],
                    avg_val_losses=avg_val_loss[model][-1],
                    avg_train_log_probs=avg_train_log_prob[model][-1],
                    avg_val_log_probs=avg_val_log_prob[model][-1],
                    avg_train_KLDs=avg_train_KLD[model][-1],
                    avg_val_KLDs=avg_val_KLD[model][-1],
                    grad=grad_param)

                # Save training progress
                self.save(best=False)
                self.save_training(avg_train_loss=avg_train_loss,
                                   avg_train_log_prob=avg_train_log_prob,
                                   avg_train_KLD=avg_train_KLD,
                                   grad=grad,
                                   avg_val_loss=avg_val_loss,
                                   avg_val_log_prob=avg_val_log_prob,
                                   avg_val_KLD=avg_val_KLD,
                                   best_avg_val_epoch=best_avg_val_epoch,
                                   best_avg_val_loss=best_avg_val_loss,
                                   current_model=model,
                                   current_epoch=epoch,
                                   epoch_time=epoch_time)

                # Check if the average validation loss is the best
                if avg_val_loss[model][-1][-1] < best_avg_val_loss[model]:
                    best_avg_val_epoch[model] = epoch
                    best_avg_val_loss[model] = avg_val_loss[model][-1][-1]
                    # Save the model
                    self.save(best=True)
                elif (epoch - best_avg_val_epoch[model] + 1) == max_no_improvement:
                    self.write_log(f"No improvement after {max_no_improvement} epochs...")
                    break

        # Log training stats
        self.log_training(
            best_avg_val_epoch=best_avg_val_epoch,
            epoch_time=epoch_time,
            avg_train_loss=avg_train_loss,
            avg_val_loss=avg_val_loss,
            avg_train_log_prob=avg_train_log_prob,
            avg_val_log_prob=avg_val_log_prob
        )

        # Plot training/validation ELBO and KL divergence
        self.plot_training_ELBO(avg_train_loss=avg_train_loss, avg_val_loss=avg_val_loss)
        self.plot_training_KLD(avg_train_KLD=avg_train_KLD, avg_val_KLD=avg_val_KLD)
        if get_grad:
            self.plot_grad(grad=grad)

    def test(self,
             test_loader,
             *,
             avg_var=True,
             output_images=True,
             output_images_opt=None,
             conceptual_compression=True,
             conceptual_compression_opt=None):
        # Try and load the best model
        self.load(best=True, verbose=True)

        if avg_var:
            # Calculate mu, var, z mean and variance
            mu = None
            logvar = None

            for i, data in enumerate(test_loader):
                output, loss, log_prob, KLD = self.method.test(i=i, data=data)

                if (i == 0):
                    mu = output["mu"]
                    logvar = output["logvar"]
                else:
                    mu = torch.vstack([mu, output["mu"]])
                    logvar = torch.vstack([logvar, output["logvar"]])

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

        # This is only compatible with the Single method
        if conceptual_compression:
            if not conceptual_compression_opt:
                # Set conceptual compression options if it doesn't exist
                conceptual_compression_opt = { "number": 8, "size": 28, "random": True, "separate": True }

            # Get conceptual compression options
            number = conceptual_compression_opt["number"]
            size = conceptual_compression_opt["size"]
            random_opt = conceptual_compression_opt["random"]
            separate = conceptual_compression_opt["separate"]
            rows = (1 + self.method.get_num_groups()) # Original and each group
            columns = number
            height = size * rows
            width = size * columns

            # Get the data
            data = next(iter(test_loader))
            images, labels = data

            # Start making grid of images
            images = images[:number].unsqueeze(dim=1).reshape(number, 1, size, size)

            # Get the output
            output, loss, log_prob, KLD = self.method.test(i=0, data=data)
            mu = output["mu"]

            def get_images_from_order(images, order):
                images_temp = images.detach().clone()
                z_decs = []
                groups = []

                for group in order:
                    # For each group, we need to get z_decs and pass it to the decoder
                    # Start is inclusive, end is not
                    start = group * self.method.get_num_latents_group()
                    end = (group * self.method.get_num_latents_group()) + self.method.get_num_latents_group()

                    if self.method.get_type() == "Single":
                        if not len(z_decs):
                            z_decs.append(self.method.z_to_z_dec(mu[:number, start:end], group=group))
                        else:
                            z_decs[0] = z_decs[0] + self.method.z_to_z_dec(mu[:number, start:end], group=group)

                        logits = self.method.z_dec_to_logits(z_decs[0])
                    else:
                        z_decs.append(self.method.z_to_z_dec(mu[:number, start:end], group=group))
                        groups.append(group)

                        logits = self.method.z_decs_to_logits(z_decs, groups)

                    # Add to grid of images
                    logits_images = torch.sigmoid(logits).unsqueeze(dim=1).reshape(number, 1, size, size)
                    images_temp = torch.vstack([images_temp, logits_images])

                return images_temp

            def get_separate_images_from_order(images, order):
                images_temp = images.detach().clone()

                for group in order:
                    # For each group, we need to get z_decs and pass it to the decoder
                    # Start is inclusive, end is not
                    start = group * self.method.get_num_latents_group()
                    end = (group * self.method.get_num_latents_group()) + self.method.get_num_latents_group()

                    z_dec = self.method.z_to_z_dec(mu[:number, start:end], group=group)

                    if self.method.get_type() == "Single":
                        logits = self.method.z_dec_to_logits(z_dec)
                    else:
                        logits = self.method.z_decs_to_logits([z_dec], [group])

                    # Add to grid of images
                    logits_images = torch.sigmoid(logits).unsqueeze(dim=1).reshape(number, 1, size, size)
                    images_temp = torch.vstack([images_temp, logits_images])

                return images_temp

            def images_to_graph(images, order, title, path):
                # Make grid of images
                images_grid = torchvision.utils.make_grid(tensor=images, nrow=number)

                # Create and display the 2D graph
                plt.title(title)
                # 0.95 is a hack to make the graph look good
                x_ticks = (np.linspace(start=0, stop=width, num=number) + (size // 2)) * 0.95
                y_ticks = (np.linspace(start=height, stop=0, num=self.method.get_num_groups()+1) + (size // 2)) * 0.95
                x_labels = [x.item() for x in labels[:number]]
                y_labels = [f"Z{i+1}" for i in order[::-1]] + ["Original"]
                plt.xticks(ticks=x_ticks, labels=x_labels)
                plt.yticks(ticks=y_ticks, labels=y_labels)
                plt.imshow(X=np.transpose(images_grid.numpy(), (1, 2, 0)))
                plt.savefig(path)
                plt.show()

            order = range(self.method.get_num_groups())
            images_temp = get_images_from_order(images, order)
            images_to_graph(
                images_temp,
                order,
                f"{type(self.method).__name__} {self.experiment} conceptual compression",
                f"{self.path}_Conceptual_Compression.png"
            )

            if random_opt:
                # Get random order
                order = [x for x in range(self.method.get_num_groups())]
                random.shuffle(order)
                images_temp = get_images_from_order(images, order)
                images_to_graph(
                    images_temp,
                    order,
                    f"{type(self.method).__name__} {self.experiment} conceptual compression (random order)",
                    f"{self.path}_Conceptual_Compression_Random.png"
                )

            if separate:
                # Get separate images
                order = range(self.method.get_num_groups())
                images_temp = get_separate_images_from_order(images, order)
                images_to_graph(
                    images_temp,
                    order,
                    f"{type(self.method).__name__} {self.experiment} conceptual compression (separate)",
                    f"{self.path}_Conceptual_Compression_Separate.png"
                )

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
                      current_model,
                      current_epoch,
                      epoch_time,
                      *,
                      grad):
        # Get the directory name and if it doesn't exist create it
        make_dir(self.path)

        training_progress = {
            "avg_train_loss": avg_train_loss,
            "avg_train_log_prob": avg_train_log_prob,
            "avg_train_KLD": avg_train_KLD,
            "grad": grad,
            "avg_val_loss": avg_val_loss,
            "avg_val_log_prob": avg_val_log_prob,
            "avg_val_KLD": avg_val_KLD,
            "best_avg_val_epoch": best_avg_val_epoch,
            "best_avg_val_loss": best_avg_val_loss,
            "current_model": current_model,
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
                  avg_val_KLDs,
                  grad):
        message = f"[Epoch {epoch:3} ({epoch_time:.2f}s)]\t" \
                  f"ELBO: {', '.join([f'{-x:.3f}' for x in avg_train_losses])}" \
                  f" ({', '.join([f'{-x:.3f}' for x in avg_val_losses])})\t" \
                  f"Log prob: {', '.join([f'{x:.3f}' for x in avg_train_log_probs])}" \
                  f" ({', '.join([f'{x:.3f}' for x in avg_val_log_probs])})\t" \
                  f"KLD: {', '.join([f'{x:.3f}' for x in avg_train_KLDs])}" \
                  f" ({', '.join([f'{x:.3f}' for x in avg_val_KLDs])})"

        if grad is not None:
            message += f"\tGrad: {', '.join([f'{x:.3f}' for x in grad])}"

        # Log message
        self.write_log(message)

    def log_training(self,
                     best_avg_val_epoch,
                     epoch_time,
                     avg_train_loss,
                     avg_val_loss,
                     avg_train_log_prob,
                     avg_val_log_prob):
        epoch_time_per_model = [sum(x) for x in epoch_time]
        epoch_time_total = sum(epoch_time_per_model)

        # Each of avg_train_loss, ..., avg_val_log_prob is a list of lists of NumPy arrays
        # Only the last dimensions are guaranteed to be the same size
        # For simplicity, only stack the list corresponding to the last model
        avg_train_loss = np.stack(avg_train_loss[-1])
        avg_val_loss = np.stack(avg_val_loss[-1])
        avg_train_log_prob = np.stack(avg_train_log_prob[-1])
        avg_val_log_prob = np.stack(avg_val_log_prob[-1])

        best_train_ELBO = -avg_train_loss[:, -1].min()
        best_val_ELBO = -avg_val_loss[:, -1].min()
        best_train_log_prob = avg_train_log_prob[:, -1].max()
        best_val_log_prob = avg_val_log_prob[:, -1].max()

        message = f"Best epoch(s): {best_avg_val_epoch}\t"\
                  f"Training time(s): {', '.join([f'{x:.2f}s' for x in epoch_time_per_model])} ({epoch_time_total:.2f}s)\t" \
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

    def plot_training_ELBO(self, avg_train_loss, avg_val_loss, *, avg_train_log_prob=None, avg_val_log_prob=None):
        avg_train_loss = [np.stack(x) for x in avg_train_loss]
        avg_val_loss = [np.stack(x) for x in avg_val_loss]
        if avg_train_log_prob is not None:
            avg_train_log_prob = [np.stack(x) for x in avg_train_log_prob]
        if avg_val_log_prob is not None:
            avg_val_log_prob = [np.stack(x) for x in avg_val_log_prob]

        models = len(avg_train_loss)
        for model in range(models):
            epochs, train_groups = avg_train_loss[model].shape
            X = np.arange(1, epochs+1)

            for g_i in range(train_groups):
                train_loss_label = f"Train ELBO (Z<={g_i+1})"
                train_log_prob_label = f"Train log-likelihood (Z<={g_i+1})"
                if models > 1:
                    train_loss_label = f"Train ELBO (Model {model+1}) (Z<={g_i+1})"
                    train_log_prob_label = f"Train log-likelihood (Model {model+1}) (Z<={g_i+1})"

                plt.plot(X, -avg_train_loss[model][:, g_i], label=train_loss_label)
                if avg_train_log_prob:
                    plt.plot(X, avg_train_log_prob[model][:, g_i], label=train_log_prob_label)

            val_loss_label = "Val. ELBO" if models == 1 else f"Val. ELBO (Model {model+1})"
            plt.plot(X, -avg_val_loss[model][:, 0], label=val_loss_label)
            if avg_val_log_prob:
                val_log_prob_label = "Val. log-likelihood" if models == 1 else f"Val. log-likelihood (Model {model+1})"
                plt.plot(X, avg_val_log_prob[model][:, 0], label=val_log_prob_label)

        plt.title(f"{type(self.method).__name__} {self.experiment} avg. training/validation ELBO")
        plt.legend()
        plt.savefig(f"{self.path}_ELBO.png")
        plt.show()

    def plot_training_KLD(self, avg_train_KLD, avg_val_KLD):
        # Training
        avg_train_KLD = [np.stack(x) for x in avg_train_KLD]

        models = len(avg_train_KLD)
        for model in range(models):
            epochs, train_groups = avg_train_KLD[model].shape
            X = np.arange(1, epochs+1)

            for g_i in range(train_groups):
                train_KLD_label = f"Z{g_i+1}" if models == 1 else f"Model {model+1} Z{g_i+1}"
                plt.plot(X, avg_train_KLD[model][:, g_i], label=train_KLD_label)

        plt.title(f"{type(self.method).__name__} {self.experiment} avg. training KL divergence")
        plt.legend()
        plt.savefig(f"{self.path}_KLD_train.png")
        plt.show()

        # Validation
        avg_val_KLD = [np.stack(x) for x in avg_val_KLD]

        models = len(avg_val_KLD)
        for model in range(models):
            epochs, train_groups = avg_val_KLD[model].shape
            X = np.arange(1, epochs + 1)

            val_KLD_label = "Validation" if models == 1 else f"Model {model+1}"
            plt.plot(X, avg_val_KLD[model][:, 0], label=val_KLD_label)

        plt.title(f"{type(self.method).__name__} {self.experiment} avg. validation KL divergence")
        plt.legend()
        plt.savefig(f"{self.path}_KLD_val.png")
        plt.show()

    def plot_grad(self, grad):
        grad = [np.stack(x) for x in grad]

        models = len(grad)
        for model in range(models):
            epochs, train_groups = grad[model].shape
            X = np.arange(1, epochs+1)

            for g_i in range(train_groups):
                grad_label = f"Z{g_i+1}" if models == 1 else f"Model {model+1} Z{g_i+1}"
                plt.plot(X, grad[model][:, g_i], label=grad_label)

        plt.title(f"{type(self.method).__name__} {self.experiment} avg. gradient norm")
        plt.legend()
        plt.savefig(f"{self.path}_grad.png")
        plt.show()
