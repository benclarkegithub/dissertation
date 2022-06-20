import numpy as np
import matplotlib.pyplot as plt


class Evaluate:
    def __init__(self, method, experiment):
        self.method = method
        self.experiment = experiment
        self.path = f"{type(method).__name__}_{experiment}"

    def train(self, train_loader, val_loader, max_epochs, max_no_improvement):
        # Try and load the model
        self.method.load(f"{self.path}.pth")

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
                self.method.save(f"{self.path}.pth")
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

    def test(self):
        pass

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
