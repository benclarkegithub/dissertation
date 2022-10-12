from os.path import exists
import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np

"""
Only works when trained with type="Single"...
"""
class Experiments:
    def __init__(self, dataset, experiments, size, channels):
        self.dataset = dataset
        self.experiments = experiments
        # To get reconstruction error (/dim) and ELBO (bits/dim)
        self.denominator = (size ** 2) * channels
        self.denominator_bits = self.denominator * torch.log(torch.tensor(2))

    def load_experiments(self):
        self.training = []
        self.training_stacked = []

        for i, experiment in enumerate(self.experiments):
            training = []

            for trial in range(experiment["trials"]):
                path = experiment["identifier"] + f"/{trial}"

                training.append(self.load_training(path))

            self.training.append(training)

            # We can't stack them twice because different trials might have different numbers of epochs
            self.training_stacked.append({
                "avg_train_loss": [np.hstack(x["avg_train_loss"]) for x in self.training[i]],
                "avg_train_log_prob": [np.hstack(x["avg_train_log_prob"]) for x in self.training[i]],
                "avg_train_KLD": [np.hstack(x["avg_train_KLD"]) for x in self.training[i]],
                "grad": [np.hstack(x["grad"]) for x in self.training[i]],
                "avg_val_loss": [np.hstack(x["avg_val_loss"]) for x in self.training[i]],
                "avg_val_log_prob": [np.hstack(x["avg_val_log_prob"]) for x in self.training[i]],
                "avg_val_KLD": [np.hstack(x["avg_val_KLD"]) for x in self.training[i]],
                "best_avg_val_epoch": [x["best_avg_val_epoch"] for x in self.training[i]],
                "best_avg_val_loss": [x["best_avg_val_loss"] for x in self.training[i]],
                "current_model": [x["current_model"] for x in self.training[i]],
                "current_epoch": [x["current_epoch"] for x in self.training[i]],
                "epoch_time": [np.hstack(x["epoch_time"]) for x in self.training[i]]
            })

    def load_training(self, path):
        training_progress = None

        if exists(f"{path}/Training.pkl"):
            with open(f"{path}/Training.pkl", "rb") as file:
                training_progress = pickle.load(file)

        return training_progress

    def test(self, test_loader):
        self.testing = []

        for i, experiment in enumerate(self.experiments):
            avg_loss = []
            avg_log_prob = []
            avg_KLD = []

            for trial in range(experiment["trials"]):
                # Load the best model
                self.load(experiment, trial, verbose=True)

                avg_loss_2 = []
                avg_log_prob_2 = []
                avg_KLD_2 = []

                for i, data in enumerate(test_loader):
                    output, loss, log_prob, KLD = experiment["method"].test(i=i, data=data)

                    avg_loss_2.append(loss)
                    avg_log_prob_2.append(log_prob)
                    avg_KLD_2.append(KLD)

                avg_loss.append(np.array(avg_loss_2).mean(axis=0))
                avg_log_prob.append(np.array(avg_log_prob_2).mean(axis=0))
                avg_KLD.append(np.array(avg_KLD_2).mean(axis=0))
                
            self.testing.append({
                "avg_loss": avg_loss,
                "avg_log_prob": avg_log_prob,
                "avg_KLD": avg_KLD
            })

    def load(self, experiment, trial, verbose=False):
        path = experiment["identifier"] + f"/{trial}/Parameters/Best"

        # Check that the path exists and load the model if it does
        if exists(f"{path}.pth"):
            if verbose:
                print(f"Loading model... {path}")
            experiment["method"].load(path)
        else:
            if verbose:
                print(f"No model exists... {path}")

    def summary(self):
        # Set print options
        torch.set_printoptions(precision=3, profile="short", sci_mode=False, threshold=torch.inf)
        np.set_printoptions(precision=3, formatter={"all": lambda x: f"{x:.3f}"}, threshold=np.inf)

        for i, t_s in enumerate(self.training_stacked):
            best_train_ELBO = np.array([-x[:, -1].min() / self.denominator_bits for x in t_s["avg_train_loss"]])
            best_val_ELBO = np.array([-x[:, -1].min() / self.denominator_bits for x in t_s["avg_val_loss"]])
            best_train_log_prob = np.array([x[:, -1].max() / self.denominator for x in t_s["avg_train_log_prob"]])
            best_val_log_prob = np.array([x[:, -1].max() / self.denominator for x in t_s["avg_val_log_prob"]])
            best_train_KLD = np.vstack([x[t_s["best_avg_val_epoch"][i]] for i, x in enumerate(t_s["avg_train_KLD"])])
            best_val_KLD = np.vstack([x[t_s["best_avg_val_epoch"][i]] for i, x in enumerate(t_s["avg_val_KLD"])])

            best_test_ELBO = np.array([-x.item() / self.denominator_bits for x in self.testing[i]["avg_loss"]])
            best_test_log_prob = np.array([x.item() / self.denominator for x in self.testing[i]["avg_log_prob"]])
            best_test_KLD = np.array([x.item() for x in self.testing[i]["avg_KLD"]])

            message = f"##################################################\n" \
                      f"# {self.experiments[i]['name']}\n" \
                      f"## Training\n" \
                      f"Avg. best ELBO (bits/dim): " \
                      f"{best_train_ELBO.mean():.3f} +- {best_train_ELBO.std():.3f}\n" \
                      f"Avg. best reconstruction error (/dim): " \
                      f"{best_train_log_prob.mean():.3f} +- {best_train_log_prob.std():.3f}\n" \
                      f"Avg. best KL divergence: " \
                      f"{best_train_KLD.mean(axis=0)} +- {best_train_KLD.std(axis=0)}\n" \
                      f"## Validation\n" \
                      f"Avg. best ELBO (bits/dim): " \
                      f"{best_val_ELBO.mean():.3f} +- {best_val_ELBO.std():.3f}\n" \
                      f"Avg. best reconstruction error (/dim): " \
                      f"{best_val_log_prob.mean():.3f} +- {best_val_log_prob.std():.3f}\n" \
                      f"Avg. best KL divergence: " \
                      f"{best_val_KLD.mean():.3f} +- {best_val_KLD.std():.3f}\n" \
                      f"## Test\n" \
                      f"Avg. best ELBO (bits/dim): " \
                      f"{best_test_ELBO.mean():.3f} +- {best_test_ELBO.std():.3f}\n" \
                      f"Avg. best reconstruction error (/dim): " \
                      f"{best_test_log_prob.mean():.3f} +- {best_test_log_prob.std():.3f}\n" \
                      f"Avg. best KL divergence: " \
                      f"{best_test_KLD.mean():.3f} +- {best_test_KLD.std():.3f}\n" \
                      f"##################################################"

            print(message)

        # Set print options to default
        torch.set_printoptions()
        np.set_printoptions()

    def graphs(self, num_groups, max_epochs, *, start_epoch=1):
        X = np.arange(start_epoch, max_epochs + 1)

        # Loss
        avg_train_loss_avg = []
        avg_train_loss_std = []
        avg_val_loss_avg = []
        avg_val_loss_std = []

        # Reconstruction error
        avg_train_log_prob_avg = []
        avg_train_log_prob_std = []
        avg_val_log_prob_avg = []
        avg_val_log_prob_std = []

        # KL divergence
        avg_train_KLD_avg = []
        avg_train_KLD_std = []
        avg_val_KLD_avg = []
        avg_val_KLD_std = []

        for i, t_s in enumerate(self.training_stacked):
            avg_train_loss_temp = t_s["avg_train_loss"]
            avg_val_loss_temp = t_s["avg_val_loss"]
            avg_train_log_prob_temp = t_s["avg_train_log_prob"]
            avg_val_log_prob_temp = t_s["avg_val_log_prob"]
            avg_train_KLD_temp = t_s["avg_train_KLD"]
            avg_val_KLD_temp = t_s["avg_val_KLD"]

            # avg_train_loss_temp is size (trials, epochs, num_groups)
            # avg_val_loss_temp is size (trials, epochs)
            for j in range(len(avg_train_loss_temp)):
                epochs = avg_train_loss_temp[j].shape[0]

                if epochs < max_epochs:
                    # Loss
                    train_loss_copied = np.tile(avg_train_loss_temp[j][-1, :], (max_epochs - epochs, 1))
                    val_loss_copied = np.tile(avg_val_loss_temp[j][-1, :], (max_epochs - epochs, 1))
                    avg_train_loss_temp[j] = np.vstack([avg_train_loss_temp[j], train_loss_copied])
                    avg_val_loss_temp[j] = np.vstack([avg_val_loss_temp[j], val_loss_copied])

                    # Reconstruction error
                    train_log_prob_copied = np.tile(avg_train_log_prob_temp[j][-1, :], (max_epochs - epochs, 1))
                    val_log_prob_copied = np.tile(avg_val_log_prob_temp[j][-1, :], (max_epochs - epochs, 1))
                    avg_train_log_prob_temp[j] = np.vstack([avg_train_log_prob_temp[j], train_log_prob_copied])
                    avg_val_log_prob_temp[j] = np.vstack([avg_val_log_prob_temp[j], val_log_prob_copied])

                    # KL divergence
                    train_KLD_copied = np.tile(avg_train_KLD_temp[j][-1, :], (max_epochs - epochs, 1))
                    val_KLD_copied = np.tile(avg_val_KLD_temp[j][-1, :], (max_epochs - epochs, 1))
                    avg_train_KLD_temp[j] = np.vstack([avg_train_KLD_temp[j], train_KLD_copied])
                    avg_val_KLD_temp[j] = np.vstack([avg_val_KLD_temp[j], val_KLD_copied])

            # Loss
            avg_train_loss_avg.append(np.array(avg_train_loss_temp).mean(axis=0))
            avg_train_loss_std.append(np.array(avg_train_loss_temp).std(axis=0))
            avg_val_loss_avg.append(np.array(avg_val_loss_temp).mean(axis=0))
            avg_val_loss_std.append(np.array(avg_val_loss_temp).std(axis=0))

            # Reconstruction error
            avg_train_log_prob_avg.append(np.array(avg_train_log_prob_temp).mean(axis=0))
            avg_train_log_prob_std.append(np.array(avg_train_log_prob_temp).std(axis=0))
            avg_val_log_prob_avg.append(np.array(avg_val_log_prob_temp).mean(axis=0))
            avg_val_log_prob_std.append(np.array(avg_val_log_prob_temp).std(axis=0))

            # KL divergence
            avg_train_KLD_avg.append(np.array(avg_train_KLD_temp).mean(axis=0))
            avg_train_KLD_std.append(np.array(avg_train_KLD_temp).std(axis=0))
            avg_val_KLD_avg.append(np.array(avg_val_KLD_temp).mean(axis=0))
            avg_val_KLD_std.append(np.array(avg_val_KLD_temp).std(axis=0))

        # ELBO
        ax = plt.gca()
        for i, y in enumerate(avg_train_loss_avg):
            if "ELBO" in self.experiments[i]["graphs"]:
                colour = next(ax._get_lines.prop_cycler)['color']

                train_label = self.experiments[i]['abbreviation'] + " (train)"
                val_label = self.experiments[i]['abbreviation'] + " (val.)"

                plt.plot(X, -y[start_epoch-1:, -1] / self.denominator_bits, color=colour, label=train_label)
                plt.plot(X, -avg_val_loss_avg[i][start_epoch-1:] / self.denominator_bits, "--", color=colour, label=val_label)

        plt.title(f"{self.dataset} ELBO")
        plt.xlabel("Epoch")
        plt.ylabel("Avg. ELBO (bits/dim)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.dataset}_ELBO.png", dpi=300)
        plt.show(dpi=300)

        # ELBO Training (Z1)
        for i, y in enumerate(avg_train_loss_avg):
            if "ELBO_Z1" in self.experiments[i]["graphs"]:
                plt.plot(X, -y[start_epoch-1:, 0] / self.denominator_bits, label=self.experiments[i]['abbreviation'])

        plt.title(f"{self.dataset} ELBO (Z1)")
        plt.xlabel("Epoch")
        plt.ylabel("Avg. ELBO (bits/dim)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.dataset}_ELBO_Z1.png", dpi=300)
        plt.show(dpi=300)

        # Reconstruction error
        for i, y in enumerate(avg_train_log_prob_avg):
            if "R" in self.experiments[i]["graphs"]:
                colour = next(ax._get_lines.prop_cycler)['color']

                train_label = self.experiments[i]['abbreviation'] + " (train)"
                val_label = self.experiments[i]['abbreviation'] + " (val.)"

                plt.plot(X, y[start_epoch-1:, -1] / self.denominator, color=colour, label=train_label)
                plt.plot(X, avg_val_log_prob_avg[i][start_epoch - 1:] / self.denominator, "--", color=colour, label=val_label)

        plt.title(f"{self.dataset} Reconstruction Error")
        plt.xlabel("Epoch")
        plt.ylabel("Avg. reconstruction error (/dim)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.dataset}_R.png", dpi=300)
        plt.show(dpi=300)

        # Reconstruction error Training (Z1)
        for i, y in enumerate(avg_train_log_prob_avg):
            if "R_Z1" in self.experiments[i]["graphs"]:
                plt.plot(X, y[start_epoch-1:, 0] / self.denominator, label=self.experiments[i]['abbreviation'])

        plt.title(f"{self.dataset} Reconstruction Error (Z1)")
        plt.xlabel("Epoch")
        plt.ylabel("Avg. reconstruction error (/dim)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.dataset}_R_Z1.png", dpi=300)
        plt.show(dpi=300)

        # KL divergence (Z=10/16)
        for i, y in enumerate(avg_train_KLD_avg):
            if "KLD_Z_10_16" in self.experiments[i]["graphs"]:
                plt.plot(X, y[start_epoch-1:, -1], label=self.experiments[i]['abbreviation'])

        plt.title(f"{self.dataset} KL Divergence (Z{num_groups})")
        plt.xlabel("Epoch")
        plt.ylabel("Avg. KL divergence")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.dataset}_KL_Z_10_16.png", dpi=300)
        plt.show(dpi=300)

        # KL divergence Training (Z=1)
        for i, y in enumerate(avg_train_KLD_avg):
            if "KLD_Z_1" in self.experiments[i]["graphs"]:
                plt.plot(X, y[start_epoch-1:, 0], label=self.experiments[i]['abbreviation'])

        plt.title(f"{self.dataset} KL Divergence (Z1)")
        plt.xlabel("Epoch")
        plt.ylabel("Avg. KL divergence")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.dataset}_KL_Z_1.png", dpi=300)
        plt.show(dpi=300)

        # KL divergence Validation (Z<=10/16)
        for i, y in enumerate(avg_val_KLD_avg):
            if "KLD_Validation" in self.experiments[i]["graphs"]:
                plt.plot(X, y[start_epoch-1:], label=self.experiments[i]['abbreviation'])

        plt.title(f"{self.dataset} KL Divergence Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Avg. KL divergence")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.dataset}_KL_Validation.png", dpi=300)
        plt.show(dpi=300)
