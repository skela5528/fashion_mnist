import os
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import OrderedDict
import torch
import torch.cuda
import torch.nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchsummary import summary

from networks import Net3Conv, Net9Conv, get_efficientnet_pretrained_on_imagenet, get_resnet18


def time_format(time_in_seconds):
    return time.strftime("%M:%S", time.gmtime(time_in_seconds))


def params_format(n_params):
    params_k = int(round(n_params / 1000))
    params_k_str = "{:04d}K".format(params_k)
    return params_k_str


class DataHandler:
    """F-MNIST data utils."""

    FMNIST_DATA_DIR = "../data"
    N_WORKERS = 4

    @classmethod
    def preprocess(cls, augmentations=False, padding_to_32=False):
        """Preprocess pipeline. Options: padding, norm by /255, rotate, translate, scale, horizontal flip."""
        transforms_list = list()
        if padding_to_32:
            # min dimension supported by efficientnet is 32x32
            transforms_list.append(transforms.Pad(padding=2, padding_mode="edge"))
        if augmentations:
            # transforms_list.append(transforms.RandomCrop(32 if padding_to_32 else 28))
            transforms_list.append(transforms.RandomAffine(degrees=0, translate=(.15, .15), scale=(.90, 1.10)))
            transforms_list.append(transforms.RandomHorizontalFlip())
            # inverse with p=.5
            # transforms_list.append(transforms.Lambda(lambda x: 1 - x if int(time.time()) % 2 == 0 else x))
        transforms_list.append(transforms.ToTensor())
        return transforms.Compose(transforms_list)

    @classmethod
    def get_train_dataloader(cls, batch_size, transform) -> DataLoader:
        data = FashionMNIST(train=True, root=cls.FMNIST_DATA_DIR, transform=transform, download=True)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=cls.N_WORKERS)
        return dataloader

    @classmethod
    def get_validation_data(cls, n_samples=1000, transform=None) -> (torch.Tensor, torch.Tensor):
        # prep = cls.preprocess(augmentations=False, padding_to_32=padding_to_32)
        test_data = FashionMNIST(train=False, root=cls.FMNIST_DATA_DIR, transform=transform, download=True)
        dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=True, num_workers=cls.N_WORKERS)
        validation_input, validation_labels = iter(dataloader).next()
        # print(Counter(validation_labels.numpy()))
        return validation_input.cuda(), validation_labels.cuda()

    @classmethod
    def get_test_dataloader(cls, transform) -> (torch.Tensor, torch.Tensor):
        # prep = cls.preprocess(augmentations=False, padding_to_32=padding_to_32)
        test_data = FashionMNIST(train=False, root=cls.FMNIST_DATA_DIR, transform=transform, download=True)
        test_loader = DataLoader(test_data, batch_size=1000, num_workers=cls.N_WORKERS)
        return test_loader

    @classmethod
    def get_class_names(cls):
        # ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        data = FashionMNIST(train=True, root=cls.FMNIST_DATA_DIR, download=True)
        return data.classes


class Benchmarker:
    """Quality evaluation and performance benchmarking."""

    @staticmethod
    def get_accuracy_statistics(predictions: torch.Tensor, labels: torch.Tensor):
        """Return accuracy metric and number of correct predictions."""
        predicted_classes = predictions.argmax(dim=1)
        assert predicted_classes.nelement() == labels.nelement(), "Predictions and Labels have different lengths!!"
        num_correct = float(sum(predicted_classes == labels.cuda()))
        acc = num_correct / float(labels.nelement())
        return acc, num_correct

    @classmethod
    def evaluate(cls, model: nn.Module, input_tensor, labels, criterion, description="", verbose=True):
        model.eval()
        with torch.no_grad():
            predictions = model(input_tensor)
            loss = criterion(predictions, labels)
            acc, _ = cls.get_accuracy_statistics(predictions, labels)
            n = len(labels)
        if verbose:
            print("\n{}".format("=" * 50))
            print("[{:^17s}] || Loss={:.4f}  Acc={:4.1f}%  n={}".
                  format(description.upper(), loss.detach().item(), acc * 100, n))
        return loss, acc

    @classmethod
    def evaluate_test_data(cls, model: nn.Module, criterion):
        model.eval()

        padding_to_32 = True if model.__class__.__name__ == "EfficientNet" else False
        prep_default = DataHandler.preprocess(augmentations=False, padding_to_32=padding_to_32)
        prep_augment = DataHandler.preprocess(augmentations=True, padding_to_32=padding_to_32)
        dataloaders_list = list()
        dataloaders_list.append(DataHandler.get_test_dataloader(transform=prep_default))
        dataloaders_list.append(DataHandler.get_test_dataloader(transform=prep_augment))

        n = len(dataloaders_list[0].dataset)
        acc_list, loss_list = [], []
        with torch.no_grad():
            for dlid, data_loader in enumerate(dataloaders_list):
                num_correct = 0
                losses = []
                for test_input, test_labels in data_loader:
                    test_labels = test_labels.cuda()
                    predictions = model(test_input.cuda())
                    loss = criterion(predictions, test_labels)
                    losses.append(loss.detach().item())
                    _, num_correct_in_batch = cls.get_accuracy_statistics(predictions, test_labels)
                    num_correct += num_correct_in_batch
                test_loss = np.mean(losses)
                test_acc = num_correct / n
                acc_list.append(test_acc)
                loss_list.append(test_loss)
                print("[{:^17s}] || Loss={:.4f}  Acc={:4.2f}%  n={}".
                      format("TEST EVAL- [{:}]".format(dlid), test_loss, test_acc * 100, n))
        return min(loss_list), max(acc_list)

    @staticmethod
    def run_net_summary(model, input_shape=(1, 28, 28)):
        net_class_name = model.__class__.__name__
        print("\n\n[Net Summary] - {}".format(net_class_name))
        summary(model, input_shape)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(model)
        return total_params

    @staticmethod
    def get_inference_time(model: nn.Module):
        batch_size = 1000
        n_iters = 10
        padding_to_32 = True if model.__class__.__name__ == "EfficientNet" else False
        prep = DataHandler.preprocess(augmentations=False, padding_to_32=padding_to_32)
        data = DataHandler.get_train_dataloader(batch_size=batch_size, transform=prep)
        times_per_iter = []
        for iter_id, (x, _) in enumerate(data):
            if iter_id >= n_iters:
                break
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x.cuda())
            end.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            t = start.elapsed_time(end)
            times_per_iter.append(t)
        t_mean = np.mean(times_per_iter)
        t_std = np.std(times_per_iter)
        print("[{:^17s}] || t_mean={:.1f}[millisec]  t_std={:.1f}[millisec]  n_iters={}".
              format("TIMING per 1k", t_mean, t_std, n_iters))
        return t_mean, t_std

    @classmethod
    def run_full_benchmark(cls, model: nn.Module, verbose=False):
        print("\nStarting full_benchmark . . .")
        loss_criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = cls.evaluate_test_data(model, loss_criterion)
        inference_time = cls.get_inference_time(model)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results = OrderedDict()
        results["model"] = model.__class__.__name__
        results["test_acc"] = test_acc
        results["test_cross_entropy"] = test_loss
        results["n_params"] = n_params
        results["inference_time_mean"] = inference_time[0]
        results["inference_time_std"] = inference_time[1]
        if verbose:
            print(results)
        return results


class Trainer:
    """Networks training utils."""

    def __init__(self, lr, n_epochs=10, momentum=0.9, weight_decay=1e-5, lr_step=60, out_dir="../fmnist_out", exp_name=""):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.lr_step = lr_step
        self.out_dir = out_dir
        self.exp_name = exp_name
        self.best_model_val_acc = 0

    def __get_sgd_optimizer(self, model: nn.Module) -> optim.SGD:
        optimizer = optim.SGD(params=model.parameters(),
                              lr=self.lr,
                              momentum=self.momentum,
                              weight_decay=self.weight_decay)
        return optimizer

    def __get_rmsprop_optimizer(self, model: nn.Module) -> optim.RMSprop:
        optimizer = optim.RMSprop(params=model.parameters(),
                                  lr=self.lr,
                                  momentum=self.momentum,
                                  weight_decay=self.weight_decay)

    def __aggregate_epoch_stats(self, loss_per_batch, n_correct_epoch, n_training_data, time_elapsed):
        epoch_loss = np.mean(loss_per_batch)
        epoch_acc = float(n_correct_epoch) / n_training_data
        t = time_format(time_elapsed)
        print("\n{}".format("-" * 50))
        print("[TRAIN - epoch avg] || Loss={:.4f}  Acc={:4.1f}%  T-{}| Best val_acc={:4.1f}%".
              format(epoch_loss, epoch_acc * 100, t, self.best_model_val_acc * 100))
        return epoch_loss, epoch_acc

    def __adjust_learning_rate(self, optimizer, epoch, step=60):
        if (epoch + 1) % step == 0:
            self.lr = self.lr * .2
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr

    @staticmethod
    def __get_learning_rate(optimizer):
        return [param_group['lr'] for param_group in optimizer.param_groups]

    def _update_best_model(self, model, val_acc):
        if val_acc > self.best_model_val_acc:
            self.best_model_val_acc = val_acc
            self.save_model(model, time_stamp="BEST", test_acc=0)

    def train(self, model: nn.Module, dataloader: DataLoader, validation_data=None, verbose_batch=True):
        # optimizer = self.__get_sgd_optimizer(model)
        optimizer = self.__get_sgd_optimizer(model)
        criterion = nn.CrossEntropyLoss()
        name = model.__class__.__name__
        train_loss, validation_loss = [], []
        train_acc, validation_acc = [], []
        time_start = time.time()
        time_elapsed = 0
        model.train()

        # for each epoch
        print("\n\n[Net Training] - {}".format(name))
        for eid in range(self.n_epochs):
            self.__adjust_learning_rate(optimizer, eid, self.lr_step)
            print("\n\n{}".format("=" * 50))
            print("EPOCH {:3d} | LR={:}".format(eid, self.__get_learning_rate(optimizer)))
            loss_per_batch = []
            acc_per_batch = []
            n_correct_in_epoch = 0

            # for each batch in epoch
            for bid, (inputs, labels) in enumerate(dataloader):
                # # TODO REMOVE
                # if bid > 2:
                #     break
                # print(inputs.shape)

                optimizer.zero_grad()
                # get predictions
                labels = labels.cuda()
                predictions = model(inputs.cuda())
                loss = criterion(predictions, labels)
                # update grads
                loss.backward()
                optimizer.step()
                # update training stats
                acc, n_correct = Benchmarker.get_accuracy_statistics(predictions, labels)
                loss_per_batch.append(loss.detach().item())
                acc_per_batch.append(acc)
                n_correct_in_epoch += n_correct
                time_elapsed = time.time() - time_start
                if verbose_batch and bid % 10 == 0:
                    print(" * batch -{:3d} || Loss={:.4f}  TrainAcc={:4.1f}%  T-{:s}".
                          format(bid, loss_per_batch[-1], acc * 100, time_format(time_elapsed)))

            # train stat per epoch
            n = len(dataloader.dataset)
            e_loss, e_acc = self.__aggregate_epoch_stats(loss_per_batch, n_correct_in_epoch, n, time_elapsed)
            train_loss.append(e_loss)
            train_acc.append(e_acc)

            # validation stats once in an epoch
            if not validation_data:
                continue
            val_loss, val_acc = Benchmarker.evaluate(model, *validation_data, criterion, description="validation")
            self._update_best_model(model, val_acc)

            model.train()
            validation_loss.append(val_loss)
            validation_acc.append(val_acc)

        # training finished - plot, evaluate test, save model
        time_stamp = time.strftime("%b%d_%H%M", time.localtime(time.time()))
        model.eval()
        del validation_data, inputs, labels
        torch.cuda.empty_cache()

        bench_results = Benchmarker.run_full_benchmark(model)
        self.plot_learning_stats(train_loss, validation_loss, train_acc, validation_acc, time_stamp, name)
        self.save_model(model, time_stamp, bench_results["test_acc"], bench_results["inference_time_mean"])

    def plot_learning_stats(self, train_loss, validation_loss, train_acc, validation_acc, time_stamp="", model_name=""):
        colors = ["tab:blue", "tab:orange"]

        os.makedirs(self.out_dir, exist_ok=True)
        n_epochs = len(train_loss)
        assert len(train_loss) == len(validation_loss) == len(train_acc) == len(validation_acc)
        x = range(1, n_epochs + 1, 1)
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

        # loss
        axs[0].plot(x, train_loss, color=colors[0], alpha=.8, marker="o", label="train", ms=3)
        axs[0].plot(x, validation_loss, color=colors[1], alpha=.8, marker="s", label="validation", ms=3)
        axs[0].set_ylim([0, 2.5])
        axs[0].grid(True)
        axs[0].legend()
        axs[0].set_title("Loss")

        # acc
        axs[1].plot(x, train_acc, color=colors[0], alpha=.8, marker="o", label="train", lw=1, ms=3)
        axs[1].plot(x, validation_acc, color=colors[1], alpha=.8, marker="s", label="validation", lw=1, ms=3)
        axs[1].plot(x, [.95] * n_epochs, color="tab:gray", alpha=.7, ls="--", lw=1)
        axs[1].plot(x, [.90] * n_epochs, color="tab:gray", alpha=.5, ls="--", lw=1)
        axs[1].grid(True)
        axs[1].legend()
        axs[1].set_title("Accuracy")
        axs[1].set_xlabel("#epoch")

        # save plot
        plot_path = os.path.join(self.out_dir, "plot_{}_{}.png".format(model_name, time_stamp))
        fig.set_size_inches(8, 8)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=80)

    def save_model(self, model: nn.Module, time_stamp, test_acc, inf_time=-1.0):
        os.makedirs(self.out_dir, exist_ok=True)
        name = model.__class__.__name__
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_params = params_format(n_params)
        model_path = os.path.join(self.out_dir,
                                  "model-{:}_{:}_epochs-{:03d}_acc-{:5.3f}_params-{:}_t-{:05.1f}_exp-{:}.pth".
                                  format(name, time_stamp, self.n_epochs, test_acc, n_params, inf_time, self.exp_name))
        print("\nSaving model to {} ..".format(model_path))
        torch.save(model.state_dict(), model_path)

    @staticmethod
    def load_model(model_path: str, model_instance: nn.Module) -> nn.Module:
        model_dict = torch.load(model_path)
        model_instance.load_state_dict(model_dict)
        model_instance.eval()
        print("Model *{}* is loaded from {}".format(model_instance.__class__.__name__, model_path))
        return model_instance


# #################################################################################################################### #

def get_model(model_name):
    model = None
    if model_name =="Net3Conv":
        model = Net3Conv()
    elif model_name == "Net9Conv":
        model = Net9Conv()
    elif model_name == "EfficientNet":
        model = get_efficientnet_pretrained_on_imagenet()
    elif model_name == "resnet-18":
        model = get_resnet18()
    else:
        print("Model - {} is not supported! Exit!".format(model_name))
        exit(-1)
    return model.cuda()


def run_training(train_config):
    # get model
    model = get_model(train_config["model_name"])
    padding_to_32 = True if model.__class__.__name__ == "EfficientNet" else False
    prep = DataHandler.preprocess(augmentations=train_config["augmentations"], padding_to_32=padding_to_32)

    print(model)
    summary(model, (1, 28, 28))

    # data
    train_dataloader = DataHandler.get_train_dataloader(batch_size=1000, transform=prep)
    validation_data = DataHandler.get_validation_data(transform=prep, n_samples=3000)

    # train
    trainer = Trainer(lr=train_config["lr"], n_epochs=train_config["n_epochs"], exp_name="")
    trainer.train(model, train_dataloader, validation_data, verbose_batch=True)


TRAIN_CONFIG = dict()
TRAIN_CONFIG["model_name"] = "resnet-18"
TRAIN_CONFIG["n_epochs"] = 16
TRAIN_CONFIG["lr"] = 1e-1  # 1e-1 * 2
TRAIN_CONFIG["augmentations"] = True


if __name__ == '__main__':
    run_training(TRAIN_CONFIG)
    print("\n Done ..")
