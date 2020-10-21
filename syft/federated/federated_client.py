import torch as th
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
import numpy as np
import time
import pdb
from syft.generic.object_storage import ObjectStorage
from syft.federated.train_config import TrainConfig
from syft.federated.model_config import ModelConfig
from syft.frameworks.torch.fl.loss_fn import nll_loss
import torch.nn as nn

class FederatedClient(ObjectStorage):
    """A Client able to execute federated learning in local datasets."""

    def __init__(self, datasets=None):
        super().__init__()
        self.datasets = datasets if datasets is not None else dict()
        self.optimizer = None
        self.train_config = None
        self.model_config = None

    def add_dataset(self, dataset, key: str):
        if key not in self.datasets:
            self.datasets[key] = dataset
        else:
            raise ValueError(f"Key {key} already exists in Datasets")

    def remove_dataset(self, key: str):
        if key in self.datasets:
            del self.datasets[key]

    def set_obj(self, obj: object):
        """Registers objects checking if which objects it should cache.

        Args:
            obj: An object to be registered.
        """
        if isinstance(obj, TrainConfig):
            self.train_config = obj
            self.optimizer = None
        elif isinstance(obj, ModelConfig):
            print("[trace] ModelConfigSend recv COORD", time.time())
            self.model_config = obj
            self.optimizer = None
        else:
            if isinstance(obj.id, str):
                recv_obj_time = time.time()
                if obj.id[:11] == "GlobalModel":
                    print("[trace] GlobalModelSend recv COORD", recv_obj_time)
                elif obj.id == "LossFunc":
                    print("[trace] LossFuncSend recv COORD", recv_obj_time)
                elif len(obj.id) > 11 and obj.id[:10] == "Share_From":   ## len("Share_From_") == 11
                    id_str_list = obj.id.split("_")
                    worker_id = id_str_list[2]
                    obj_id = id_str_list[3]
                    print("[trace]", "GetShare_" + obj_id, "recv", worker_id, recv_obj_time)
            super().set_obj(obj)

    def _check_train_config(self):
        if self.train_config is None:
            raise ValueError("Operation needs TrainConfig object to be set.")

    def _check_model_config(self):
        if self.model_config is None:
            raise ValueError("Operation needs ModelConfig object to be set.")

    def _build_optimizer(
        self, optimizer_name: str, model, optimizer_args: dict
    ) -> th.optim.Optimizer:
        """Build an optimizer if needed.

        Args:
            optimizer_name: A string indicating the optimizer name.
            optimizer_args: A dict containing the args used to initialize the optimizer.
        Returns:
            A Torch Optimizer.
        """
        if self.optimizer is not None:
            return self.optimizer

        if optimizer_name in dir(th.optim):
            optimizer = getattr(th.optim, optimizer_name)
            optimizer_args.setdefault("params", model.parameters())
            self.optimizer = optimizer(**optimizer_args)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        return self.optimizer

    def fit(self, dataset_key: str, device: str = "cpu", **kwargs):
        """Fits a model on the local dataset as specified in the local TrainConfig object.

        Args:
            dataset_key: Identifier of the local dataset that shall be used for training.
            **kwargs: Unused.

        Returns:
            loss: Training loss on the last batch of training data.
        """
        self._check_train_config()

        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset {dataset_key} unknown.")

        model = self.get_obj(self.train_config._model_id).obj
        loss_fn = self.get_obj(self.train_config._loss_fn_id).obj

        self._build_optimizer(
            self.train_config.optimizer, model, optimizer_args=self.train_config.optimizer_args
        )

        return self._fit(model=model, dataset_key=dataset_key, loss_fn=loss_fn, device=device)

    def fit_mc(self, dataset_key: str, device: str = "cpu", **kwargs):

        self._check_model_config()
        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset {dataset_key} unknown.")

        model = self.get_obj(self.model_config._model_id)

        # loss_fn = self.get_obj(self.model_config._loss_fn_id)
        loss_fn = nll_loss

        self._build_optimizer(
            self.model_config.optimizer, model, optimizer_args=self.model_config.optimizer_args
        )
        loss, num_of_training_data = self._plan_fit(model=model, dataset_key=dataset_key, loss_fn=loss_fn, device=device)

        ## multiply the weights to the model
        with th.no_grad():
            for parameter in model.parameters():
                parameter.set_(parameter.data * num_of_training_data)

        return [loss, num_of_training_data]

    ## added by bobsonlin
    def model_share(self, encrypters):
        self._check_model_config()
        model = self.get_obj(self.model_config._model_id)

        ## generate shares
        enc_params = []
        params = list(model.parameters())
        for param_index in range(len(params)):
            fix_para = params[param_index].fix_precision(precision_fractional=5)
            enc_para = fix_para.share(*encrypters)
            enc_params.append(enc_para)

        return enc_params

    def fit_sagg_mc(self, dataset_key: str, encrypters, device: str = "cpu", **kwargs):

        self._check_model_config()
        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset {dataset_key} unknown.")

        model = self.get_obj(self.model_config._model_id)

        # loss_fn = self.get_obj(self.model_config._loss_fn_id)
        model_type = model.id.split("_")[1]
        if model_type == "MNIST":
            loss_fn = nll_loss
        elif model_type == "RESNET":
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nll_loss

        self._build_optimizer(
            self.model_config.optimizer, model, optimizer_args=self.model_config.optimizer_args
        )

        training_start_time = time.time()
        pdb.set_trace()
        loss, num_of_training_data = self._plan_fit(model=model, dataset_key=dataset_key, loss_fn=loss_fn, device=device)

        training_end_time = time.time()
        print("[trace] LocalTraining duration", self.id, training_end_time - training_start_time)
        ## multiply the weights to the model
#         with th.no_grad():
#             for parameter in model.parameters():
#                 parameter.set_(parameter.data * num_of_training_data)

        ## encrypt model and multiply with weight
        enc_params = []
        params = list(model.parameters())

        encrypt_multiply_start_time = time.time()

        for param_index in range(len(params)):
            fix_para = params[param_index].fix_precision(precision_fractional=5)
            encrypt_start = time.time()
            enc_para = fix_para.share(*encrypters)
            encrypt_end = time.time()
            print("[trace]", "EncryptParameter"+str(param_index), "duration", self.id, encrypt_end - encrypt_start)

            multiply_start = time.time()
            enc_para = enc_para * int(num_of_training_data)
            multiply_end = time.time()
            print("[trace]", "MultiplyParameter"+str(param_index), "duration", self.id, multiply_end - multiply_start)
            enc_params.append(enc_para)

        encrypt_multiply_end_time = time.time()
        print("[trace] EncryptMultiply duration", self.id, encrypt_multiply_end_time - encrypt_multiply_start_time)

        result_list = [loss, num_of_training_data]
        result_list.extend(enc_params)

        super().de_register_obj(model, _recurse_torch_objs=True)

        return result_list

    def _create_data_loader(self, dataset_key: str, shuffle: bool = False):
        data_range = range(len(self.datasets[dataset_key]))
        if shuffle:
            sampler = RandomSampler(data_range)
        else:
            sampler = SequentialSampler(data_range)
        data_loader = th.utils.data.DataLoader(
            self.datasets[dataset_key],
            batch_size=self.train_config.batch_size,
            sampler=sampler,
            num_workers=0,
        )
        return data_loader

    def _create_data_loader_mc(self, dataset_key: str, shuffle: bool = False):
        data_range = range(len(self.datasets[dataset_key]))
        if shuffle:
            sampler = RandomSampler(data_range)
        else:
            sampler = SequentialSampler(data_range)
        data_loader = th.utils.data.DataLoader(
            self.datasets[dataset_key],
            batch_size=self.model_config.batch_size,
            sampler=sampler,
            num_workers=0,
        )
        return data_loader

    def _fit(self, model, dataset_key, loss_fn, device="cpu"):
        model.train()
        data_loader = self._create_data_loader(
            dataset_key=dataset_key, shuffle=self.train_config.shuffle
        )

        loss = None
        iteration_count = 0

        for _ in range(self.train_config.epochs):
            for (data, target) in data_loader:
                # Set gradients to zero
                self.optimizer.zero_grad()

                # Update model
                output = model(data.to(device))
                loss = loss_fn(target=target.to(device), pred=output)
                loss.backward()
                self.optimizer.step()

                # Update and check interation count
                iteration_count += 1
                if iteration_count >= self.train_config.max_nr_batches >= 0:
                    break

        return loss

    def _plan_fit(self, model, dataset_key, loss_fn, device="cpu"):
        num_of_training_data = len(self.datasets[dataset_key])
        num_of_training_data = th.tensor(num_of_training_data)
        data_loader = self._create_data_loader_mc(
            dataset_key=dataset_key, shuffle=self.model_config.shuffle
        )

        loss = None
        iteration_count = 0

        for _ in range(self.model_config.epochs):
            # iter_start = time.time()
            for batch_idx, (data, target) in enumerate(data_loader):
                # batch_start = time.time()
                # print("[DEBUG]", "Iter time:", batch_start - iter_start)

                # Set gradients to zero
                self.optimizer.zero_grad()

                # Update model
                output = model(data.to(device))
                loss = loss_fn(output, target.to(device))
                loss.backward()
                self.optimizer.step()

                # batch_end = time.time()
                # print("[DEBUG]", "Batch", batch_idx, "   ", "Time:", batch_end - batch_start)

                # Update and check interation count
                iteration_count += 1
                if iteration_count >= self.model_config.max_nr_batches >= 0:
                    break
                # iter_start = time.time()

        return loss, num_of_training_data

    def evaluate(
        self,
        dataset_key: str,
        return_histograms: bool = False,
        nr_bins: int = -1,
        return_loss: bool = True,
        return_raw_accuracy: bool = True,
        device: str = "cpu",
    ):
        """Evaluates a model on the local dataset as specified in the local TrainConfig object.

        Args:
            dataset_key: Identifier of the local dataset that shall be used for training.
            return_histograms: If True, calculate the histograms of predicted classes.
            nr_bins: Used together with calculate_histograms. Provide the number of classes/bins.
            return_loss: If True, loss is calculated additionally.
            return_raw_accuracy: If True, return nr_correct_predictions and nr_predictions
            device: "cuda" or "cpu"

        Returns:
            Dictionary containing depending on the provided flags:
                * loss: avg loss on data set, None if not calculated.
                * nr_correct_predictions: number of correct predictions.
                * nr_predictions: total number of predictions.
                * histogram_predictions: histogram of predictions.
                * histogram_target: histogram of target values in the dataset.
        """
        self._check_train_config()

        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset {dataset_key} unknown.")

        eval_result = dict()
        model = self.get_obj(self.train_config._model_id).obj
        loss_fn = self.get_obj(self.train_config._loss_fn_id).obj
        device = "cuda" if device == "cuda" else "cpu"
        data_loader = self._create_data_loader(dataset_key=dataset_key, shuffle=False)
        test_loss = 0.0
        correct = 0
        if return_histograms:
            hist_target = np.zeros(nr_bins)
            hist_pred = np.zeros(nr_bins)

        with th.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                if return_loss:
                    test_loss += loss_fn(output, target).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                if return_histograms:
                    hist, _ = np.histogram(target, bins=nr_bins, range=(0, nr_bins))
                    hist_target += hist
                    hist, _ = np.histogram(pred, bins=nr_bins, range=(0, nr_bins))
                    hist_pred += hist
                if return_raw_accuracy:
                    correct += pred.eq(target.view_as(pred)).sum().item()

        if return_loss:
            test_loss /= len(data_loader.dataset)
            eval_result["loss"] = test_loss
        if return_raw_accuracy:
            eval_result["nr_correct_predictions"] = correct
            eval_result["nr_predictions"] = len(data_loader.dataset)
        if return_histograms:
            eval_result["histogram_predictions"] = hist_pred
            eval_result["histogram_target"] = hist_target

        return eval_result

    def evaluate_mc(
        self,
        dataset_key: str,
        return_histograms: bool = False,
        nr_bins: int = -1,
        return_loss: bool = True,
        return_raw_accuracy: bool = True,
        device: str = "cpu",
    ):
        """Evaluates a model on the local dataset as specified in the local TrainConfig object.

        Args:
            dataset_key: Identifier of the local dataset that shall be used for training.
            return_histograms: If True, calculate the histograms of predicted classes.
            nr_bins: Used together with calculate_histograms. Provide the number of classes/bins.
            return_loss: If True, loss is calculated additionally.
            return_raw_accuracy: If True, return nr_correct_predictions and nr_predictions
            device: "cuda" or "cpu"

        Returns:
            Dictionary containing depending on the provided flags:
                * loss: avg loss on data set, None if not calculated.
                * nr_correct_predictions: number of correct predictions.
                * nr_predictions: total number of predictions.
                * histogram_predictions: histogram of predictions.
                * histogram_target: histogram of target values in the dataset.
        """
        self._check_model_config()

        if dataset_key not in self.datasets:
            raise ValueError(f"Dataset {dataset_key} unknown.")

        eval_result = dict()
        model = self.get_obj(self.model_config._model_id)
        loss_fn = nll_loss
        device = "cuda" if device == "cuda" else "cpu"
        data_loader = self._create_data_loader_mc(dataset_key=dataset_key, shuffle=False)
        test_loss = 0.0
        correct = 0
        if return_histograms:
            hist_target = np.zeros(nr_bins)
            hist_pred = np.zeros(nr_bins)

        with th.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                if return_loss:
                    test_loss += loss_fn(output, target).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                if return_histograms:
                    hist, _ = np.histogram(target, bins=nr_bins, range=(0, nr_bins))
                    hist_target += hist
                    hist, _ = np.histogram(pred, bins=nr_bins, range=(0, nr_bins))
                    hist_pred += hist
                if return_raw_accuracy:
                    correct += pred.eq(target.view_as(pred)).sum().item()
        if return_loss:
            test_loss /= len(data_loader.dataset)
            eval_result["loss"] = test_loss
        if return_raw_accuracy:
            eval_result["nr_correct_predictions"] = correct
            eval_result["nr_predictions"] = len(data_loader.dataset)
        if return_histograms:
            eval_result["histogram_predictions"] = hist_pred
            eval_result["histogram_target"] = hist_target

        super().de_register_obj(model, _recurse_torch_objs=True)
        return eval_result

    def _log_msgs(self, value):
        self.log_msgs = value

    ## added by bobsonlin
    def list_objects(self, *args):
        return str(self._objects)

    def list_tensors(self, *args):
        return str(self._tensors)
