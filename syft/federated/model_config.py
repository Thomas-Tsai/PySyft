from typing import Union
import weakref

import torch
import torch.nn as nn
import pdb
import time

import syft as sy
from syft.execution.plan import Plan
from syft.generic.pointers.object_wrapper import ObjectWrapper
from syft.workers.abstract import AbstractWorker
from syft.workers.base import BaseWorker


class ModelConfig:
    """TrainConfig abstraction.
    A wrapper object that contains all that is needed to run a training loop
    remotely on a federated learning setup.
    """

    def __init__(
        self,
        model: Plan,
        loss_fn: Plan,
        owner: AbstractWorker = None,
        batch_size: int = 32,
        epochs: int = 1,
        optimizer: str = "SGD",
        optimizer_args: dict = {"lr": 0.1},
        id: Union[int, str] = None,
        max_nr_batches: int = -1,
        shuffle: bool = True,
        loss_fn_id: int = None,
        model_id: int = None,
    ):
        """Initializer for ModelConfig.
        Args:
            model: A traced torch nn.Module instance.
            loss_fn: A jit function representing a loss function which
                shall be used to calculate the loss.
            batch_size: Batch size used for training.
            epochs: Epochs used for training.
            optimizer: A string indicating which optimizer should be used.
            optimizer_args: A dict containing the arguments to initialize the optimizer. Defaults to {'lr': 0.1}.
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the tensor.
            max_nr_batches: Maximum number of training steps that will be performed. For large datasets
                            this can be used to run for less than the number of epochs provided.
            shuffle: boolean, whether to access the dataset randomly (shuffle) or sequentially (no shuffle).
            loss_fn_id: The id_at_location of (the ObjectWrapper of) a loss function which
                        shall be used to calculate the loss. This is used internally for train config deserialization.
            model_id: id_at_location of a traced torch nn.Module instance (objectwrapper). . This is used internally for train config deserialization.
        """
        # syft related attributes
        self.owner = owner if owner else sy.hook.local_worker
        self.id = id if id is not None else sy.ID_PROVIDER.pop()
        self.location = None

        # training related attributes
        self.model = model
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.max_nr_batches = max_nr_batches
        self.shuffle = shuffle

        # pointers
        self.model_ptr = None
        self.loss_fn_ptr = None

        # internal ids
        self._model_id = model_id
        self._loss_fn_id = loss_fn_id

    def __str__(self) -> str:
        """Returns the string representation of a ModelConfig."""
        out = "<"
        out += str(type(self)).split("'")[1].split(".")[-1]
        out += " id:" + str(self.id)
        out += " owner:" + str(self.owner.id)

        if self.location:
            out += " location:" + str(self.location.id)

        out += " epochs: " + str(self.epochs)
        out += " batch_size: " + str(self.batch_size)
        out += " optimizer_args: " + str(self.optimizer_args)

        out += ">"
        return out

    def _wrap_and_send_obj(self, obj, location):
        """Wrappers object and send it to location."""
        obj_with_id = ObjectWrapper(id=sy.ID_PROVIDER.pop(), obj=obj)
        obj_ptr = self.owner.send(obj_with_id, location)
        obj_id = obj_ptr.id_at_location
        return obj_ptr, obj_id

    def send(self, location: BaseWorker) -> weakref:
        """Gets the pointer to a new remote object.
        One of the most commonly used methods in PySyft, this method serializes
        the object upon which it is called (self), sends the object to a remote
        worker, creates a pointer to that worker, and then returns that pointer
        from this function.
        Args:
            location: The BaseWorker object which you want to send this object
                to. Note that this is never actually the BaseWorker but instead
                a class which instantiates the BaseWorker abstraction.
        Returns:
            A weakref instance.
        """
        # Send traced model
        print("[trace] GlobalModelSend", "start", location.id, time.time())
        self.model_ptr = self.model.send(location)
        print("[trace] GlobalModelSend", "end", location.id, time.time())
        self._model_id = self.model_ptr.id_at_location
#         self.model_ptr, self._model_id = self._wrap_and_send_obj(self.model, location)

        # Send loss function
        print("[trace] LossFuncSend", "start", location.id, time.time())
        self.loss_fn_ptr = self.loss_fn.send(location)
        print("[trace] LossFuncSend", "end", location.id, time.time())
        self._loss_fn_id = self.loss_fn_ptr.id_at_location
#         self.loss_fn_ptr, self._loss_fn_id = self._wrap_and_send_obj(self.loss_fn, location)

        # Send train configuration itself
        print("[trace] ModelConfigSend", "start", location.id, time.time())
        ptr = self.owner.send(self, location)
        print("[trace] ModelConfigSend", "end", location.id, time.time())

        return ptr

    def get(self, location):
        return self.owner.request_obj(self, location)

    def get_model(self):
        if self.model is not None:
            return self.model_ptr.get()

    def get_loss_fn(self):
        if self.loss_fn is not None:
            return self.loss_fn.get()

    @staticmethod
    def simplify(worker: AbstractWorker, model_config: "ModelConfig") -> tuple:
        """Takes the attributes of a TrainConfig and saves them in a tuple.
        Attention: this function does not serialize the model and loss_fn attributes
        of a TrainConfig instance, these are serialized and sent before. TrainConfig
        keeps a reference to the sent objects using _model_id and _loss_fn_id which
        are serialized here.
        Args:
            worker: the worker doing the serialization
            train_config: a TrainConfig object
        Returns:
            tuple: a tuple holding the unique attributes of the TrainConfig object
        """
        return (
            model_config._model_id,
            model_config._loss_fn_id,
            model_config.batch_size,
            model_config.epochs,
            sy.serde.msgpack.serde._simplify(worker, model_config.optimizer),
            sy.serde.msgpack.serde._simplify(worker, model_config.optimizer_args),
            sy.serde.msgpack.serde._simplify(worker, model_config.id),
            model_config.max_nr_batches,
            model_config.shuffle,
        )

    @staticmethod
    def detail(worker: AbstractWorker, train_config_tuple: tuple) -> "TrainConfig":
        """This function reconstructs a TrainConfig object given it's attributes in the form of a tuple.
        Args:
            worker: the worker doing the deserialization
            train_config_tuple: a tuple holding the attributes of the TrainConfig
        Returns:
            train_config: A TrainConfig object
        """

        (
            model_id,
            loss_fn_id,
            batch_size,
            epochs,
            optimizer,
            optimizer_args,
            id,
            max_nr_batches,
            shuffle,
        ) = train_config_tuple

        id = sy.serde.msgpack.serde._detail(worker, id)
        detailed_optimizer = sy.serde.msgpack.serde._detail(worker, optimizer)
        detailed_optimizer_args = sy.serde.msgpack.serde._detail(worker, optimizer_args)

        model_config = ModelConfig(
            model=None,
            loss_fn=None,
            owner=worker,
            id=id,
            model_id=model_id,
            loss_fn_id=loss_fn_id,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=detailed_optimizer,
            optimizer_args=detailed_optimizer_args,
            max_nr_batches=max_nr_batches,
            shuffle=shuffle,
        )

        return model_config
