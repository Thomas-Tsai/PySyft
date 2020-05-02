from typing import Union
import weakref

import torch

import syft as sy
from syft.generic.pointers.object_wrapper import ObjectWrapper
from syft.workers.abstract import AbstractWorker
from syft.workers.base import BaseWorker

class ModelConfig:
    
    def __init__(
        self,
        model: sy.Plan,
        loss_fn: sy.Plan,
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
        """Returns the string representation of a TrainConfig."""
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
        self.model_ptr = self.model.send(location)

        # Send loss function
        self.loss_fn_ptr = self.loss_fn.send(location)

        # Send train configuration itself
#         obj_with_id = ObjectWrapper(id=sy.ID_PROVIDER.pop(), obj=self)
#         ptr = self.owner.send(obj_with_id, location)

        return ptr