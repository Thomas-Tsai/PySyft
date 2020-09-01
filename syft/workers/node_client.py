import json

from typing import Union
from urllib.parse import urlparse
from typing import List

import websockets
import syft as sy
from syft.messaging.message import ObjectRequestMessage
from syft.messaging.message import ObjectMessage
# Syft imports
from syft.serde import serialize
from syft.version import __version__
from syft.execution.plan import Plan
from syft.codes import REQUEST_MSG, RESPONSE_MSG
from syft.federated.federated_client import FederatedClient
from syft.workers.websocket_client import WebsocketClientWorker
from syft.grid.authentication.credential import AbstractCredential
import pdb
import time

TIMEOUT_INTERVAL = 60

class NodeClient(WebsocketClientWorker, FederatedClient):
    """Federated Node Client."""

    def __init__(
        self,
        hook,
        address,
        credential: AbstractCredential = None,
        id: Union[int, str] = 0,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        encoding: str = "ISO-8859-1",
    ):
        """
        Args:
            hook : a normal TorchHook object.
            address : Address used to connect with remote node.
            credential : Credential used to perform authentication.
            id : the unique id of the worker (string or int)
            is_client_worker : An optional boolean parameter to indicate
                whether this worker is associated with an end user client. If
                so, it assumes that the client will maintain control over when
                variables are instantiated or deleted as opposed to handling
                tensor/variable/model lifecycle internally. Set to True if this
                object is not where the objects will be stored, but is instead
                a pointer to a worker that eists elsewhere.
                log_msgs : whether or not all messages should be
                saved locally for later inspection.
            verbose : a verbose option - will print all messages
                sent/received to stdout.
            encoding : Encoding pattern used to send/retrieve models.
        """
        self.address = address
        self.encoding = encoding
        self.credential = credential

        # Parse address string to get scheme, host and port
        self.secure, self.host, self.port = self._parse_address(address)

        # Initialize WebsocketClientWorker / Federated Client
        super().__init__(
            hook,
            self.host,
            self.port,
            self.secure,
            id,
            is_client_worker,
            log_msgs,
            verbose,
            None,  # initial data
        )

        # Update Node reference using node's Id given by the remote node
        self._update_node_reference(self._get_node_infos())

        if self.credential:
            self._authenticate()

    @property
    def url(self) -> str:
        """ Get Node URL Address.
            Returns:
                address (str) : Node's address.
        """
        if self.port:
            return (
                f"wss://{self.host}:{self.port}" if self.secure else f"ws://{self.host}:{self.port}"
            )
        else:
            return self.address

    @property
    def models(self) -> list:
        """ Get models stored at remote node.

            Returns:
                models (List) : List of models stored in this node.
        """
        message = {REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.LIST_MODELS}
        response = self._forward_json_to_websocket_server_worker(message)
        return self._return_bool_result(response, RESPONSE_MSG.MODELS)

    def _authenticate(self):
        """ Perform Authentication Process using credentials grid credentials.
            Raises:
                RuntimeError : If authentication process fail.
        """
        if not isinstance(self.credential, AbstractCredential):
            raise RuntimeError("Your credential needs to be an instance of grid credentials.")

        cred_dict = self.credential.json()

        # Prepare a authentication request to remote grid node
        cred_dict[REQUEST_MSG.TYPE_FIELD] = REQUEST_MSG.AUTHENTICATE
        response = self._forward_json_to_websocket_server_worker(cred_dict)

        # If succeeded, update node's reference and update client's credential.
        node_id = self._return_bool_result(response, RESPONSE_MSG.NODE_ID)

        if node_id:
            self._update_node_reference(node_id)
        else:
            raise RuntimeError("Invalid user.")

    def _update_node_reference(self, new_id: str):
        """ Update worker references changing node id references at hook structure.
            Args:
                new_id (str) : New worker ID.
        """
        del self.hook.local_worker._known_workers[self.id]
        self.id = new_id
        self.hook.local_worker._known_workers[new_id] = self

    def _parse_address(self, address: str) -> tuple:
        """ Parse Address string to define secure flag and split into host and port.
            Args:
                address (str) : Adress of remote worker.
        """
        url = urlparse(address)
        secure = True if url.scheme == "wss" else False
        return (secure, url.hostname, url.port)

    def _get_node_infos(self) -> str:
        """ Get Node ID from remote node worker
            Returns:
                node_id (str) : node id used by remote worker.
        """
        message = {REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.GET_ID}
        response = self._forward_json_to_websocket_server_worker(message)
        node_version = response.get(RESPONSE_MSG.SYFT_VERSION, None)
        if node_version != __version__:
            raise RuntimeError(
                "Library version mismatch, The PySyft version of your environment is "
                + __version__
                + " the Grid Node Syft version is "
                + node_version
            )

        return response.get(RESPONSE_MSG.NODE_ID, None)

    def _forward_json_to_websocket_server_worker(self, message: dict) -> dict:
        """ Prepare/send a JSON message to a remote node and receive the response.
            Args:
                message (dict) : message payload.
            Returns:
                node_response (dict) : response payload.
        """
        self.ws.send(json.dumps(message))
        return json.loads(self.ws.recv())

    def _forward_to_websocket_server_worker(self, message: bin) -> bin:
        """ Send a bin message to a remote node and receive the response.
            Args:
                message (bytes) : message payload.
            Returns:
                node_response (bytes) : response payload.
        """
        # print("[PROF]", "MessageSize", len(message), "bytes")
        # print("[PROF]", "NC_SEND_BIN", "start", "sender", time.time())
        self.ws.send_binary(message)
        # print("[PROF]", "NC_SEND_BIN", "end", "sender", time.time())
        response = self.ws.recv()
        # print("[PROF]", "NC_WS_RECV", "end", "sender", time.time())
        return response

    def _return_bool_result(self, result, return_key=None):
        if result.get(RESPONSE_MSG.SUCCESS):
            return result[return_key] if return_key is not None else True
        elif result.get(RESPONSE_MSG.ERROR):
            raise RuntimeError(result[RESPONSE_MSG.ERROR])
        else:
            raise RuntimeError("Something went wrong.")

    def connect_nodes(self, node) -> dict:
        """ Connect two remote workers between each other.
            Args:
                node (WebsocketFederatedClient) : Node that will be connected with this remote worker.
            Returns:
                node_response (dict) : node response.
        """
        message = {
            REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.CONNECT_NODE,
            "address": node.address,
            "id": node.id,
        }
        return self._forward_json_to_websocket_server_worker(message)

    def serve_model(
        self,
        model,
        model_id: str = None,
        mpc: bool = False,
        allow_download: bool = False,
        allow_remote_inference: bool = False,
    ):
        """ Hosts the model and optionally serve it using a Socket / Rest API.
            Args:
                model : A jit model or Syft Plan.
                model_id (str): An integer/string representing the model id.
                If it isn't provided and the model is a Plan we use model.id,
                if the model is a jit model we raise an exception.
                allow_download (bool) : Allow to copy the model to run it locally.
                allow_remote_inference (bool) : Allow to run remote inferences.
            Returns:
                result (bool) : True if model was served sucessfully.
            Raises:
                ValueError: model_id isn't provided and model is a jit model.
                RunTimeError: if there was a problem during model serving.
        """

        # If the model is a Plan we send the model
        # and host the plan version created after
        # the send action
        if isinstance(model, Plan):
            # We need to use the same id in the model
            # as in the POST request.
            pointer_model = model.send(self)
            res_model = pointer_model
        else:
            res_model = model

        serialized_model = serialize(res_model)

        message = {
            REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.HOST_MODEL,
            "encoding": self.encoding,
            "model_id": model_id,
            "allow_download": str(allow_download),
            "mpc": str(mpc),
            "allow_remote_inference": str(allow_remote_inference),
            "model": serialized_model.decode(self.encoding),
        }
        response = self._forward_json_to_websocket_server_worker(message)
        return self._return_bool_result(response)

    def run_remote_inference(self, model_id, data):
        """ Run a dataset inference using a remote model.

            Args:
                model_id (str) : Model ID.
                data (Tensor) : dataset to be inferred.
            Returns:
                inference (Tensor) : Inference result
            Raises:
                RuntimeError : If an unexpected behavior happen.
        """
        serialized_data = serialize(data).decode(self.encoding)
        message = {
            REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.RUN_INFERENCE,
            "model_id": model_id,
            "data": serialized_data,
            "encoding": self.encoding,
        }
        response = self._forward_json_to_websocket_server_worker(message)
        return self._return_bool_result(response, RESPONSE_MSG.INFERENCE_RESULT)

    def delete_model(self, model_id: str) -> bool:
        """ Delete a model previously registered.

            Args:
                model_id (String) : ID of the model that will be deleted.
            Returns:
                result (bool) : If succeeded, return True.
        """
        message = {REQUEST_MSG.TYPE_FIELD: REQUEST_MSG.DELETE_MODEL, "model_id": model_id}
        response = self._forward_json_to_websocket_server_worker(message)
        return self._return_bool_result(response)

    async def async_fit(self, dataset_key: str, device: str = "cpu", return_ids: List[int] = None):
        """Asynchronous call to fit function on the remote location.
        Args:
            dataset_key: Identifier of the dataset which shall be used for the training.
            return_ids: List of return ids.
        Returns:
            See return value of the FederatedClient.fit() method.
        """
        if return_ids is None:
            return_ids = [sy.ID_PROVIDER.pop()]

        # Close the existing websocket connection in order to open a asynchronous connection
        # This code is not tested with secure connections (wss protocol).
        self.close()
        async with websockets.connect(
            self.url, timeout=TIMEOUT_INTERVAL, max_size=None, ping_timeout=TIMEOUT_INTERVAL
        ) as websocket:
            message = self.create_worker_command_message(
                command_name="fit", return_ids=return_ids, dataset_key=dataset_key, device=device
            )

            # Send the message and return the deserialized response.
            serialized_message = sy.serde.serialize(message)

            await websocket.send(serialized_message)
            await websocket.recv()  # returned value will be None, so don't care

        # Reopen the standard connection
        self.connect()

        # Send an object request message to retrieve the result tensor of the fit() method
        msg = ObjectRequestMessage(return_ids[0], None, "")
        serialized_message = sy.serde.serialize(msg)
        response = self._send_msg(serialized_message)

        # Return the deserialized response.
        return sy.serde.deserialize(response)

    async def async_fit_mc(self, dataset_key: str, device: str = "cpu", return_ids: List[int] = None):
        """Asynchronous call to fit function on the remote location.
        Args:
            dataset_key: Identifier of the dataset which shall be used for the training.
            return_ids: List of return ids.
        Returns:
            See return value of the FederatedClient.fit() method.
        """
        if return_ids is None:
            return_ids = [sy.ID_PROVIDER.pop()]

        # Close the existing websocket connection in order to open a asynchronous connection
        # This code is not tested with secure connections (wss protocol).
        self.close()
        async with websockets.connect(
            self.url, timeout=TIMEOUT_INTERVAL, max_size=None, ping_timeout=TIMEOUT_INTERVAL
        ) as websocket:
            message = self.create_worker_command_message(
                command_name="fit_mc", return_ids=return_ids, dataset_key=dataset_key, device=device
            )

            # Send the message and return the deserialized response.
            serialized_message = sy.serde.serialize(message)

            await websocket.send(serialized_message)
            await websocket.recv()  # returned value will be None, so don't care

        # Reopen the standard connection
        self.connect()

        # Send an object request message to retrieve the result tensor of the fit() method
        msg = ObjectRequestMessage(return_ids[0], None, "")
        serialized_message = sy.serde.serialize(msg)
        response = self._send_msg(serialized_message)
        loss = sy.serde.deserialize(response)

        msg = ObjectRequestMessage(return_ids[1], None, "")
        serialized_message = sy.serde.serialize(msg)
        response = self._send_msg(serialized_message)
        num_of_training_data = sy.serde.deserialize(response)

        # Return the deserialized response.
        return loss, num_of_training_data

    ## added by bobsonlin
    async def async_fit2_mc(self, model_config, dataset_key: str, device: str = "cpu", return_ids: List[int] = None):
        """Asynchronous call to fit function on the remote location.
        Args:
            dataset_key: Identifier of the dataset which shall be used for the training.
            return_ids: List of return ids.
        Returns:
            See return value of the FederatedClient.fit() method.
        """
        if return_ids is None:
            return_ids = [sy.ID_PROVIDER.pop()]

        # Close the existing websocket connection in order to open a asynchronous connection
        # This code is not tested with secure connections (wss protocol).
        self.close()
        async with websockets.connect(
            self.url, timeout=TIMEOUT_INTERVAL, max_size=None, ping_timeout=TIMEOUT_INTERVAL
        ) as websocket:
            message = self.create_worker_command_message(
                command_name="fit_mc", return_ids=return_ids, dataset_key=dataset_key, device=device
            )

            # Send the message and return the deserialized response.
            serialized_message = sy.serde.serialize(message)

            await websocket.send(serialized_message)
            await websocket.recv()  # returned value will be None, so don't care

        # Reopen the standard connection
        self.connect()

        # Send an object request message to retrieve the result tensor of the fit() method
        msg = ObjectRequestMessage(return_ids[0], None, "")
        serialized_message = sy.serde.serialize(msg)
        response = self._send_msg(serialized_message)
        loss = sy.serde.deserialize(response)

        msg = ObjectRequestMessage(return_ids[1], None, "")
        serialized_message = sy.serde.serialize(msg)
        response = self._send_msg(serialized_message)
        num_of_training_data = sy.serde.deserialize(response)

        # Return the deserialized response.
        return loss, num_of_training_data



    ## added by bobsonlin
    async def async_model_share(self, encrypters, return_ids: List[int] = None):

        if return_ids is None:
            return_ids = [sy.ID_PROVIDER.pop()]

        self.close()
        async with websockets.connect(
            self.url, timeout=TIMEOUT_INTERVAL, max_size=None, ping_timeout=TIMEOUT_INTERVAL
        ) as websocket:
            message = self.create_worker_command_message(
                command_name="model_share", encrypters=encrypters, return_ids=return_ids)
            serialized_message = sy.serde.serialize(message)
            await websocket.send(serialized_message)
            await websocket.recv()    ## make sure that the command is executed
            # bin_response = await websocket.recv()
            # response = sy.serde.deserialize(bin_response)
            # print("response:", response)

        # Reopen the standard connection
        self.connect()

        ## Retrieve the results !
#         msg = ObjectRequestMessage(return_ids[0], None, "")
#         serialized_message = sy.serde.serialize(msg)
#         bin_response = self._send_msg(serialized_message)
#         response = sy.serde.deserialize(bin_response)

        enc_params = []
        for i in range(len(return_ids)):
            msg = ObjectRequestMessage(return_ids[i], None, "")
            serialized_message = sy.serde.serialize(msg)
            bin_response = self._send_msg(serialized_message)
            response = sy.serde.deserialize(bin_response)
            enc_params.append(response)
        return enc_params

    async def async_fit_sagg_mc(self, dataset_key: str, encrypters, device: str = "cpu", return_ids: List[int] = None):
        """Asynchronous call to fit_sagg_mc function on the remote location.
        Args:
            dataset_key: Identifier of the dataset which shall be used for the training.
            return_ids: List of return ids.
        Returns:
            See return value of the FederatedClient.fit() method.
        """
        if return_ids is None:
            return_ids = [sy.ID_PROVIDER.pop()]

        # Close the existing websocket connection in order to open a asynchronous connection
        # This code is not tested with secure connections (wss protocol).
        self.close()
        async with websockets.connect(
            self.url, timeout=TIMEOUT_INTERVAL, max_size=None, ping_timeout=TIMEOUT_INTERVAL
        ) as websocket:
            message = self.create_worker_command_message(
                command_name="fit_sagg_mc", return_ids=return_ids, dataset_key=dataset_key, encrypters=encrypters, device=device
            )

            # Send the message and return the deserialized response.
            serialized_message = sy.serde.serialize(message)

            await websocket.send(serialized_message)
            await websocket.recv()  # returned value will be None, so don't care

        # Reopen the standard connection
        self.connect()

        # Send an object request message to retrieve the result tensor of the fit() method
        result_list = []
        retrieve_start_time = time.time()

        for i in range(len(return_ids)):
            msg = ObjectRequestMessage(return_ids[i], None, "")
            serialized_message = sy.serde.serialize(msg)
            bin_response = self._send_msg(serialized_message)
            response = sy.serde.deserialize(bin_response)
            result_list.append(response)

        retrieve_end_time = time.time()
        print("[trace]", "RetrieveTime", "duration", self.id, retrieve_end_time - retrieve_start_time)

        return result_list

    def evaluate_mc(
        self,
        dataset_key: str,
        return_histograms: bool = False,
        nr_bins: int = -1,
        return_loss=True,
        return_raw_accuracy: bool = True,
        device: str = "cpu",
    ):
        """Call the evaluate() method on the remote worker (WebsocketServerWorker instance).

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

        return self._send_msg_and_deserialize(
            "evaluate_mc",
            dataset_key=dataset_key,
            return_histograms=return_histograms,
            nr_bins=nr_bins,
            return_loss=return_loss,
            return_raw_accuracy=return_raw_accuracy,
            device=device,
        )

    def __str__(self) -> str:
        return "Federated Worker < id: " + self.id + " >"
