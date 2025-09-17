import time
import pickle
import typing as tp

from dataclasses import dataclass
from functools import wraps

import zmq

from speechflow.logging import log_to_file, trace
from speechflow.utils.serialize import Serialize

__all__ = [
    "ZMQPatterns",
    "ZMQServer",
    "ZMQClient",
    "ZMQWorker",
    "ZMQProxy",
]


def retry(
    func: tp.Callable,
) -> tp.Callable:
    num_retry: int = 5

    @wraps(func)
    def retry_loop(*args, **kwargs) -> bool:

        for i in range(num_retry):
            try:
                func(*args, **kwargs)
                return True
            except zmq.error.Again as e:
                if i + 1 == num_retry:
                    log_to_file(trace("retry_loop", e))
                    return False
                else:
                    time.sleep(2)

        return func(*args, **kwargs)

    return retry_loop


@dataclass
class ZMQServer:
    context: zmq.Context
    frontend: zmq.Socket
    backend: zmq.Socket
    poller: zmq.Poller
    socks: tp.Dict[zmq.Socket, tp.Any] = None  # type: ignore
    flags: int = zmq.NOBLOCK

    def pool(self, timeout: tp.Optional[int] = None):  # in milliseconds
        self.socks = dict(self.poller.poll(timeout))

    def is_frontend_ready(self) -> bool:
        if self.frontend:
            return self.socks.get(self.frontend) == zmq.POLLIN
        else:
            return False

    def is_backend_ready(self) -> bool:
        if self.backend:
            return self.socks.get(self.backend) == zmq.POLLIN
        else:
            return False

    def close(self):
        if self.frontend:
            self.frontend.close()
        if self.backend:
            self.backend.close()

    @retry
    def frontend_send_multipart(self, data: tp.List[tp.Any]):
        if not isinstance(data, list):
            data = [data]
        self.frontend.send_multipart(data, flags=self.flags)

    def frontend_recv_multipart(self) -> tp.List[tp.Any]:
        return self.frontend.recv_multipart()

    @retry
    def backend_send_multipart(self, data: tp.List[tp.Any]):
        if not isinstance(data, list):
            data = [data]
        self.backend.send_multipart(data, flags=self.flags)

    def backend_recv_multipart(self) -> tp.List[tp.Any]:
        return self.backend.recv_multipart()


@dataclass
class ZMQClient:
    context: zmq.Context
    socket: zmq.Socket
    flags: int = 0

    def close(self):
        self.socket.close()

    @retry
    def send(self, data: tp.Any, serialize: bool = True):
        if serialize:
            data = Serialize.dump(data)
        self.socket.send(data, flags=self.flags)

    def recv(
        self,
        deserialize: bool = True,
        timeout: tp.Optional[int] = None,  # in milliseconds
    ) -> tp.Optional[tp.Any]:
        msg = None

        if timeout and self.socket.poll(timeout=timeout) == 0:
            return msg

        try:
            msg = self.socket.recv(flags=self.flags)
        except zmq.ZMQError as e:
            if not self.flags:
                raise e

        if msg is not None and deserialize:
            msg = Serialize.load(msg)

        return msg

    @retry
    def send_multipart(self, data: tp.List[tp.Any], serialize: bool = True):
        if serialize:
            data = Serialize.dumps(data)
        self.socket.send_multipart(data, flags=self.flags)

    def recv_multipart(
        self,
        deserialize: bool = True,
        timeout: tp.Optional[int] = None,  # in milliseconds
        max_num_message: tp.Optional[int] = None,
    ) -> tp.List[tp.Any]:
        if timeout and self.socket.poll(timeout=timeout) == 0:
            return []
        else:
            list_bytes = []
            while True or (max_num_message and len(list_bytes) < max(max_num_message, 1)):
                try:
                    flags = self.flags if timeout else 0
                    msg = self.socket.recv_multipart(flags=flags)
                    if msg is not None:
                        list_bytes += msg
                    else:
                        break
                    if not timeout:
                        break
                except zmq.ZMQError:
                    break

            list_bytes = [item for item in list_bytes if item != b""]

            if deserialize:
                return [pickle.loads(item) for item in list_bytes]
            else:
                return list_bytes

    def request(
        self,
        data: tp.Any,
        serialize: bool = True,
        deserialize: bool = True,
        timeout: tp.Optional[int] = None,  # in milliseconds
        multipart: bool = False,
    ) -> tp.Optional[tp.Any]:
        if multipart:
            self.send_multipart(data, serialize)
        else:
            self.send(data, serialize)

        if timeout is None or timeout == -1:
            msg = None
            while msg is None:
                if multipart:
                    msg = self.recv_multipart(deserialize, 1000)
                else:
                    msg = self.recv(deserialize, 1000)

            return msg
        else:
            return self.recv(deserialize, timeout)

    @retry
    def send_string(self, data: str):
        self.socket.send_string(data, flags=self.flags)

    def recv_string(
        self, timeout: tp.Optional[int] = None
    ) -> tp.Optional[str]:  # in milliseconds
        if timeout is not None and self.socket.poll(timeout=timeout) == 0:  # wait
            return None  # timeout reached before any events were queued
        else:
            return self.socket.recv_string(
                flags=self.flags
            )  # events queued within our time limit

    def request_as_string(
        self, data: str, timeout: tp.Optional[int] = None  # in milliseconds
    ) -> tp.Optional[str]:
        self.send_string(data)
        return self.recv_string(timeout)


@dataclass
class ZMQWorker:
    context: zmq.Context
    socket: zmq.Socket
    flags: int = 0

    def close(self):
        self.socket.close()

    @retry
    def send(self, data: tp.Any, serialize: bool = True):
        if serialize:
            data = Serialize.dump(data)
        self.socket.send(data, flags=self.flags)

    @retry
    def send_multipart(self, data: tp.List[tp.Any], serialize: bool = True):
        if serialize:
            data = Serialize.dumps(data)
        self.socket.send_multipart(data, flags=self.flags)

    def recv_multipart(
        self,
        deserialize: bool = True,
        timeout: tp.Optional[int] = None,  # in milliseconds
    ) -> tp.List:
        if timeout and self.socket.poll(timeout=timeout) == 0:
            return []
        else:
            data = self.socket.recv_multipart(flags=self.flags)
            if data:
                if deserialize:
                    data = Serialize.loads(data)

                return data
            else:
                return []


@dataclass
class ZMQProxy:
    context: zmq.Context
    frontend: zmq.Socket
    backends: tp.List[ZMQClient]
    poller: zmq.Poller
    socks: tp.Dict[zmq.Socket, tp.Any] = None  # type: ignore
    flags: int = zmq.NOBLOCK

    def close(self):
        self.frontend.close()
        [backend.close() for backend in self.backends]

    def pool(self, timeout: tp.Optional[int] = None):  # in milliseconds
        self.socks = dict(self.poller.poll(timeout))

    def is_frontend_ready(self) -> bool:
        return self.socks.get(self.frontend) == zmq.POLLIN

    def request(self, data: tp.Any) -> tp.List:
        results = []
        for backend in self.backends:
            results.append(backend.request(data))

        return results

    @retry
    def frontend_send_multipart(self, data: tp.List[tp.Any]):
        if not isinstance(data, list):
            data = [data]
        self.frontend.send_multipart(data, flags=self.flags)

    def frontend_recv_multipart(self) -> tp.List[tp.Any]:
        try:
            return self.frontend.recv_multipart(flags=self.flags)
        except zmq.ZMQError:
            return []

    def backend_send_multipart(self, data: tp.List[tp.Any]):
        @retry
        def send(backend):
            backend.socket.send_multipart(data, flags=self.flags)

        for b in self.backends:
            send(b)


class ZMQPatterns:
    @staticmethod
    def __create_socket_and_bind(
        context: zmq.Context, addr: str, socket_type
    ) -> zmq.Socket:
        socket = context.socket(socket_type)
        socket.bind(f"tcp://{addr}")
        socket.setsockopt(zmq.LINGER, 0)
        return socket

    @staticmethod
    def __create_socket_and_connect(
        context: zmq.Context, addr: str, socket_type
    ) -> zmq.Socket:
        socket = context.socket(socket_type)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(f"tcp://{addr}")
        return socket

    @staticmethod
    def __get_req(context: zmq.Context, addr: str) -> zmq.Socket:
        return ZMQPatterns.__create_socket_and_connect(context, addr, zmq.REQ)

    @staticmethod
    def __get_rep(context: zmq.Context, addr: str) -> zmq.Socket:
        return ZMQPatterns.__create_socket_and_connect(context, addr, zmq.REP)

    @staticmethod
    def __get_sub(context: zmq.Context, addr: str) -> zmq.Socket:
        return ZMQPatterns.__create_socket_and_connect(context, addr, zmq.SUB)

    @staticmethod
    def __get_pub(context: zmq.Context, addr: str) -> zmq.Socket:
        return ZMQPatterns.__create_socket_and_bind(context, addr, zmq.PUB)

    @staticmethod
    def __get_router(context: zmq.Context, addr: str) -> zmq.Socket:
        return ZMQPatterns.__create_socket_and_bind(context, addr, zmq.ROUTER)

    @staticmethod
    def __get_dealer(context: zmq.Context, addr: str, bind: bool = True) -> zmq.Socket:
        if bind:
            return ZMQPatterns.__create_socket_and_bind(context, addr, zmq.DEALER)
        else:
            return ZMQPatterns.__create_socket_and_connect(context, addr, zmq.DEALER)

    @staticmethod
    def __get_poller(sockets: tp.List[zmq.Socket]) -> zmq.Poller:
        poller = zmq.Poller()
        for s in sockets:
            poller.register(s, zmq.POLLIN)
        return poller

    @classmethod
    def server(
        cls,
        addr_for_clients: tp.Optional[str] = None,
        addr_for_workers: tp.Optional[str] = None,
    ) -> ZMQServer:
        if addr_for_clients is None and addr_for_workers is None:
            raise AttributeError("Least one socket address must be specified")

        sockets = []
        context = zmq.Context.instance()

        if addr_for_clients:
            log_to_file(trace(cls, f"bind socket {addr_for_clients} for clients"))
            frontend = cls.__get_router(context, addr_for_clients)
            sockets.append(frontend)
        else:
            frontend = None

        if addr_for_workers:
            log_to_file(trace(cls, f"bind socket {addr_for_workers} for workers"))
            backend = cls.__get_dealer(context, addr_for_workers)
            sockets.append(backend)
        else:
            backend = None

        poller = cls.__get_poller(sockets)

        return ZMQServer(
            context=context, frontend=frontend, backend=backend, poller=poller
        )

    @classmethod
    def client(cls, server_addr: str) -> ZMQClient:
        log_to_file(trace(cls, f"connection to {server_addr}"))

        context = zmq.Context()
        socket = cls.__get_req(context, server_addr)

        return ZMQClient(context=context, socket=socket)

    @classmethod
    def async_client(cls, server_addr: str) -> ZMQClient:
        log_to_file(trace(cls, f"connection to {server_addr}"))

        context = zmq.Context()
        socket = cls.__get_dealer(context, server_addr, bind=False)

        return ZMQClient(context=context, socket=socket, flags=zmq.NOBLOCK)

    @classmethod
    def worker(cls, server_addr: str) -> ZMQWorker:
        log_to_file(trace(cls, f"connection to {server_addr}"))

        context = zmq.Context()
        socket = cls.__get_rep(context, server_addr)

        return ZMQWorker(context=context, socket=socket)

    @classmethod
    def proxy(cls, proxy_addr: str, server_addrs: tp.List[str]) -> ZMQProxy:
        try:
            log_to_file(trace(cls, f"bind socket {proxy_addr}"))

            context = zmq.Context()
            frontend = cls.__get_router(context, proxy_addr)
            poller = cls.__get_poller([frontend])

            backends = []
            for addr in server_addrs:
                log_to_file(trace(cls, f"connection to {addr}"))
                backends.append(cls.async_client(addr))

        except zmq.error.ZMQError as e:
            raise e

        return ZMQProxy(
            context=context,
            frontend=frontend,
            backends=backends,
            poller=poller,
        )
