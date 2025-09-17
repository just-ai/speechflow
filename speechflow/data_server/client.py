import uuid
import typing as tp
import logging

from speechflow.data_server.patterns import ZMQPatterns
from speechflow.data_server.server import SubscriberTypes
from speechflow.logging import trace
from speechflow.utils.dictutils import flatten_dict

__all__ = ["DataClient"]

LOGGER = logging.getLogger("root")


class DataClient:
    def __init__(
        self, server_addr: str, sub_type: str = SubscriberTypes.CLIENT, uid: str = None
    ):
        self._uid = uid if uid else uuid.uuid4().hex
        self._server_addr = server_addr
        self._zmq_client = ZMQPatterns.async_client(server_addr)
        self._info = self.request(
            {"message": "info", "sub_type": sub_type, "client_uid": self._uid}
        )
        LOGGER.debug(trace(self, message=f"Start DataClient {self._server_addr}"))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._zmq_client.close()
        LOGGER.debug(trace(self, message=f"Finish DataClient {self._server_addr}"))

    @property
    def uid(self):
        return self._uid

    @property
    def server_address(self):
        return self._server_addr

    @property
    def info(self) -> tp.Dict:
        return self._info

    def find_info(self, name: str, default: tp.Any = None, section: str = None) -> tp.Any:
        if section is None:
            flatten_info = flatten_dict(self.info)
        else:
            flatten_info = flatten_dict(self.info[section])

        found = []
        for key, field in flatten_info.items():
            if key.endswith(name):
                if field not in [None, {}]:
                    found.append(field)

        if len(found):
            return found[0]
        else:
            return default

    def find_section(self, name_or_value: str, default: tp.Any = None):
        if name_or_value in self.info:
            return self.info[name_or_value]
        else:
            section_path = None
            flatten_info = flatten_dict(self.info)
            for key, field in flatten_info.items():
                if name_or_value in key:
                    section_path = key.split(".")[1:]
                if isinstance(field, str) and name_or_value == field:
                    section_path = key.split(".")[1:-1]

        if section_path:
            section = self.info
            for name in section_path:
                section = section[name]
            return section

        return default

    def request(
        self,
        message: tp.Any,
        deserialize: bool = True,
        timeout: tp.Optional[int] = None,  # in milliseconds
    ) -> tp.Optional[tp.Any]:
        message["client_uid"] = self._uid
        return self._zmq_client.request(message, deserialize=deserialize, timeout=timeout)

    def send(self, message: tp.Any):
        message["client_uid"] = self._uid
        self._zmq_client.send(message)

    def recv_multipart(
        self,
        deserialize: bool = True,
        timeout: tp.Optional[int] = None,  # in milliseconds
        max_num_message: tp.Optional[int] = None,
    ) -> tp.List[tp.Any]:
        return self._zmq_client.recv_multipart(deserialize, timeout, max_num_message)
