import typing as tp
import logging

from speechflow.concurrency import ProcessWorker
from speechflow.data_pipeline.core import Batch, DataPipeline
from speechflow.data_server.patterns import ZMQPatterns, ZMQProxy
from speechflow.data_server.system_messages import DataClientMessages as DCM
from speechflow.io import Config, tp_PATH
from speechflow.logging import trace
from speechflow.utils.init import init_class_from_config
from speechflow.utils.serialize import Serialize
from speechflow.utils.sockopt import find_free_port

__all__ = ["Proxy"]

LOGGER = logging.getLogger("root")


class Proxy(ProcessWorker):
    def __init__(
        self,
        server_addrs: tp.List[str],
    ):
        ProcessWorker.__init__(self, daemon=True)
        self._server_addrs = server_addrs
        self._proxy_addr = f"127.0.0.1:{find_free_port()}"
        self._zmq_proxy: ZMQProxy = None  # type: ignore

    @property
    def address(self):
        return self._proxy_addr

    @staticmethod
    def init_from_config(
        file_path: tp_PATH,
        value_select: tp.Optional[tp.List[str]] = None,
        server_addr: tp.Optional[str] = None,
        config_section: str = "proxy_server",
    ) -> "Proxy":
        cfg = Config.create_from_file(
            file_path, section=config_section, value_select=value_select
        )

        if server_addr:
            cfg["server_addrs"] = [server_addr]
        else:
            if cfg.is_empty:
                raise RuntimeError("Missing proxy settings!")

        clsmembers = {v.__name__: v for v in Proxy.__subclasses__()}
        custom_proxy_cls = clsmembers[cfg["type"]]
        custom_proxy_cls = init_class_from_config(custom_proxy_cls, cfg)
        return custom_proxy_cls()

    def on_start(self):
        self._zmq_proxy = ZMQPatterns.proxy(self._proxy_addr, self._server_addrs)
        message = f"Start {self.__class__.__name__} Server {self._proxy_addr}"
        LOGGER.info(trace(self, message=message))

    def on_finish(self):
        self._zmq_proxy.close()
        LOGGER.info(
            trace(
                self,
                message=f"Finish {self.__class__.__name__} Server {self._proxy_addr}",
            )
        )

    def batch_preprocessing(self, batch: Batch) -> tp.List[Batch]:
        return [batch]

    def do_preprocessing(self, response: tp.List[bytes]) -> tp.List[bytes]:
        new_response = []
        for _bytes in response:
            if _bytes == b"" or _bytes[0] == 0 or b"info:" in _bytes[:100]:
                new_response.append(_bytes)
            else:
                batch = Serialize.load(_bytes)
                if not isinstance(batch, Batch):
                    new_response.append(_bytes)
                    continue
                else:
                    try:
                        batches = self.batch_preprocessing(batch)
                        new_response += Serialize.dumps(batches)
                    except Exception as e:
                        LOGGER.error(trace(self, e))

        return new_response

    def do_work_once(self):
        try:
            self._zmq_proxy.pool(timeout=10)

            if self._zmq_proxy.is_frontend_ready():
                message = self._zmq_proxy.frontend_recv_multipart()
                if message is not None:
                    request = Serialize.load(message[-1])

                    if request["message"] == DCM.INFO:
                        all_info = []
                        for b in self._zmq_proxy.backends:
                            info = b.request(
                                message,
                                serialize=False,
                                deserialize=False,
                                multipart=True,
                            )
                            all_info.append(Serialize.load(info[-1]))

                        message[-1] = Serialize.dump(
                            DataPipeline.aggregate_info(all_info)
                        )
                        self._zmq_proxy.frontend_send_multipart(message)
                    else:
                        self._zmq_proxy.backend_send_multipart(message)

            for b in self._zmq_proxy.backends:
                response = b.recv_multipart(
                    deserialize=False,
                    timeout=1,
                    max_num_message=5,
                )
                if response:
                    response = self.do_preprocessing(response)
                    self._zmq_proxy.frontend_send_multipart(response)

        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            LOGGER.error(trace(self, e))
