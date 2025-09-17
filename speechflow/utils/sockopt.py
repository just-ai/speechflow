import time
import socket

from contextlib import closing

__all__ = ["find_free_port"]


def find_free_port():
    time.sleep(1)
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_nane = s.getsockname()[1]
        s.close()
        return port_nane
