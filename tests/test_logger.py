import logging

from speechflow.concurrency import ProcessWorker
from speechflow.logging.server import LoggingServer


class DummyProcess(ProcessWorker):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("root")

    def do_work_once(self):
        self.logger.info("Forked Process logger created")
        for i in range(5):
            self.logger.info(f"Forked Process tick!, i = {i}")

        self.deactivate()


def test_logger():
    with LoggingServer.ctx(log_name="name_1") as logger_1:
        logger_1.info("start-1")
        with LoggingServer.ctx(log_name="name_2") as logger_2:
            print(f"Statr logger {logger_2.name}")
            logger_2.info("start-2")
            logger_1.info("start DummyProcess")
            processes = [DummyProcess() for _ in range(2)]
            [process.start() for process in processes]
            [process.join() for process in processes]
            [process.finish() for process in processes]
            logger_2.info("stop-2")
        logger_1.info("stop-1")

    assert "start DummyProcess" in " ".join(logger_1.get_all_messages())
    assert "stop-1" in " ".join(logger_1.get_all_messages())
    assert "stop-2" in " ".join(logger_2.get_all_messages())


if __name__ == "__main__":
    test_logger()
