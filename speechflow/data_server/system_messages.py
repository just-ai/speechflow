from strenum import StrEnum

__all__ = ["DataServerMessages", "DataClientMessages"]


class DataServerMessages(StrEnum):
    READY = "ready"
    NO_WORKERS = "workers not found"
    OVERLOAD = "server overload"
    SKIP_BATCH = "batch skipped"
    QUEUE_EXCEEDED = "request batch queue exceeded"
    QUEUE_CLEARED = "batch queue cleared"
    EPOCH_ENDING = "sampler has reached the end of an epoch"
    EPOCH_COMPLETE = "all batches of an epoch have been processed"
    ABORT = "abort processing the current batch"
    RESET = "reset sampler state"


class DataClientMessages(StrEnum):
    INFO = "info"
    IS_READY = "is ready"
    GET_BATCH = "get batch"
    EPOCH_COMPLETE = "all batches received"
    ABORT = "abort processing"
    RESET = "reset"


if __name__ == "__main__":
    print(DataServerMessages.NO_WORKERS)
