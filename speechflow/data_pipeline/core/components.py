import pickle
import typing as tp
import inspect
import logging
import functools

from copy import copy, deepcopy
from os import environ as env
from pathlib import Path

from speechflow.data_pipeline import (
    collate_functions,
    datasample_processors,
    dataset_parsers,
    samplers,
)
from speechflow.data_pipeline.core import (
    Batch,
    DataSample,
    Dataset,
    PipeRegistry,
    Singleton,
    data_processor,
)
from speechflow.data_pipeline.core.abstract import (
    AbstractDataProcessor,
    AbstractDatasetParser,
)
from speechflow.data_pipeline.core.parser_types import Metadata
from speechflow.io import Config, check_path, generate_file_list, read_file_list, tp_PATH
from speechflow.logging import trace
from speechflow.utils.init import (
    get_default_args,
    init_class_from_config,
    init_method_from_config,
)
from speechflow.utils.serialize import Serialize

__all__ = [
    "DataPipeline",
    "PipelineComponents",
    "init_metadata_preprocessing_from_config",
    "init_data_preprocessing_from_config",
]

LOGGER = logging.getLogger("root")


def _find_handlers(method: tp.Callable, handlers: tp.Dict[str, tp.Any]):
    method_signature = inspect.signature(method)
    ret = {}
    for param_name, param_type in method_signature.parameters.items():
        _metaclass = type(param_type.annotation)
        if _metaclass == type or _metaclass == Singleton:
            for name, handler in handlers.items():
                if isinstance(handler, param_type.annotation):
                    ret[param_name] = handler
    return ret


def init_singleton_handlers_from_config(
    cfg: Config,
    data_subset_name: tp.Optional[str],
    preinit_singleton_handlers: tp.Optional[tp.Dict[str, tp.Callable]] = None,
) -> tp.Dict[str, tp.Callable]:
    cfg_handlers = cfg.section("singleton_handlers")

    handlers = {}
    for class_name in cfg_handlers.get("handlers", []):
        if (
            preinit_singleton_handlers is not None
            and class_name in preinit_singleton_handlers
        ):
            handlers[class_name] = preinit_singleton_handlers[class_name]
        else:
            config = cfg_handlers.section(class_name)
            config.setdefault("data_subset_name", data_subset_name)
            cls = getattr(datasample_processors, class_name)
            handlers[class_name] = init_class_from_config(cls, config)(cfg_data=cfg)

    return handlers


def init_metadata_preprocessing_from_config(
    cls,
    cfg: Config,
    singleton_handlers: tp.Optional[tp.Dict[str, tp.Callable]] = None,
) -> tp.List[tp.Callable]:
    pipe = cfg.get("pipe", ())
    pipe_cfg = cfg.section("pipe_cfg")

    steps = []
    for step_name in pipe:
        method = getattr(cls, step_name)
        step_config: Config = pipe_cfg.section(step_name)

        if singleton_handlers:
            step_config.update(_find_handlers(method, singleton_handlers))

        steps.append(functools.partial(method, **step_config))

    return steps  # type: ignore


def init_data_preprocessing_from_config(
    cfg: Config,
    singleton_handlers: tp.Optional[tp.Dict[str, tp.Callable]] = None,
    cache: tp.Optional[tp.Dict] = None,
) -> tp.List[tp.Callable]:
    pipe = cfg.get("pipe", ())
    pipe_cfg = cfg.section("pipe_cfg")

    if cache is None:
        cache = {}

    steps = []
    for step_name in pipe:
        step_config = pipe_cfg.section(step_name)
        init_params = step_config.copy()

        if step_name in cache and cache[step_name][1] == init_params:
            steps.append(cache[step_name][0])
            continue

        if step_config.get("disable", False):
            continue

        if "type" in step_config:  # we need a class instance, not a function!
            obj = getattr(datasample_processors, step_config["type"])
            if (
                env.get("DEVICE")
                and "device" in get_default_args(obj)
                and "device" not in step_config
            ):
                step_config["device"] = env.get("DEVICE")

            if isinstance(obj, type):
                instance = init_class_from_config(obj, step_config)()
                method = instance.process
            elif callable(obj):
                method = init_method_from_config(obj, step_config)
            else:
                raise NotImplementedError
        else:
            method = getattr(datasample_processors, step_name)
            if (
                env.get("DEVICE")
                and "device" in get_default_args(method)
                and "device" not in step_config
            ):
                step_config["device"] = env.get("DEVICE")

            method = init_method_from_config(method, step_config)

        if singleton_handlers:
            additional_arguments = _find_handlers(method, singleton_handlers)
            method = functools.partial(method, **additional_arguments)

        try:
            setattr(method, "init_params", init_params)
        except AttributeError:
            pass

        steps.append(method)
        cache[step_name] = (method, init_params)

    return steps


def init_collate_from_config(
    cfg: Config,
    singleton_handlers: tp.Optional[tp.Dict[str, tp.Callable]] = None,
) -> tp.Union[tp.Callable, None]:
    if cfg.is_empty:
        return None

    collate_cls = getattr(collate_functions, cfg["type"])
    method = init_class_from_config(collate_cls, cfg)()

    if singleton_handlers:
        arguments = _find_handlers(method, singleton_handlers)
        method = functools.partial(method, **arguments)

    return method


class PipelineComponents:
    """Encompasses the whole data data_pipeline from disk up to batch processor for
    specific model.

    In practice it serves as a contaner for components
    so they can be shared and used in different places.

    Initialization is split into two stages:
      - resource-light component initizalization (__init__)
      - resource-heavy metadata reading and preprocessing (load_data)

    """

    def __init__(
        self,
        cfg: Config,
        data_subset_name: tp.Optional[str] = None,
        preinit_singleton_handlers: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        cache: tp.Optional[tp.Dict] = None,
    ):
        if data_subset_name:
            cfg = cfg.trim(key=data_subset_name)

        self._cfg = cfg
        self._subset_name = data_subset_name

        cfg.setdefault("processor", {"type": "DataProcessor"})
        processor_cls = getattr(data_processor, cfg["processor"]["type"])

        cfg.setdefault("parser", {"type": "SimpleDSParser"})
        parser_cls = getattr(dataset_parsers, cfg["parser"]["type"])

        cfg.setdefault("sampler", {"type": "SimpleSampler"})
        sampler_cls = getattr(samplers, cfg["sampler"]["type"])

        self.singleton_handlers = init_singleton_handlers_from_config(
            cfg, data_subset_name, preinit_singleton_handlers
        )
        self.metadata_preprocessing = init_metadata_preprocessing_from_config(
            parser_cls, cfg.section("parser"), self.singleton_handlers
        )
        self.data_preprocessing = init_data_preprocessing_from_config(
            cfg.section("preproc"),
            self.singleton_handlers,
            cache,
        )
        self.collate = init_collate_from_config(
            cfg.section("collate"), self.singleton_handlers
        )
        self.sampler = init_class_from_config(sampler_cls, cfg.section("sampler"))()

        self.dataset_parser: AbstractDatasetParser = init_class_from_config(
            parser_cls, cfg.section("parser")
        )(preproc_fn=self.metadata_preprocessing)

        self.data_processor: AbstractDataProcessor = init_class_from_config(
            processor_cls, cfg.section("processor")
        )(self.data_preprocessing, self.collate)

    def _apply_handlers(self, dataset: Dataset) -> Dataset:
        for name, handler in self.singleton_handlers.items():
            LOGGER.info(trace(self, f"Apply {name}"))
            if hasattr(handler, "__call__"):
                dataset = handler(data=dataset)

        return dataset

    def _init_sampler(self, dataset: Dataset):
        dataset = self._apply_handlers(dataset)
        if len(dataset) == 0:
            raise ValueError(f"Dataset '{self.subset_name}' is empty!")

        LOGGER.info(trace(self, f"Prepare dataset for {self.sampler.__class__.__name__}"))
        self.sampler.set_dataset(dataset)

    @property
    def subset_name(self):
        return self._subset_name

    @property
    def config(self):
        return self._cfg.copy()

    def load_data(self, file_list, n_processes: int = 1):
        dataset = self.dataset_parser.read_datasamples(
            file_list=file_list,
            data_root=self._cfg.section("dirs").get("data_root"),
            n_processes=n_processes,
        )
        self._init_sampler(dataset)

    def set_dataset(self, dataset: Dataset):
        assert len(dataset) > 0, "dataset is empty!"
        self._init_sampler(dataset)  # type: ignore

    def get_file_list(self) -> tp.Tuple[str, ...]:
        return self.sampler.dataset.get_file_list()

    @check_path(assert_file_exists=True)
    def metadata_from_file(
        self, file_path: tp_PATH, label: tp.Optional[str] = None
    ) -> tp.List[Metadata]:
        metadata = self.dataset_parser.reader(file_path, label)
        return metadata

    def metadata_to_datasample(
        self, metadata: tp.Union[tp.List[Metadata], Dataset], as_dataset: bool = False
    ) -> tp.Union[tp.List[DataSample], Dataset]:
        metadata = self.dataset_parser.do_preprocessing(
            metadata, self.metadata_preprocessing
        )
        dataset = self.dataset_parser.to_datasample(metadata)
        return dataset if as_dataset else dataset.to_list()

    def preprocessing_datasample(
        self, samples: tp.List[DataSample], skip_corrupted_samples: bool = True
    ) -> tp.List[DataSample]:
        return self.data_processor.do_preprocessing(
            samples,
            self.data_preprocessing,
            skip_corrupted_samples=skip_corrupted_samples,
        )

    def to_batch(self, samples: tp.List[DataSample]) -> Batch:
        collated_samples = self.collate(samples) if self.collate else None
        return Batch(
            size=len(samples),
            data_samples=samples,
            collated_samples=collated_samples,
        )

    def datasample_to_batch(
        self, samples: tp.List[DataSample], skip_corrupted_samples: bool = True
    ) -> Batch:
        samples = self.preprocessing_datasample(samples, skip_corrupted_samples)
        collated_samples = self.collate(samples) if self.collate else None
        return Batch(
            size=len(samples),
            data_samples=samples,
            collated_samples=collated_samples,
        )

    def metadata_to_batch(
        self, metadatas: tp.List[Metadata], skip_corrupted_samples: bool = True
    ) -> Batch:
        samples = self.metadata_to_datasample(metadatas)
        return self.datasample_to_batch(
            samples, skip_corrupted_samples=skip_corrupted_samples
        )

    def with_ignored_fields(
        self,
        ignored_metadata_fields: tp.AbstractSet[str] = frozenset(),
        ignored_data_fields: tp.AbstractSet[str] = frozenset(),
    ) -> "PipelineComponents":
        """Filters out preprocessing steps which use specified fields.

        Returns a new instance of PipelineComponents with insonsistent state. Use
        with caution (ex.: evaluation scripts)

        """
        new_pipeline_components = copy(self)

        if self.metadata_preprocessing:
            metadata_fns = PipeRegistry.filter(
                self.metadata_preprocessing,
                lambda fields: not fields & ignored_metadata_fields,
            )
        else:
            metadata_fns = []

        if self.data_preprocessing:
            data_fns = PipeRegistry.filter(
                self.data_preprocessing, lambda fields: not fields & ignored_data_fields
            )
        else:
            data_fns = []

        new_pipeline_components.metadata_preprocessing = metadata_fns
        new_pipeline_components.data_preprocessing = data_fns
        return new_pipeline_components

    def with_ignored_handlers(
        self,
        ignored_metadata_handlers: tp.AbstractSet[str] = frozenset(),
        ignored_data_handlers: tp.AbstractSet[str] = frozenset(),
    ) -> "PipelineComponents":
        """Filters out preprocessing steps which use specified handlers.

        Returns a new instance of PipelineComponents with insonsistent state. Use
        with caution (ex.: evaluation scripts)

        """
        new_pipeline_components = copy(self)

        if ignored_metadata_handlers:
            metadata_fns = PipeRegistry.filter(
                self.metadata_preprocessing,
                lambda name: not name & ignored_metadata_handlers,
                by_field=False,
                by_handler_name=True,
            )
            new_pipeline_components.metadata_preprocessing = metadata_fns

        if ignored_data_handlers:
            data_fns = PipeRegistry.filter(
                self.data_preprocessing,
                lambda name: not name & ignored_data_handlers,
                by_field=False,
                by_handler_name=True,
            )
            new_pipeline_components.data_preprocessing = data_fns

        return new_pipeline_components

    @staticmethod
    def _remove_handler(
        handlers: tp.List,
        name: str,
        init_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> tp.List:
        new_handlers = []
        for proc in handlers:
            if name in str(proc):
                if init_params and set(init_params.keys()).issubset(
                    proc.init_params.keys()
                ):
                    if all(proc.init_params[k] == v for k, v in init_params.items()):
                        continue
                else:
                    continue

            new_handlers.append(proc)

        return new_handlers

    def remove_metadata_handler(
        self, name: str, init_params: tp.Optional[tp.Dict[str, tp.Any]] = None
    ) -> "PipelineComponents":
        new_pipeline_components = copy(self)
        handlers = new_pipeline_components.metadata_preprocessing
        new_pipeline_components.metadata_preprocessing = self._remove_handler(
            handlers, name, init_params
        )
        return new_pipeline_components

    def remove_data_handler(
        self, name: str, init_params: tp.Optional[tp.Dict[str, tp.Any]] = None
    ) -> "PipelineComponents":
        new_pipeline_components = copy(self)
        handlers = new_pipeline_components.data_preprocessing
        new_pipeline_components.data_preprocessing = self._remove_handler(
            handlers, name, init_params
        )
        return new_pipeline_components

    def filter(
        self,
        specified_metadata_handlers: tp.AbstractSet[str] = frozenset(),
        specified_data_handlers: tp.AbstractSet[str] = frozenset(),
        before_metadata_handler: tp.Optional[str] = None,
        before_data_handler: tp.Optional[str] = None,
        after_metadata_handler: tp.Optional[str] = None,
        after_data_handler: tp.Optional[str] = None,
    ) -> "PipelineComponents":
        """Filters out preprocessing steps which use name of handlers.

        Returns a new instance of PipelineComponents with insonsistent state. Use
        with caution (ex.: evaluation scripts)

        """
        new_pipeline_components = copy(self)

        metadata_fns = []
        if before_metadata_handler:
            for h in self.metadata_preprocessing:
                if before_metadata_handler in str(h):
                    break
                metadata_fns.append(h)
        elif after_metadata_handler:
            flag = False
            for h in self.metadata_preprocessing:
                if after_metadata_handler in str(h):
                    flag = True
                if flag:
                    metadata_fns.append(h)
        else:
            metadata_fns = self.metadata_preprocessing

        data_fns = []
        if before_data_handler:
            for h in self.data_preprocessing:
                if before_data_handler in str(h):
                    break
                data_fns.append(h)
        elif after_data_handler:
            data_fns = []
            flag = False
            for h in self.data_preprocessing:
                if after_data_handler in str(h):
                    flag = True
                if flag:
                    data_fns.append(h)
        else:
            data_fns = self.data_preprocessing

        if specified_metadata_handlers:
            metadata_fns = [
                h
                for h in metadata_fns
                if any(fn in str(h) for fn in specified_metadata_handlers)
            ]

        if specified_data_handlers:
            data_fns = [
                h for h in data_fns if any(fn in str(h) for fn in specified_data_handlers)
            ]

        new_pipeline_components.metadata_preprocessing = metadata_fns
        new_pipeline_components.data_preprocessing = data_fns
        return new_pipeline_components


class DataPipeline:
    """A convenient wrapper over a Dict[str, PipelineComponents]"""

    def __init__(self, cfg: Config):
        self._cfg: Config = cfg
        self._tag: str = str(cfg.get("tag", ""))
        self._flist_by_subsets: tp.Dict[str, tp.List[tp_PATH]] = {}
        self._pipelines: tp.Dict[str, PipelineComponents] = {}

    def __getitem__(self, name: str) -> PipelineComponents:
        assert self._pipelines, "Call first of init_components function!"
        return self._pipelines[name]

    @staticmethod
    @check_path(assert_file_exists=True)
    def init_from_config(
        file_path: tp_PATH,
        value_select: tp.Optional[tp.List[str]] = None,
    ) -> "DataPipeline":
        cfg = Config.create_from_file(file_path, value_select=value_select)
        return DataPipeline(cfg)

    @staticmethod
    def init_from_components(
        data_subsets: tp.List[str],
        parser=None,
        preprocessing=None,
        collate=None,
        sampler=None,
        singleton_handlers=None,
        **kwargs,
    ):
        cfg = Config.empty({"dataset"})
        cfg["dataset"]["subsets"] = data_subsets
        cfg.update(**kwargs)

        data_pipeline = DataPipeline(cfg)
        data_pipeline.init_components()

        def set_component(_pipe, _component_name, _component):
            if _component is not None:
                if not isinstance(_component, tp.Mapping):
                    setattr(_pipe, _component_name, deepcopy(_component))
                else:
                    setattr(_pipe, _component_name, _component[_pipe.subset_name])

        for name in data_subsets:
            pipe = data_pipeline._pipelines[name]
            pipe.metadata_preprocessing = []

            set_component(pipe, "dataset_parser", parser)
            set_component(pipe, "data_preprocessing", preprocessing)
            set_component(pipe, "collate", collate)
            set_component(pipe, "sampler", sampler)

            pipe.singleton_handlers = (
                singleton_handlers if singleton_handlers is not None else {}
            )

            pipe.data_processor = init_class_from_config(
                data_processor.DataProcessor, cfg.section("processor")
            )(pipe.data_preprocessing, pipe.collate)

        return data_pipeline

    @property
    def subsets(self) -> tp.List[str]:
        return list(self._pipelines.keys())

    @property
    def tag(self):
        return self._tag

    @property
    def config(self):
        return self._cfg.copy()

    @property
    def config_raw(self):
        return self._cfg.raw_file

    @property
    def is_init(self) -> bool:
        return True if self._pipelines else False

    @property
    def is_data_loaded(self) -> bool:
        if self.is_init:
            for pipe in self._pipelines.values():
                if pipe.sampler is None or not pipe.sampler.dataset:
                    return False
            return True
        else:
            return False

    def init_components(
        self,
        preinit_singleton_handlers: tp.Optional[
            tp.Dict[str, tp.Dict[str, tp.Callable]]
        ] = None,
    ):
        if self.is_init or self._cfg.section("dataset").is_empty:
            return

        cache = {}
        for name in self._cfg.section("dataset")["subsets"]:
            self._pipelines[name] = PipelineComponents(
                self._cfg, name, (preinit_singleton_handlers or {}).get(name), cache
            )

    def load_data(self, n_processes: int = 1):
        assert self._pipelines, "Call of init_components function before loading data!"
        if self.is_data_loaded:
            return

        flist_path = self._cfg.section("dirs").get("file_list")
        if isinstance(flist_path, tp.MutableMapping):
            self._flist_by_subsets = flist_path
        else:
            self._flist_by_subsets = self.get_file_list(flist_path)

        for name, pipe in self._pipelines.items():
            pipe.load_data(self._flist_by_subsets[name], n_processes)

    def remove_pipeline(self, name: str):
        self._pipelines.pop(name)

    @check_path(assert_file_exists=True)
    def get_file_list(
        self, flist_path: tp.Optional[tp_PATH] = None
    ) -> tp.Dict[str, tp.List[tp_PATH]]:
        if flist_path is None or not Path(flist_path).exists():
            flist_path = generate_file_list(
                data_root=self._cfg.section("dirs")["data_root"],
                ext=self._cfg.section("file_search")["ext"],
                with_subfolders=self._cfg.section("file_search").get(
                    "with_subfolders", True
                ),
            )

        _read_flist = init_method_from_config(
            read_file_list, self._cfg.section("dataset")
        )
        return _read_flist(flist_path=flist_path)

    def set_file_list(self, files_by_subsets: tp.Dict[str, tp.List[tp_PATH]]):
        self._cfg.create_section({"dirs"})
        self._cfg["dirs"]["file_list"] = files_by_subsets
        assert set(self.subsets) == set(files_by_subsets.keys())

    def get_info(
        self, object_size_limit: float = 10, size_format=Serialize.Format.MB
    ) -> tp.Dict:  # type: ignore
        info = {
            "data_config_raw": self.config_raw,
            "data_config": self.config,
            "subsets": self.subsets,
            "epoch_size": {name: self[name].sampler.epoch_size for name in self.subsets},
        }

        temp = {}
        for name in self.subsets:
            temp[name] = self[name].sampler
            cfg = self[name].config.section("sampler")
            sampler_cls = getattr(samplers, cfg["type"])
            self[name].sampler = init_class_from_config(sampler_cls, cfg)()

        try:
            LOGGER.debug(trace(self, message="Pickling DataPipeline"))
            info["data_pipeline"] = Serialize.dump(self)
        except (TypeError, pickle.PickleError) as e:
            LOGGER.debug(
                trace(self, e, "Current pipelines configuration not support pickle!")
            )
        finally:
            for name in self.subsets:
                self[name].sampler = temp[name]

        if object_size_limit > 0:
            singleton_handlers = {}
            for name in self.subsets:
                handlers = self[name].singleton_handlers
                handlers = {
                    k: v
                    for k, v in handlers.items()
                    if Serialize.get_obj_size(v, size_format) <= object_size_limit
                }
                singleton_handlers[name] = handlers
        else:
            singleton_handlers = {
                name: self[name].singleton_handlers for name in self.subsets
            }

        for name in self.subsets:
            handlers = singleton_handlers.pop(name)
            for key, handler in list(handlers.items()):
                if isinstance(type(handler), Singleton):
                    singleton_handlers[key] = handlers.pop(key)

        dataset = {name: tuple(self[name].get_file_list()) for name in self.subsets}
        if 0 < object_size_limit < Serialize.get_obj_size(dataset, size_format):
            dataset = {}

        info.update({"singleton_handlers": singleton_handlers, "dataset": dataset})
        return info

    @staticmethod
    def aggregate_info(
        all_info: tp.List[tp.Dict[str, tp.Any]],
        target_keys: tp.Tuple[str, ...] = ("data_config_raw", "singleton_handlers"),
    ) -> tp.Dict:
        if len(all_info) == 1:
            return all_info[0]

        aggr_info = deepcopy(all_info[0])
        for key, field in aggr_info.items():
            if key not in target_keys:
                continue

            if key != "singleton_handlers":
                aggr_info[key] = [field]

        for info in all_info[1:]:
            assert aggr_info["subsets"] == info["subsets"]

            for key, field in info.items():
                if key not in target_keys:
                    continue

                if key != "singleton_handlers":
                    aggr_info[key].append(field)
                else:
                    for name, instance in field.items():
                        if hasattr(instance, "aggregate"):
                            aggr_info[key][name] = instance.aggregate(
                                aggr_info[key][name], instance
                            )
                        else:
                            if not isinstance(aggr_info[key][name], tp.MutableSequence):
                                aggr_info[key][name] = [aggr_info[key][name]]

                            aggr_info[key][name].append(instance)

        aggr_info["orig_info"] = all_info
        return aggr_info


if __name__ == "__main__":
    from speechflow.utils.fs import get_root_dir

    _cfg = Config.empty({"dirs", "file_search", "dataset"})
    _cfg["dirs"]["data_root"] = get_root_dir() / "examples/simple_datasets/speech/SEGS"
    _cfg["file_search"]["ext"] = ".TextGridStage3"
    _cfg["dataset"]["subsets"] = ["train", "test"]

    _data_pipe = DataPipeline(_cfg)
    _data_pipe.init_components()
    _data_pipe.load_data(n_processes=1)

    for subset in _data_pipe.subsets:
        _sampler = _data_pipe[subset].sampler
        print(f"{subset.upper()} dataset length: {len(_sampler)}")

        index = 0
        while True:
            if _sampler.is_last_batch:
                break
            for _ds in _sampler.sampling(batch_size=8):
                if _ds is None:
                    break
                print(f"{index}: {_ds.file_path}")
                index += 1
