import typing as tp

__all__ = [
    "Metadata",
    "SingleMetadataTransform",
    "MultiMetadataTransform",
    "MetadataTransform",
]


Metadata = tp.Dict
SingleMetadataTransform = tp.Callable[[Metadata], tp.List[Metadata]]
MultiMetadataTransform = tp.Callable[[tp.List[Metadata]], tp.List[Metadata]]
MetadataTransform = tp.Union[SingleMetadataTransform, MultiMetadataTransform]
