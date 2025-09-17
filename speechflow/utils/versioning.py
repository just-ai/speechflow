__all__ = ["version_check"]


def version_check(module, target_vers: str, strict: bool = False):
    if not hasattr(module, "__version__"):
        raise AttributeError(f"'{module}' object has no attribute '__version__'")

    module_vers = module.__version__

    is_supported = True

    if strict and module_vers != target_vers:
        is_supported = False

    if module_vers.split(".")[0] != target_vers.split(".")[0]:
        is_supported = False

    if not is_supported:
        raise RuntimeError(
            f"version of {module.__name__} module is not match! (current {module_vers}, required {target_vers})"
        )
