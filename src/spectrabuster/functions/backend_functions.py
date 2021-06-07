from importlib import import_module


def use(name):
    """
    Tells spectrabuster which backend to use.
    """

    global _backend

    _backend = name


def get_backend():
    try:
        imported_backend = import_module(f"spectrabuster.backends.{_backend}")
        return imported_backend
    except NameError:
        raise RuntimeError(
            f"No backend defined. Please call spectrabuster.use to specify a backend."
        )
