from importlib import import_module
from importlib.util import find_spec


def use(name):
    """
    Tells spectrabuster which backend to use.
    """

    check_backend(name)

    global _backend
    _backend = name


def get_backend(name=None):
    """
    Retrieves the specified backend, or the backend from use().
    """
    
    if name is None:
        try:
            imported_backend = import_module(f"spectrabuster.backends.{_backend}")
        except NameError:
            raise RuntimeError(
                f"No backend defined. Please call spectrabuster.use to specify a backend."
            )
    elif type(name) is str and check_backend(name):
        imported_backend = import_module(f"spectrabuster.backends.{name}")
        
    return imported_backend


def check_backend(name):
    """
    Checks whether a backend exists and can be imported.
    """

    check = find_spec(f"spectrabuster.backends.{name}")

    if check is None:
        raise NameError(
            f"Couldn't find backend {name} in spectrabuster.backends"
        )

    return True
