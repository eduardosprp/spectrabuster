from importlib import import_module

def use(name):
    """
    Tells spectrabuster which backend to use. For the while you can choose
    either seabreeze for measurements or seatease, for tests without spectrometers
    """
    
    global _backend

    if name in ("seabreeze", "seatease"):
        _backend = name
    else:
        raise ValueError(f"Invalid _backend {name} for spectrabuster")

def get_backend():
    try:
        imported_backend = import_module(f"spectrabuster.backends.{_backend}")
        return imported_backend
    except NameError:
        raise RuntimeError(f"No backend defined. Please call spectrabuster.use to specify a backend.")
