"""
This will be used by default if pyseabreeze isn't installed. Use this if you
don't need to measure any spectrums. Also useful as a minimal template for
implementing your own backends.
"""

"""
Every backend must contain a dictionary listing of its features. Spectrabuster
looks for the following features in this dictionary, a False value or the
absence of a key is interpreted as absence of that feature.
"""
features = {
    "correct_nl": False,  # Correction of non-linearity
    "correct_dc": False,  # Correction of dark counts
    "temperature": False,  # Measurement of the device's temperature
}


class Device(object):
    """
    All of the following methods are required of the Device class.
    """

    def measure(self, **kwargs):
        return None

    def wavelengths(self, **kwargs):
        return None

    def set_int_time(self, int_time, **kwargs):
        return None


"""
All of the following functions are required for the backend to work.
"""

def devices():
   return [Device()]

def first_available_device():
   return Device()
