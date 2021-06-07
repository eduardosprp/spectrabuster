import seabreeze.use
from seabreeze.spectrometers import Spectrometer, list_devices
from time import sleep

seabreeze.use("pyseabreeze")

features = {
    "measure": True
    "correct_nl": True,
    "correct_dc": True,
    "temperature": True,
    "int_time_limits": True,
    "max_intensity": True,
}


class Device(object):
    # {{{
    def __init__(self, device_obj=None, **kwargs):

        # Yes, I am aware that having a self._device inside the
        # Device class can be somewhat confusing
        if device_obj is None:
            self._device = Spectrometer.from_first_available()
        else:
            self._device = Spectrometer(device_obj)

    def measure(self, **kwargs):

        correct_nl = kwargs["correct_nl"] if "correct_nl" in kwargs else False
        correct_dc = kwargs["correct_dc"] if "correct_dc" in kwargs else False

        return self._device.intensities(correct_dc, correct_nl)

    def wavelengths(self, **kwargs):
        return self._device.wavelengths()

    def set_int_time(self, int_time, sleep_time=0, **kwargs):
        self._device.integration_time_micros(int_time)

        # This is to account for the delay involved in changing
        # the spectrometer's integration time
        sleep(sleep_time)

    @property
    def int_time_limits(self):
        return self._device.integration_time_micros_limits

    @property
    def max_intensity(self):
        return self._device.max_intensity

    @property
    def temperature(self):
        return 0  # Implement this
# }}}


def devices():
    return list_devices()


def first_available_device():
    return Device()
