import numpy as np
from time import sleep
from math import exp
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from os import path
import struct
import spectrabuster.functions
from importlib import import_module
from datetime import date, datetime


class Spectrum(object):

    """
    Class variables. Mainly default values that are used whenever their
    instance counterparts are not specified when initializing a Spectrum
    object.
    """

    # {{{
    device = None  # will use the first spectrometer seabreeze finds by default
    correct_nl = False  # correct non-linearity
    correct_dc = False  # correct dark counts
    int_time = None  # integration time in microseconds
    _inten = None  # intensities for internal use
    _wavel = None  # wavelengths for internal use
    from_index = None
    to_index = None
    wavel_slice = None
    opt_warnings = True  # determines whether or not to show warnings
    _current_int_time = None  # keeps track of the integration time most recently set in the spectrometer
    calibration_file = None  # the default calibration file with which to obtain the irradiance spectrum
    to_save = {
        "int_time",
        "correct_nl",
        "correct_dc",
        "UV_index",
        "capture_date",
        "capture_time",
        "temp",
    }  # fields that will be written to the file
    samples = 1  # how many samples to average
    optimize = True  # whether or not to call optimize_int_time when sensor saturation is detected
    UV_index = None
    # }}}

    def __init__(
        self,
        int_time=None,
        wavelengths=None,
        device=None,
        from_index=None,
        to_index=None,
        intensities=None,
        backend=None,
        **kwargs,
    ):
        # {{{
        """
        Initializes the instance with the default values, unless specified otherwise.
        If you wish to use all the default values except for a few, specify them with kwargs.
        """

        if device is not None:
            self.device = device

        if wavelengths is not None:
            if isinstance(wavelengths, (np.ndarray, list, tuple)):
                self._wavel = np.array(wavelengths)
            else:
                raise TypeError(
                    "Invalid type for wavelengths array. Please enter a numpy array, a list, or a tuple."
                )
        else:
            # Will use the wavelenghts array provided by the spectrometer by default

            if not self.device:
                """
                If by this point the spectrometer hasn't been specified, it will use the first
                one the backend provides. Aditionally, if the backend hasn't been specified, it'll
                use none.py by default.
                """

                if backend is not None:
                    self.backend = import_module(f"spectrabuster.backends.{backend}")
                else:
                    try:
                        self.backend = spectrabuster.functions.get_backend()
                    except RuntimeError:
                        self.backend = import_module(f"spectrabuster.backends.none")
                        print(self.backend.features["measure"])

                self.device = self.backend.first_available_device()


            self._wavel = np.array(self.device.wavelengths())
            self._warn(
                "No wavelength array provided. Using the device's wavelength array by default."
            )

        if int_time is not None:
            self.int_time = int(int_time)

        if from_index is not None:
            if isinstance(from_index, (int, float)):
                self.from_index = from_index
            else:
                raise TypeError(
                    "to_index and from_index must be either integers for proper indexes, or floats, for wavelengths."
                )

        if to_index is not None:
            if isinstance(to_index, (int, float)):
                self.to_index = to_index
            else:
                raise TypeError(
                    "to_index and from_index must be either integers for proper indexes, or floats, for wavelengths."
                )

        if "correct_nl" in kwargs:
            self.correct_nl = kwargs["correct_nl"]

        if "correct_dc" in kwargs:
            self.correct_dc = kwargs["correct_dc"]

        if "UV_index" in kwargs:
            self.UV_index = kwargs["UV_index"]

        if "samples" in kwargs:
            self.samples = kwargs["samples"]

        if "capture_date" in kwargs:
            self.capture_date = kwargs["capture_date"]
        else:
            self.capture_date = date.today().strftime("%d-%m-%Y")

        if "capture_time" in kwargs:
            self.capture_time = kwargs["capture_time"]
        else:
            self.capture_time = datetime.now().strftime("%H:%M:%S")

        if intensities is not None:
            if isinstance(intensities, (np.ndarray, list, tuple)):
                self._inten = np.array(intensities)
            else:
                raise TypeError(
                    "Invalid type for intensities array. Please enter an interable type."
                )
        else:
            self._inten = self._measure_inten()

    # }}}

    def _measure_inten(self, correct_nl=None, correct_dc=None, samples=None):
        # {{{
        if not self.backend.features["measure"]:
            return []           

        if correct_nl is None:
            correct_nl = self.correct_nl

        if correct_dc is None:
            correct_dc = self.correct_dc

        if not samples:
            samples = self.samples

        if self.int_time and (Spectrum._current_int_time != self.int_time):
            self.device.set_int_time(self.int_time)
            sleep(
                10
            )  # accounting for the delay of the spectrometer to change its integration time
            Spectrum._current_int_time = self.int_time

        if type(self.samples) is not int or self.samples < 0:
            raise ValueError(
                f"Invalid value of {self.samples} for the number of samples to average."
            )

        inten_avg = self.device.measure(correct_dc=correct_dc, correct_nl=correct_nl)
        for i in range(samples - 1):
            inten_avg += self.device.measure(correct_dc=self.correct_dc, correct_nl=self.correct_nl)
        inten_avg /= samples

        return inten_avg

    # }}}

    def write_to_file(self, file_path=None, save_fields=True, **kwargs):
        # {{{
        """
        Stores spectrum in a .dat text file, using a format that is easy to parse
        in gnu octave, R or any other programming language, or visualize in gnuplot,
        or any spreadsheet program.
        """

        overwrite = kwargs["overwrite"] if "overwrite" in kwargs else None
        if path.exists(file_path):
            if overwrite:
                self._warn(f"WARNING: File {file_path} exists. Overwriting it.")
            else:
                raise RuntimeError(
                    f"File {file_path} already exists. Pass 'overwrite=True' if you are sure you want to overwrite it."
                )

        # set this kwarg to True if you wish to store the entire wavelengths and intensities array,
        # as opposed to just the array delimited by from_index and to_index
        entire_spectrum = (
            kwargs["entire_spectrum"] if "entire_spectrum" in kwargs else False
        )
        only_wavelengths = (
            kwargs["only_wavelengths"] if "only_wavelengths" in kwargs else False
        )
        only_intensities = (
            kwargs["only_intensities"] if "only_intensities" in kwargs else False
        )

        to_save = self.to_save  # fields that will be written to the file
        if (
            entire_spectrum
        ):  # will only write these fields if entire_spectrum is set to True
            to_save = to_save.union({"from_index", "to_index"})

        if not file_path or not isinstance(file_path, str):
            raise (
                ValueError(
                    "Please pass a string as the file path wherein to save the spectrum."
                )
            )

        with open(file_path, "w+") as arq:
            gen_comments = (
                f"# {name} = {value}\n"
                for name, value in vars(self).items()
                if name in to_save
            )
            arq.writelines(gen_comments)

            if only_wavelengths:
                if entire_spectrum:
                    gen_wavel_inten = (f"{wavel}\n" for wavel in self._wavel)
                else:
                    gen_wavel_inten = (f"{wavel}\n" for wavel in self.wavelengths)
            elif only_intensities:
                if entire_spectrum:
                    gen_wavel_inten = (f"{inten}\n" for inten in self._inten)
                else:
                    gen_wavel_inten = (f"{inten}\n" for inten in self.intensities)
            else:
                if entire_spectrum:
                    gen_wavel_inten = (
                        f"{wavel}\t{inten}\n" for wavel, inten in zip(*self._spec)
                    )
                else:
                    gen_wavel_inten = (
                        f"{wavel}\t{inten}\n" for wavel, inten in zip(*self.spectrum)
                    )

            arq.writelines(gen_wavel_inten)

    # }}}

    def set_wavel_slice(self):
        # {{{
        from_index = self.from_index if self.from_index else None
        to_index = self.to_index if self.to_index else None

        if type(from_index) in (
            float,
            np.float64,
        ):  # assumes the value is a wavelength if it has a decimal point
            from_index = self.find_wavel_index(self._wavel, from_index)
        elif (
            type(from_index) == int
        ):  # assumes the value is a proper index if it is an integer
            if abs(from_index) > self._wavel.size:
                raise IndexError(
                    "Invalid index of {} for wavelength array of size {}".format(
                        from_index, self._wavel.size
                    )
                )
        elif type(from_index) == str:
            try:
                float(from_index)
            except ValueError:
                raise TypeError(
                    "Invalid type of {} for wavelength index. Please enter either a float for a wavelength or an integer for a proper index.".format(
                        from_index
                    )
                )
            if "." in from_index:
                from_index = self.find_wavel_index(self._wavel, float(from_index))
            else:
                from_index = int(from_index)

        if type(to_index) in (
            float,
            np.float64,
        ):  # assumes the value is a wavelength if it has a decimal point
            to_index = self.find_wavel_index(self._wavel, to_index) + 1
        elif (
            type(to_index) == int
        ):  # assumes the value is a proper index if it is an integer
            if abs(to_index) > self._wavel.size:
                raise IndexError(
                    "Invalid index of {} for wavelength array of size {}".format(
                        from_index, self._wavel.size
                    )
                )
        elif type(to_index) == str:
            try:
                float(to_index)
            except ValueError:
                raise TypeError(
                    "Invalid type of {} for wavelength index. Please enter either a float for a wavelength or an integer for a proper index.".format(
                        to_index
                    )
                )
            if "." in to_index:
                to_index = self.find_wavel_index(self._wavel, float(to_index)) + 1
            else:
                to_index = int(to_index)

        self.wavel_slice = slice(from_index, to_index)

    # }}}

    def get_wavel_slice(self):
        if self.wavel_slice:  # {{{
            pass
        else:
            self.set_wavel_slice()
        return self.wavel_slice  # }}}

    def to_spectral_irrad(self, calibration_file=None, int_time=None):
        # {{{
        """
        Applies the spectral irradiance calibration and returns another
        Spectrum object for the irradiance spectrum.

        It also has to be a file with the wavelengths and spectral sensitivity,
        by the way.
        """

        if not calibration_file:
            raise RuntimeError(
                "Please pass the path to the calibration file as an argument."
            )

        if not int_time and not self.int_time:
            raise ValueError(
                "No integration time argument passed, and this spectrum's int_time field is empty."
            )
        elif not int_time and self.int_time:
            int_time = self.int_time

        calib_wavelengths, calib_intensities, _ = self._read_file(calibration_file)

        if self._wavel.size > calib_wavelengths.size:
            from_index = self.find_wavel_index(self._wavel, calib_wavelengths[0])
            to_index = self.find_wavel_index(self._wavel, calib_wavelengths[-1])

            wavel_array = self._wavel[from_index : to_index + 1]
            inten_array = self._inten[from_index : to_index + 1]

        elif calib_wavelengths.size > self._wavel.size:
            from_index = self.find_wavel_index(calib_wavelengths, self._wavel[0])
            to_index = self.find_wavel_index(calib_wavelengths, self._wavel[-1])

            calib_intensities = calib_intensities[from_index : to_index + 1]
            wavel_array = calib_wavelengths[from_index : to_index + 1]
            inten_array = self._inten

        else:
            inten_array = self._inten
            wavel_array = self._wavel

        apply_calibration = lambda counts, calib: counts / (int_time * calib * 0.000001)

        inten_array = apply_calibration(inten_array, calib_intensities)
        return Spectrum(
            intensities=inten_array,
            wavelengths=wavel_array,
            int_time=int_time,
            from_index=self.from_index,
            to_index=self.to_index,
        )

        self_params = vars(self).copy()
        self_params.update({"intensities": inten_array, "wavelengths": wavel_array})

        return Spectrum(**self_params)

    # }}}

    def to_count_rate(self):
        # {{{
        """
        Divides the spectrum by its integration time and that's it.
        """

        if self.int_time:
            return self / (self.int_time * 0.000001)
        else:
            raise ValueError(
                "Integration time undefined for calculation of count rate."
            )

    # }}}

    def calc_uv_index(self, from_wavel=286.0):
        # {{{
        """
        Calculates the UV index based on Mckinley-Diffey's action spectra for erythema.
        """

        weighted_irrad = np.array(
            [
                self.weight_irrad(wavel, irrad, from_wavel)
                for wavel, irrad in zip(*self.spectrum)
            ]
        )
        self.UV_index = round(0.04 * trapz(weighted_irrad, self.wavelengths), 2)

        return self.UV_index  # just for convenience

    # }}}

    def optimize_int_time(self, initial=None, limits=(0.8, 1), max_tries=5):
        # {{{
        """
        Attemps to find an integration time that maximizes signal to noise
        ratio while avoiding sensor saturation.

        This could probably be done more elegantly with recursion, but I
        haven't got time to think about that. Also BEWARE that this will
        overwrite the current spectrum.
        """

        if initial is None:
            initial = self.int_time

        min_int_time, max_int_time = self.device.int_time_limits()

        max_counts = self.device.max_intensity
        target_counts = abs((limits[1] + limits[0]) / 2) * max_counts

        int_time = initial
        self.int_time = int_time

        print("Optimizing integration time...")
        i = 0
        while i < max_tries or inten_max == max_counts:
            self._inten = self._measure_inten(
                correct_nl=False, correct_dc=False, samples=1
            )
            inten_max = np.amax(self.intensities)
            ratio = inten_max / target_counts
            print(f"{i} {self.int_time} {ratio} {inten_max}")

            if limits[0] <= ratio <= limits[1]:
                break
            elif limits[1] <= ratio <= 1:
                int_time *= ratio ** 2
            elif ratio > 1:
                int_time /= ratio ** 2
            else:
                int_time /= ratio

            while int_time < min_int_time or int_time > max_int_time:
                int_time /= 2

            self.int_time = int_time

            i += 1

        self._inten = self._measure_inten()

        return int_time  # just for convenience

    # }}}

    def join(self, other):
        # {{{
        """
        Joins two spectra. It will give preference to itself when resolving
        overlaps. Probably one of the first functions to get a rewrite.
        """

        if not isinstance(other, Spectrum):
            raise TypeError("join takes only spectra as arguments")

        self_wavel_max = self.wavelengths[-1]
        self_wavel_min = self.wavelengths[0]
        other_wavel_max = other.wavelengths[-1]
        other_wavel_min = other.wavelengths[0]

        # other.wavelengths starts before self.wavelengths ends
        if np.isclose(other.wavelengths, self_wavel_max).any():

            # NOTE: These variables are indexes referring to self.wavelengths and
            # other.wavelengths respectively!
            start_overlap = np.argmax(np.isclose(self.wavelengths, other_wavel_min))
            end_overlap = np.argmax(np.isclose(other.wavelengths, self_wavel_max))

            Spectrum._warn(
                f"WARNING: The spectra overlap from {other_wavel_min} to {self_wavel_max}"
            )

            # For some god forsaken reason np.concatenate will only work if you pass
            # a tuple of arrays...
            new_wavels = np.copy(self.wavelengths)
            new_wavels = np.concatenate(
                (new_wavels, np.copy(other.wavelengths[end_overlap + 1 :]))
            )

            new_intens = np.copy(self.intensities)
            new_intens = np.concatenate(
                (new_intens, np.copy(other.intensities[end_overlap + 1 :]))
            )

        # self.wavelengths starts before other.wavelengths ends
        elif np.isclose(self.wavelengths, other_wavel_max).any():

            # NOTE: These variables are indexes referring to other.wavelengths and
            # self.wavelengths respectively!
            start_overlap = np.argmax(np.isclose(other.wavelengths, self_wavel_min))
            end_overlap = np.argmax(np.isclose(self.wavelengths, other_wavel_max))

            Spectrum._warn(
                f"WARNING: The spectra overlap from {self_wavel_min} to {other_wavel_max}"
            )

            # You see, the preference is always given to self
            new_wavels = np.copy(other.wavelengths[:start_overlap])
            new_wavels = np.concatenate((new_wavels, np.copy(self.wavelengths)))

            new_intens = np.copy(other.intensities[:start_overlap])
            new_intens = np.concatenate((new_intens, np.copy(self.intensities)))

        else:

            if other_wavel_min > self_wavel_min:
                new_wavels = np.concatenate(
                    (np.copy(self.wavelengths), np.copy(other.wavelengths))
                )

                new_intens = np.concatenate(
                    (np.copy(self.intensities), np.copy(other.intensities))
                )

            elif other_wavel_min < self_wavel_min:
                new_wavels = np.concatenate(
                    (np.copy(other.wavelengths), np.copy(self.wavelengths))
                )

                new_intens = np.concatenate(
                    (np.copy(other.intensities), np.copy(self.intensities))
                )

        self_params = vars(self).copy()
        self_params.update(
            {
                "intensities": new_intens,
                "wavelengths": new_wavels,
                "from_index": None,
                "to_index": None,
            }
        )

        return Spectrum(**self_params)

    # }}}

    @property
    def max_counts(self):
        return self.device.max_intensity

    @property
    def biggest_count(self):
        # puta que pariu esses nomes...
        return np.amax(self.intensities)

    @property
    def uv(self):
        # {{{
        # Lmao...
        if self.UV_index:
            return self.UV_index
        else:
            return self.calc_uv_index()

    # }}}

    @property
    def _spec(self):
        return self._wavel, self._inten

    @property
    def spectrum(self):
        return self.wavelengths, self.intensities

    @property
    def wavelengths(self):
        return self._wavel[self.get_wavel_slice()]

    @property
    def intensities(self):
        return self._inten[self.get_wavel_slice()]

    @classmethod
    def from_file(cls, inten_wavel_file=None, **kwargs):
        # {{{
        """
        Creates a spectrum instance with the wavelengths and/or intensities
        read from a text file. Additionally, it looks for key-word arguments at
        the first few lines of the file. If the same kwargs are passed to this
        function, they take precedence.
        """

        wavel_file = kwargs["wavel_file"] if "wavel_file" in kwargs else None
        inten_file = kwargs["inten_file"] if "inten_file" in kwargs else None
        inten_wavel_file = (
            kwargs["inten_wavel_file"]
            if "inten_wavel_file" in kwargs
            else inten_wavel_file
        )
        inten_array = None
        wavel_array = None
        new_kwargs = {}

        if inten_wavel_file:
            wavel_array, inten_array, new_kwargs = cls._read_file(inten_wavel_file)

        if wavel_file:
            wavel_array, _, new_kwargs = cls._read_file(wavel_file)

        if inten_file:
            inten_array, _, new_kwargs = cls._read_file(inten_file)

        if not inten_file and not inten_wavel_file and not wavel_file:
            cls._warn(
                "WARNING: Instantiating a spectrum with function from_file, but no file path arguments were passed."
            )

        new_kwargs["intensities"] = inten_array
        new_kwargs["wavelengths"] = wavel_array
        new_kwargs.update(kwargs)

        return cls(**new_kwargs)

    # }}}

    @staticmethod
    def _read_file(text_file):
        # {{{
        """
        Used internally by the class method from_file. Returns as many numpy arrays
        as there are columns in the file, and a dictionary with whatever comments
        (prefaced by #) it finds.
        """

        dict_args = {}
        col1 = []
        col2 = []

        with open(text_file, "r") as arq:

            # Generator for the lines in the archive
            gen_lines = (line.split() for line in arq)

            for line_split in gen_lines:

                if (
                    line_split[0] == "#"
                ):  # comment, presumably containing arguments for __init__
                    dict_args[line_split[1]] = line_split[3]
                elif (
                    len(line_split) > 1
                ):  # 2 or more columns. Will ignore anything after the second column
                    col1.append(float(line_split[0]))
                    col2.append(float(line_split[1]))
                elif len(line_split) == 1:  # 1 column
                    col1.append(float(line_split[0]))

            if not dict_args and not col1 and not col2:
                # Check if they're all empty

                raise RuntimeError(
                    f"No arguments, wavelengths and intensities found in {text_file}. Please check if this is a valid file."
                )

            return np.array(col1), np.array(col2), dict_args

    # }}}

    @staticmethod
    def weight_irrad(wavel, irrad, from_wavel=286.0):
        # {{{
        """
        Simple implementation of Mckinley-Diffey's action spectrum.
        """

        if from_wavel <= wavel < 298:
            return irrad
        elif 298 <= wavel < 328:
            return exp(0.216 * (298 - wavel)) * irrad
        elif 328 <= wavel < 400:
            return exp(0.034 * (139 - wavel)) * irrad
        else:
            return 0

    # }}}

    @staticmethod
    def find_wavel_index(wavel_array, wavel, margin=0.5):
        # {{{
        """
        Attempts to find 'wavel' in 'wavel_array'. Will try using the closest wavelength
        at most 0.5 units from 'wavel'
        """

        array_diffs = np.abs(wavel_array - wavel)
        closest_index = array_diffs.argmin()

        if np.isclose(wavel_array[closest_index], wavel):
            return closest_index

        elif array_diffs[closest_index] < 0.5:
            Spectrum._warn(
                f"Exact match for {wavel} not found. Using {wavel_array[closest_index]} instead."
            )
            return closest_index

        else:
            raise ValueError(
                f"A close enough {wavel} wasn't found. Closest value is {wavel_array[closest_index]}."
            )

    # }}}

    @staticmethod
    def _warn(string):
        # {{{
        """
        Warnings can be disabled by setting the class variable 'opt_warnings' to False
        """

        if Spectrum.opt_warnings:
            print(string)

    # }}}

    # Magic methods start here

    def __iter__(self):
        return zip(*self.spectrum)

    def __add__(self, other):
        # {{{
        """
        Adds the first spectrum's intensities with the second's.

        This operation will always return another spectrum with the added intensities.
        """

        if isinstance(other, Spectrum):

            if np.isclose(self.wavelengths, other.wavelengths).all():
                new_inten = self.intensities + other.intensities
            else:
                raise ValueError(
                    "The divided spectrums must have the same wavelengths array."
                )

        elif isinstance(other, (np.ndarray, list)):

            if len(other) == self.wavelengths.size or len(other) == 1:
                new_inten = self.intensities + other

            else:
                raise (
                    ValueError(
                        "The other operand must have the same size as the spectrum's wavelengths array, or size 1."
                    )
                )

        elif isinstance(other, (float, int)):

            new_inten = self.intensities + other

        else:
            raise (TypeError("Incompatible types for addition."))

        self_params = vars(self).copy()
        self_params.update(
            {
                "intensities": new_inten,
                "wavelengths": self.wavelengths,
                "from_index": None,
                "to_index": None,
            }
        )

        return Spectrum(**self_params)

    # }}}

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        # {{{
        if isinstance(other, Spectrum):
            if np.isclose(self.wavelengths, other.wavelengths).all():
                return self + np.negative(other.intensities)
            else:
                raise ValueError(
                    "The subtracted spectrums must have the same wavelengths array."
                )
        else:
            return self + np.negative(other)

    # }}}

    def __rsub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        # {{{
        """
        Multiplies the first spectrum's intensities by the second.

        This operation will always return another spectrum with the multiplied intensities.
        """

        if isinstance(other, Spectrum):

            if np.isclose(self.wavelengths, other.wavelengths).all():
                new_inten = self.intensities * other.intensities
            else:
                raise ValueError(
                    "The divided spectrums must have the same wavelengths array."
                )

        elif isinstance(other, (np.ndarray, list)):

            if len(other) == self.wavelengths.size or len(other) == 1:
                new_inten = self.intensities * other

            else:
                raise (
                    ValueError(
                        "The other operand must have the same size as the spectrum's wavelengths array, or size 1."
                    )
                )

        elif isinstance(other, (float, int)):

            new_inten = self.intensities * other

        else:
            raise (TypeError("Incompatible types for multiplication."))

        self_params = vars(self).copy()
        self_params.update(
            {
                "intensities": new_inten,
                "wavelengths": self.wavelengths,
                "from_index": None,
                "to_index": None,
            }
        )

        return Spectrum(**self_params)

    # }}}

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        # {{{
        """
        Divides the first spectrum's intensities by the second. I makes no checks whether
        division by zero is being requested, that I leave to numpy.

        This operation will always return another spectrum with the divided intensities.
        The new spectrum's fields will be inherited from the first operand.
        """

        if isinstance(other, Spectrum):

            if np.isclose(self.wavelengths, other.wavelengths).all():
                new_inten = self.intensities / other.intensities
            else:
                raise ValueError(
                    "The divided spectrums must have the same wavelengths array."
                )

        elif isinstance(other, (np.ndarray, list)):

            if len(other) == self.wavelengths.size or len(other) == 1:
                new_inten = self.intensities / other

            else:
                raise (
                    ValueError(
                        "The other operand must have the same size as the spectrum's wavelengths array, or size 1."
                    )
                )

        elif isinstance(other, (float, int)):

            new_inten = self.intensities / other

        else:
            raise (TypeError("Incompatible types for division."))

        self_params = vars(self).copy()
        self_params.update(
            {
                "intensities": new_inten,
                "wavelengths": self.wavelengths,
                "from_index": None,
                "to_index": None,
            }
        )

        return Spectrum(**self_params)

    # }}}

    def __rdiv__(self, other):
        raise NotImplementedError

    def __getitem__(self, key):
        # {{{
        """
        Takes the key to be a proper index if it is an integer, and as a wavelength
        if it is float. It also accepts numpy slices and regular slices, of course.
        """

        if isinstance(key, (int, list, np.ndarray)):
            return self.intensities[key]
        elif isinstance(key, float):
            int_index = self.find_wavel_index(self.wavelengths, key)
            return self.intensities[int_index]
        else:
            raise TypeError(
                "Invalid type for index. Please enter an integer, list, numpy array or a float."
            )

    # }}}

    def __setitem__(self, key, val):
        # {{{
        """
        Changes the intensity with index 'key' to 'val'. The new value must be a number,
        a tuple, list or numpy array. In the latter 3 cases, numpy will handle the assignment.
        """

        if isinstance(key, (list, tuple, np.ndarray)):
            # Melhorar isto. Adicionar gerenciamento de exceções
            key = [
                self.find_wavel_index(self.wavelengths, x)
                if isinstance(x, float)
                else x
                for x in key
            ]
        elif isinstance(key, float):
            key = self.find_wavel_index(self.wavelengths, key)
        elif isinstance(key, int):
            if abs(key) > self.wavelengths.size:
                raise IndexError(
                    f"Invalid index of {val} for wavelengths array of size {self.wavelengths.size}"
                )
        else:
            raise TypeError(
                "Invalid type for index. Please enter an integer, list, numpy array or a float."
            )

        if isinstance(val, (tuple, list, np.ndarray)):
            try:
                val = [float(x) for x in val]
            except (TypeError, ValueError) as exception:
                raise ValueError(f"The {type(val)} {val} must contain only numbers.")
        else:
            try:
                val = float(val)
            except:
                raise ValueError(
                    f"Invalid value of {val} for intensity. Please enter something convertible to float."
                )

        self.intensities[key] = val

    # }}}

    def __contains__(self, value):
        raise NotImplementedError

    def __repr__(self):
        return "Spectrum({}, {})".format(self.wavelengths, self.intensities)

    def __len__(self):
        return self.wavelengths.size

