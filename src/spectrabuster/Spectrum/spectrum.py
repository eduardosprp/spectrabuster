import numpy as np
from time import sleep
from math import exp
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from os import path
import struct
import spectrabuster.functions as sbf
from importlib import import_module
from datetime import date, datetime
from functools import partial


class Spectrum(object):

    """
    Class variables. Mainly default values that are used whenever their
    instance counterparts are not specified when initializing a Spectrum
    object.
    """

    # {{{
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

    def __init__(self, int_time=None, **kwargs):
        # {{{
        """
        Initializes the Spectrum with the values specified by kwargs. Absence
        of a key results in the attribute taking a default value.
        """

        # First of all, define a backend
        backend = kwargs["backend"] if "backend" in kwargs else None
        try:
            self.backend = sbf.get_backend(backend)
        except RuntimeError:
            self._warn("No backend specified. Using none by default.")
            self.backend = sbf.get_backend("none")

        # Then get the device
        if "device" in kwargs:
            self.device = self.backend.Device(kwargs["device"])
        else:
            self.device = self.backend.first_available_device()

        # Some features of the device
        if self.backend.features["int_time_limits"]:
            self.int_time_limits = self.device.int_time_limits
        else:
            self.int_time_limits = None
        if self.backend.features["sat_intensity"]:
            self.sat_intensity = self.device.sat_intensity
        else:
            self.sat_intensity = None

        # Bunch of optional parameters
        self.from_index = kwargs["from_index"] if "from_index" in kwargs else None
        self.to_index = kwargs["to_index"] if "to_index" in kwargs else None
        self.correct_nl = kwargs["correct_nl"] if "correct_nl" in kwargs else False
        self.correct_dc = kwargs["correct_dc"] if "correct_dc" in kwargs else False
        self.samples = kwargs["samples"] if "samples" in kwargs else None
        self.int_time = int_time

        # Then the wavelengths and the intensities. It'll get each from the
        # device unless provided at the instantiation.
        if "wavelengths" in kwargs:
            self.wavel = np.array(kwargs["wavelengths"])
        elif self.backend.features["measure"]:
            self.wavel = self.device.wavelengths()
        else:
            raise RuntimeError(
                f"No wavelengths array passed, and the {self.backend} cannot make measurements."
            )

        if "intensities" in kwargs:
            self.inten = np.array(kwargs["intensities"])
        elif self.backend.features["measure"]:
            self.inten = self.measure_inten(int_time=self.int_time)
        else:
            raise RuntimeError(
                f"No intensities array passed, and the {self.backend} cannot make masurements."
            )

        # Finally, slice the intensities and wavelengths arrays.
        self.wavel, self.inten, self.slice = self.slice_array(
            self.wavel, self.inten, (self.from_index, self.to_index)
        )

    # }}}

    def measure_inten(self, int_time=None, samples=None, **kwargs):
        # {{{
        samples = samples if samples is not None else self.samples
        int_time = int_time if int_time is not None else self.int_time
        correct_dc = kwargs["correct_dc"] if "correct_dc" in kwargs else self.correct_dc
        correct_nl = kwargs["correct_nl"] if "correct_nl" in kwargs else self.correct_nl

        if samples is None:
            samples = 1
        elif type(samples) is not int or samples < 0:
            raise ValueError(
                f"Invalid value of {self.samples} for the number of samples to average."
            )

        if int_time is None:
            pass
        else:
            try:
                self.device.set_int_time(float(int_time))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid type or value of {int_time} for integration time"
                )

        measure = partial(
            self.device.measure, correct_dc=correct_dc, correct_nl=correct_nl
        )
        inten_avg = np.average([measure() for i in range(samples)], axis=0)

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

        only_wavelengths = (
            kwargs["only_wavelengths"] if "only_wavelengths" in kwargs else False
        )
        only_intensities = (
            kwargs["only_intensities"] if "only_intensities" in kwargs else False
        )

        to_save = self.to_save  # fields that will be written to the file

        if not file_path or not isinstance(file_path, str):
            raise ValueError(
                "Please pass a string as the file path wherein to save the spectrum."
            )

        with open(file_path, "w+") as arq:
            gen_comments = (
                f"# {name} = {value}\n"
                for name, value in vars(self).items()
                if name in to_save
            )
            arq.writelines(gen_comments)

            if only_wavelengths:
                gen_wavel_inten = (f"{wavel}\n" for wavel in self.wavel)
            elif only_intensities:
                gen_wavel_inten = (f"{inten}\n" for inten in self.inten)
            else:
                gen_wavel_inten = (
                    f"{wavel}\t{inten}\n" for wavel, inten in zip(*self.spectrum)
                )

            arq.writelines(gen_wavel_inten)

    # }}}

    def to_spectral_irrad(self, calibration_file=None, int_time=None):
        # {{{
        """
        Applies the spectral irradiance calibration and returns another
        Spectrum object for the irradiance spectrum.

        It also has to be a file with the wavelengths and spectral sensitivity,
        by the way. And note that this function assumes that the current
        Spectrum and the calibration file have the same wavelengths array,
        maybe just sliced differently
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

        calib_wavel, calib_inten, _ = sbf.read_file(calibration_file)

        if self.wavel.size > calib_wavel.size:
            from_index = self.find_wavel_index(self.wavel, calib_wavel[0])
            to_index = self.find_wavel_index(self.wavel, calib_wavel[-1])

            wavel_array = self.wavel[from_index : to_index + 1]
            inten_array = self.inten[from_index : to_index + 1]

        elif calib_wavel.size > self.wavel.size:
            from_index = self.find_wavel_index(calib_wavel, self.wavel[0])
            to_index = self.find_wavel_index(calib_wavel, self.wavel[-1])

            calib_inten = calib_inten[from_index : to_index + 1]
            wavel_array = calib_wavel[from_index : to_index + 1]
            inten_array = self.inten

        else:
            inten_array = self.inten
            wavel_array = self.wavel

        apply_calibration = lambda counts, calib: counts / (int_time * calib * 0.000001)

        inten_array = apply_calibration(inten_array, calib_inten)
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

    def calc_uv_index(self, from_wavel=286.0, to_wavel=400.0):
        self.UV_index = sbf.calc_uv_index(self.spectrum, from_wavel, to_wavel)
        return self.UV_index

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

        min_int_time, max_int_time = self.device.int_time_limits

        max_counts = self.device.sat_intensity
        target_counts = abs((limits[1] + limits[0]) / 2) * max_counts

        int_time = initial
        self.int_time = int_time

        print("Optimizing integration time...")
        i = 0
        while i < max_tries or inten_max == max_counts:
            self.inten = self.measure_inten(
                correct_nl=False, correct_dc=False, samples=1
            )
            inten_max = np.amax(self.inten)
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

        self.inten = self.measure_inten()[self.slice]

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

        self_wavel_max = self.wavel[-1]
        self_wavel_min = self.wavel[0]
        other_wavel_max = other.wavel[-1]
        other_wavel_min = other.wavel[0]

        # other.wavel starts before self.wavel ends
        if np.isclose(other.wavel, self_wavel_max).any():

            # NOTE: These variables are indexes referring to self.wavelengths and
            # other.wavelengths respectively!
            start_overlap = np.argmax(np.isclose(self.wavel, other_wavel_min))
            end_overlap = np.argmax(np.isclose(other.wavel, self_wavel_max))

            Spectrum._warn(
                f"WARNING: The spectra overlap from {other_wavel_min} to {self_wavel_max}"
            )

            # For some god forsaken reason np.concatenate will only work if you pass
            # a tuple of arrays...
            new_wavels = np.copy(self.wavel)
            new_wavels = np.concatenate(
                (new_wavels, np.copy(other.wavel[end_overlap + 1 :]))
            )

            new_intens = np.copy(self.inten)
            new_intens = np.concatenate(
                (new_intens, np.copy(other.inten[end_overlap + 1 :]))
            )

        # self.wavelengths starts before other.wavelengths ends
        elif np.isclose(self.wavel, other_wavel_max).any():

            # NOTE: These variables are indexes referring to other.wavel and
            # self.wavel respectively!
            start_overlap = np.argmax(np.isclose(other.wavel, self_wavel_min))
            end_overlap = np.argmax(np.isclose(self.wavel, other_wavel_max))

            Spectrum._warn(
                f"WARNING: The spectra overlap from {self_wavel_min} to {other_wavel_max}"
            )

            # You see, the preference is always given to self
            new_wavels = np.copy(other.wavel[:start_overlap])
            new_wavels = np.concatenate((new_wavels, np.copy(self.wavel)))

            new_intens = np.copy(other.inten[:start_overlap])
            new_intens = np.concatenate((new_intens, np.copy(self.inten)))

        # There is no overlap
        else:

            if other_wavel_min > self_wavel_min:

                new_wavels = np.concatenate(
                    (np.copy(self.wavel), np.copy(other.wavel))
                )

                new_intens = np.concatenate(
                    (np.copy(self.inten), np.copy(other.inten))
                )

            elif other_wavel_min < self_wavel_min:

                new_wavels = np.concatenate(
                    (np.copy(other.wavel), np.copy(self.wavel))
                )

                new_intens = np.concatenate(
                    (np.copy(other.inten), np.copy(self.inten))
                )

        self_params = vars(self).copy()
        self_params.update(
            {
                "intensities": new_intens,
                "wavelengths": new_wavels,
                "from_index": None,
                "to_index": None,
                "backend": "none",
            }
        )

        return Spectrum(**self_params)

    # }}}

    @property
    def max_counts(self):
        return np.amax(self.inten)

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
    def spectrum(self):
        return self.wavel, self.inten

    @property
    def wavelengths(self):
        return self.wavel

    @property
    def intensities(self):
        return self.inten

    @classmethod
    def from_file(cls, inten_wavel_file=None, **kwargs):
        # {{{
        """
        Creates a spectrum instance with the wavelengths and/or intensities
        read from a text file. Additionally, it looks for key-word arguments at
        the first few lines of the file. If the same kwargs are passed to this
        function, they take precedence.

        When retrieving a Spectrum from a file, it will always be assigned the
        backend 'none'.
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
            wavel_array, inten_array, new_kwargs = sbf.read_file(inten_wavel_file)

        if wavel_file:
            wavel_array, _, new_kwargs = sbf.read_file(wavel_file)

        if inten_file:
            inten_array, _, new_kwargs = sbf.read_file(inten_file)

        if not inten_file and not inten_wavel_file and not wavel_file:
            cls._warn(
                "WARNING: Instantiating a spectrum with function from_file, but no file path arguments were passed."
            )

        new_kwargs["intensities"] = inten_array
        new_kwargs["wavelengths"] = wavel_array
        # The backend 'none' will always be used when loading a
        # Spectrum from a file
        new_kwargs["backend"] = "none"
        new_kwargs.update(kwargs)

        return cls(**new_kwargs)

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
    def slice_array(wavel, inten, indices, **kwargs):
        # {{{
        """
        Takes in two arrays and returns them sliced according to
        indices=(from_index, to_index).

        If the indeces are integers, it takes them to be literal indeces for the
        array. If they are floats, then it'll assume they are wavelengths whose
        literal indeces must be found before slicing.

        This behaviour can be overriden by passing literal_indices=True or False
        """

        literal = kwargs["literal"] if "literal" in kwargs else None
        len_array = len(wavel)

        if len(inten) != len_array:
            raise ValueError("The arrays must be of equal length.")

        new_indices = []

        for index in indices:
            if index is None:
                new_indices.append(index)
            elif type(index) is int or literal is True:
                if not (0 <= abs(index) <= len_array):
                    raise IndexError(
                        f"Invalid index of {index} for array of size {len_array}."
                    )
                else:
                    new_indices.append(index)
            elif type(index) in (float, np.float64) or literal is False:
                index_wavel = Spectrum.find_wavel_index(wavel, index)
                new_indices.append(index_wavel)

        array_slice = slice(new_indices[0], new_indices[1])
        return wavel[array_slice], inten[array_slice], array_slice

    # }}}

    @staticmethod
    def _warn(string):
        # {{{
        """
        Warnings can be disabled by setting the class variable 'opt_warnings' to False
        """

        print(string)

    # }}}

    # Magic methods start here

    def __iter__(self):
        return zip(*self.spectrum)

    def __add__(self, other):
        # {{{
        """
        Adds the first spectrum's intensities with the second's. It can add spectrums
        with numpy arrays and lists as well, as long as they are the same length as the
        spectrum's wavelengths array.

        This operation will always return another spectrum with the added intensities.
        """

        if isinstance(other, Spectrum):

            if np.isclose(self.wavel, other.wavel).all():
                new_inten = self.inten + other.inten
            else:
                raise ValueError(
                    "The divided spectrums must have the same wavelengths array."
                )

        elif isinstance(other, (np.ndarray, list)):

            if len(other) == self.wavel.size or len(other) == 1:
                new_inten = self.inten + other

            else:
                raise (
                    ValueError(
                        "The other operand must have the same size as the spectrum's wavelengths array, or size 1."
                    )
                )

        elif isinstance(other, (float, int)):

            new_inten = self.inten + other

        else:
            raise (TypeError("Incompatible types for addition."))

        self_params = vars(self).copy()
        self_params.update(
            {
                "intensities": new_inten,
                "wavelengths": self.wavel,
                "from_index": None,
                "to_index": None,
                "backend": "none",
            }
        )

        return Spectrum(**self_params)

    # }}}

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        # {{{
        if isinstance(other, Spectrum):
            if np.isclose(self.wavel, other.wavel).all():
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
        Multiplies the first spectrum's intensities by the second. It can
        multiply spectrums with numpy arrays and lists as well, as long as they
        are the same length as the spectrum's wavelengths array.

        This operation will always return another spectrum with the multiplied intensities.
        """

        if isinstance(other, Spectrum):

            if np.isclose(self.wavel, other.wavel).all():
                new_inten = self.inten * other.inten
            else:
                raise ValueError(
                    "The divided spectrums must have the same wavelengths array."
                )

        elif isinstance(other, (np.ndarray, list)):

            if len(other) == self.wavel.size or len(other) == 1:
                new_inten = self.inten * other

            else:
                raise (
                    ValueError(
                        "The other operand must have the same size as the spectrum's wavelengths array, or size 1."
                    )
                )

        elif isinstance(other, (float, int)):

            new_inten = self.inten * other

        else:
            raise (TypeError("Incompatible types for multiplication."))

        self_params = vars(self).copy()
        self_params.update(
            {
                "intensities": new_inten,
                "wavelengths": self.wavel,
                "from_index": None,
                "to_index": None,
                "backend": "none",
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

            if np.isclose(self.wavel, other.wavel).all():
                new_inten = self.inten / other.inten
            else:
                raise ValueError(
                    "The divided spectrums must have the same wavelengths array."
                )

        elif isinstance(other, (np.ndarray, list)):

            if len(other) == self.wavel.size or len(other) == 1:
                new_inten = self.inten / other

            else:
                raise (
                    ValueError(
                        "The other operand must have the same size as the spectrum's wavelengths array, or size 1."
                    )
                )

        elif isinstance(other, (float, int)):

            new_inten = self.inten / other

        else:
            raise (TypeError("Incompatible types for division."))

        self_params = vars(self).copy()
        self_params.update(
            {
                "intensities": new_inten,
                "wavelengths": self.wavel,
                "from_index": None,
                "to_index": None,
                "backend": "none",
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
            return self.inten[key]
        elif isinstance(key, float):
            int_index = self.find_wavel_index(self.wavel, key)
            return self.inten[int_index]
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
                self.find_wavel_index(self.wavel, x)
                if isinstance(x, float)
                else x
                for x in key
            ]
        elif isinstance(key, float):
            key = self.find_wavel_index(self.wavel, key)
        elif isinstance(key, int):
            if abs(key) > self.wavel.size:
                raise IndexError(
                    f"Invalid index of {val} for wavelengths array of size {self.wavel.size}"
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

        self.inten[key] = val

    # }}}

    def __contains__(self, value):
        raise NotImplementedError

    def __repr__(self):
        return "Spectrum(intensities={}, wavelengths={})".format(self.wavel, self.inten)

    def __len__(self):
        return self.wavel.size
