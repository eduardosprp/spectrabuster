import numpy as np
from time import sleep
from seabreeze.spectrometers import Spectrometer, list_devices
from os import path

class Spectrum(object):

    """
    Class variables. Mainly default values that are used whenever their instance counterparts
    are not specified when initializing a Spectrum object.
    """

# {{{
    device = None                       # will use the first spectrometer seabreeze finds by default 
    correct_nl = False                  # correct non-linearity
    correct_dc = False                  # correct dark counts
    int_time = None                     # integration time in microseconds
    _inten = None                       # intensities for internal use
    _wavel = None                       # wavelengths for internal use
    from_index = None
    to_index = None
    wavel_slice = None
    opt_warnings = True                 # determines whether or not to show warnings
    _current_int_time = None            # keeps track of the integration time most recently set in the spectrometer
# }}}

    def __init__(self, int_time = None,
                       wavelengths = None,
                       device = None,
                       from_index = None,
                       to_index = None,
                       intensities = None,
                       **kwargs): 
# {{{
        """
        Initializes the instance with the default values, unless specified otherwise.
        If you wish to use all the default values except for a few, specify them with kwargs.
        """

        if device is not None:
            self.device = device

            if self.device != Spectrometer:
                raise TypeError("Invalid type for device attribute. Please enter a seabreeze Spectrometer instance.")

        if wavelengths is not None:
            if isinstance(wavelengths, (np.ndarray, list, tuple)):
                self._wavel = np.array(wavelengths)
            else:
                raise TypeError("Invalid type for wavelengths array. Please enter a numpy array, a list, or a tuple.")
        else:
            # Will use the wavelenghts array provided by the spectrometer by default
            
            if not self.device:
                """
                If by this point the spectrometer hasn't been specified, it will use the first
                one seabreeze finds by default
                """

                if list_devices():
                    self.device = Spectrometer(list_devices()[0])
                else:
                    raise RuntimeError("No spectrometers found.")
                self._warn("No spectrometer object passed. Using {} by default.".format(self.device))

            self._wavel = np.array(self.device.wavelengths())
            self._warn("No wavelength array provided. Using the device's wavelength array by default.")

        if int_time is not None:
            self.int_time = int(int_time)

        if from_index is not None:
            if isinstance(from_index, (int, float)):
                self.from_index = from_index
            else:
                raise TypeError("to_index and from_index must be either integers for proper indexes, or floats, for wavelengths.")

        if to_index is not None:
            if isinstance(to_index, (int, float)):
                self.to_index = to_index
            else:
                raise TypeError("to_index and from_index must be either integers for proper indexes, or floats, for wavelengths.")

        if 'correct_nl' in kwargs:
            self.correct_nl = kwargs['correct_nl']

        if 'correct_dc' in kwargs:
            self.correct_dc = kwargs['correct_dc']

        if intensities is not None:
            if isinstance(intensities, (np.ndarray, list, tuple)):
                self._inten = np.array(intensities)
            else:
                raise TypeError("Invalid type for intensities array. Please enter a numpy array, a list, or a tuple.")
        else:
            self._inten = self._measure_inten()
# }}}

    def _measure_inten(self, samples = 1, boxcar = False, boxcar_len = 0):
# {{{
        if self.int_time and Spectrum._current_int_time != self.int_time:
            self.device.integration_time_micros(self.int_time)
            sleep(2)    # accounting for the delay of the spectrometer to change its integration time
            Spectrum._current_int_time = self.int_time
        
        return self.device.intensities(self.correct_dc, self.correct_nl)
# }}}

    def write_to_file(self, file_path = None, save_fields = True, **kwargs):
# {{{
        """
        Stores spectrum in a .dat text file, using a format that is easy to parse
        in gnu octave, R or any other programming language, or visualize in gnuplot,
        or any spreadsheet program.
        """

        overwrite = kwargs['overwrite'] if 'overwrite' in kwargs else None
        if path.exists(file_path):
            if overwrite:
                self._warn(f"WARNING: File {file_path} exists. Overwriting it.")
            else:
                raise RuntimeError(f"File {file_path} already exists. Pass 'overwrite=True' if you are sure you want to overwrite it.")

        # set this kwarg to True if you wish to store the entire wavelengths and intensities array,
        # as opposed to just the array delimited by from_index and to_index
        entire_spectrum = kwargs['entire_spectrum'] if 'entire_spectrum' in kwargs else False
        only_wavelengths = kwargs['only_wavelengths'] if 'only_wavelengths' in kwargs else False
        only_intensities = kwargs['only_intensities'] if 'only_intensities' in kwargs else False

        to_save = {'int_time', 'correct_nl', 'correct_dc'} # fields that will be written to the file
        if entire_spectrum: # will only write these fields if entire_spectrum is set to True
            to_save = to_save.union({'from_index', 'to_index'})

        if not file_path and isinstance(file_path, str):
            raise(ValueError("Please pass a string as the file path wherein to save the spectrum."))

        with open(file_path, 'w+') as arq:
            gen_comments = (f'# {name} = {value}\n' for name, value in vars(self).items() if name in to_save)
            arq.writelines(gen_comments)

            if only_wavelengths:
                if entire_spectrum:
                    gen_wavel_inten = (f'{wavel}\n' for wavel in self._wavel)
                else:
                    gen_wavel_inten = (f'{wavel}\n' for wavel in self.wavelengths)
            elif only_intensities:
                if entire_spectrum:
                    gen_wavel_inten = (f'{inten}\n' for inten in self._inten)
                else:
                    gen_wavel_inten = (f'{inten}\n' for inten in self.intensities)
            else:
                if entire_spectrum:
                    gen_wavel_inten = (f'{wavel}\t{inten}\n' for wavel, inten in zip(*self._spec))
                else:
                    gen_wavel_inten = (f'{wavel}\t{inten}\n' for wavel, inten in zip(*self.spectrum))

            arq.writelines(gen_wavel_inten)
# }}}

    def set_wavel_slice(self):
# {{{
        from_index = self.from_index if self.from_index else 0
        to_index = self.to_index if self.to_index else -1

        if type(from_index) == float:   # assumes the value is a wavelength if it has a decimal point
            from_index = self.find_wavel_index(self._wavel, from_index)
        elif type(from_index) == int:    # assumes the value is a proper index if it is an integer
            if abs(from_index) > self._wavel.size:
                raise IndexError("Invalid index of {} for wavelength array of size {}".format(from_index, self._wavel.size))
        elif type(from_index) == str:
            try:
                float(from_index)
            except ValueError:
                raise TypeError("Invalid type of {} for wavelength index. Please enter either a float for a wavelength or an integer for a proper index.".format(from_index))
            if '.' in from_index:
                from_index = self.find_wavel_index(self._wavel, float(from_index))
            else:
                from_index = int(from_index)
        else:
            raise TypeError("Invalid type of {} for wavelength index. Please enter either a float for a wavelength or an integer for a proper index.".format(from_index))

        if type(to_index) == float:   # assumes the value is a wavelength if it has a decimal point
            to_index = self.find_wavel_index(self._wavel, to_index)
        elif type(to_index) == int:    # assumes the value is a proper index if it is an integer
            if abs(to_index) > self._wavel.size:
                raise IndexError("Invalid index of {} for wavelength array of size {}".format(from_index, self._wavel.size))
        elif type(to_index) == str:
            try:
                float(to_index)
            except ValueError:
                raise TypeError("Invalid type of {} for wavelength index. Please enter either a float for a wavelength or an integer for a proper index.".format(to_index))
            if '.' in to_index:
                to_index = self.find_wavel_index(self._wavel, float(to_index))
            else:
                to_index = int(to_index)
        else:
            raise TypeError("Invalid type of {} for wavelength index. Please enter either a float for a wavelength or an integer for a proper index.".format(to_index))
        self.wavel_slice = slice(from_index, to_index)
# }}}

    def get_wavel_slice(self):
        if self.wavel_slice:# {{{
            pass
        else:
            self.set_wavel_slice()
        return self.wavel_slice# }}}

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
    def from_file(cls, inten_wavel_file = None, **kwargs):
# {{{
        """
        Creates a spectrum instance with the wavelengths and/or intensities read
        from a text file. Additionally, it looks for key-word arguments at the first
        few lines of the file. If the same kwargs are passed to this function, they
        take precedence.
        """
        
        wavel_file = kwargs['wavel_file'] if 'wavel_file' in kwargs else None
        inten_file = kwargs['inten_file'] if 'inten_file' in kwargs else None
        inten_array = None; wavel_array = None; new_kwargs = {}

        if 'inten_wavel_file' in kwargs:
            inten_Wavel_file = kwargs['inten_wavel_file']

        if inten_wavel_file: 
            wavel_array, inten_array, new_kwargs = cls._read_file(inten_wavel_file)

        if wavel_file:
            wavel_array, _, new_kwargs = cls._read_file(wavel_file)
        
        if inten_file:
            inten_array, _, new_kwargs = cls._read_file(inten_file)

        if not inten_file and not inten_wavel_file and not wavel_file:
            cls._warn("WARNING: Instantiating a spectrum with function from_file, but no file path arguments were passed.")

        new_kwargs['intensities'] = inten_array
        new_kwargs['wavelengths'] = wavel_array
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

        with open(text_file, 'r') as arq:

            # Generator for the lines in the archive
            gen_lines = (line.split() for line in arq)

            for line_split in gen_lines:

                if line_split[0] == '#':  # comment, presumably containing arguments for __init__
                    dict_args[line_split[1]] = line_split[3]
                elif len(line_split) > 1:  # 2 or more columns. Will ignore anything after the second column
                    col1.append(float(line_split[0]))
                    col2.append(float(line_split[1]))
                elif len(line_split) == 1:  # 1 column
                    col1.append(float(line_split[0]))
            
            if not dict_args and not col1 and not col2:
                # Check if they're all empty

                raise RuntimeError(f"No arguments, wavelengths and intensities found in {text_file}. Please check if this is a valid file.")

            return np.array(col1), np.array(col2), dict_args
# }}}

    @staticmethod
    def find_wavel_index(wavel_array, wavel):
# {{{
        """
        Attempts to find 'wavel' in 'wavel_array'. Will try using the first wavelength
        at most 0.5 units from 'wavel' if an exact match cannot be found
        """
        
        try:
            wavel_index = np.where(wavel_array == wavel)
            return wavel_index[0][0]
        except:
            try:
                wavel_index = np.where(np.isclose(wavel_array, wavel, atol = 0.5))
                print("Exact match for {} not found. Using {} instead".format(wavel, wavel_array[wavel_index[0][0]]))
                return wavel_index[0][0]
            except:
                raise ValueError(str(wavel) + " not found.")

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
        Adds the spectrum's intensities to the 'other' array. If 'other' is also a spectrum,
        their wavelengths arrays must be equal. If you wish to add spectrums with different
        wavelengths arrays, refer to the 'combine' method.

        This operation will always return another spectrum with the added intensities. The new
        spectrum's fields will be inherited from the first operand.
        """

        if isinstance(other, Spectrum):

            if np.array_equal(self._wavel, other._wavel):
                new_inten = self._inten + other._inten
            else:
                raise ValueError("The added spectrums must have the same wavelengths array.")

        elif isinstance(other, (np.ndarray, list)):

            if len(other) == self._wavel.size or len(other) == 1:
                new_inten = self._inten + other

            elif len(other) == self.wavelengths.size:
                new_inten = self._inten
                new_inten[self.get_wavel_slice()] = self.intensities + other

            else:
                raise(ValueError("The other operand must have the same size as the spectrum's wavelengths array, or size 1."))

        else:
            raise(TypeError("Incompatible types for addition."))

        self_params = vars(self).copy()
        self_params.update({'intensities' : new_inten, 'device' : None})

        return Spectrum(**self_params)

# }}}

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
#{{{
        if isinstance(other, Spectrum):
            if np.array_equal(self._wavel, other._wavel):
                return self + np.negative(other._inten)
            else:
                raise ValueError("The subtracted spectrums must have the same wavelengths array.")
        else:
            return self + np.negative(other)
#}}}

    def __rsub__(self, other):       
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __div__(self, other):
        raise NotImplementedError

    def __getitem__(self, key):
#{{{
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
            raise TypeError("Invalid type for index. Please enter an integer, list, numpy array or a float.")
#}}}

    def __setitem__(self, key, val):
#{{{
        """
        Changes the intensity with index 'key' to 'val'. The new value must be a number,
        a tuple, list or numpy array. In the latter 3 cases, numpy will handle the assignment.
        """

        if isinstance(key, (list, tuple, np.ndarray)):
            key = [self.find_wavel_index(self.wavelengths, x) if isinstance(x, float) else x for x in key]
        elif isinstance(key, float):
            key = self.find_wavel_index(self.wavelengths, key)
        elif isinstance(key, int):
            if abs(key) > self.wavelengths.size:
                raise IndexError(f"Invalid index of {val} for wavelengths array of size {self.wavelengths.size}")
        else:
            raise TypeError("Invalid type for index. Please enter an integer, list, numpy array or a float.")

        if isinstance(val, (tuple, list, np.ndarray)):
            try:
                val = [float(x) for x in val]
            except (TypeError, ValueError) as exception:
                raise ValueError(f"The {type(val)} {val} must contain only numbers.")
        else:
            try:
                val = float(val)
            except:
                raise ValueError(f"Invalid value of {val} for intensity. Please enter something convertible to float.")

        self.intensities[key] = val

#}}}
        
    def __contains__(self, value):
        raise NotImplementedError
    

    def __repr__(self):
        return "Spectrum({}, {})".format(self.wavelengths, self.intensities)

    def __len__(self):
        return self.wavelengths.size

"""
Function main() will execute when you run this file directly
"""

def main():
    print("main fora da classe")# {{{

if __name__ == '__main__':
    main()# }}}
