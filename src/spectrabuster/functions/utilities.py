import numpy as np

"""
General utilities functions. Some are internal to spectrabuster and others,
mainly those which deal with handling arrays, are useful in general.
"""

def broadcast_arrays(spec1, spec2):
# {{{
    """
    Takes two tuples consisting (in principle) of wavelengths and
    intensities, verifies which portion of one's wavelengths fits inside
    the other's, and slices them so as to preserve only the portion of
    the wavelengths common to both. Then returns a tuple with the sliced
    wavelengths array and both of the intensities. Preference is given to
    the first tuple passed.

    This kind of thing is important internally but I guess it can be pretty
    useful outside of spectrabuster as well.
    """

    spec1, _ = _type_check(spec1)
    spec2, _ = _type_check(spec2)

    wavel1, inten1 = spec1
    wavel2, inten2 = spec2

    if wavel1.size > wavel2.size:
        from_index = find_wavel_index(wavel1, wavel2[0])
        to_index = find_wavel_index(wavel1, wavel2[-1])

        wavel1 = wavel1[from_index : to_index + 1]
        inten1 = inten1[from_index : to_index + 1]

    elif wavel2.size > wavel1.size:
        from_index = find_wavel_index(wavel2, wavel1[0])
        to_index = find_wavel_index(wavel2, wavel1[-1])

        wavel1 = wavel2[from_index : to_index + 1]
        inten2 = inten2[from_index : to_index + 1]

    return wavel1, inten1, inten2
# }}}

def join(spec1, spec2):
# {{{
    raise NotImplementedError
# }}}

def find_wavel_index(wavel_array, wavel, margin=0.5):
    # {{{
    """
    Attempts to find 'wavel' in 'wavel_array'. Will try using the closest
    wavelength at most 0.5 units from 'wavel'
    """

    array_diffs = np.abs(wavel_array - wavel)
    closest_index = array_diffs.argmin()

    if np.isclose(wavel_array[closest_index], wavel):
        return closest_index

    elif array_diffs[closest_index] < 0.5:
        _warn(
            f"Exact match for {wavel} not found. Using {wavel_array[closest_index]} instead."
        )
        return closest_index

    else:
        raise ValueError(
            f"A close enough {wavel} wasn't found. Closest value is {wavel_array[closest_index]}."
        )
    # }}}

def _warn(message):
# {{{
    """
    Simply prints a message. For now it's pretty useless since there is no way
    to disable the warnings, but that'll come soon enough.
    """

    print(message)
# }}}

def _type_check(spec, int_time=None):
# {{{
    # All this type checking is so common that I found it better to just
    # relegate it to its own function in order to save some lines.

    if type(spec) in (tuple, list, np.ndarray):
        pass
    # Hacky way to avoid a circular import
    elif "Spectrum" in repr(spec):
        spec = spec.spectrum
        int_time = spec.int_time if int_time is None else int_time
    else:
        raise ValueError(
            "Please pass an interable type or a Spectrum instance to the spectrum processing function."
        )

    return spec, int_time
# }}}

def slice_array(spec, indices, **kwargs):
    # {{{
    """
    Takes in two arrays and returns them sliced according to
    indices=(from_index, to_index).

    If the indeces are integers, it takes them to be literal indeces for the
    array. If they are floats, then it'll assume they are wavelengths whose
    literal indeces must be found before slicing.

    This behaviour can be overriden by passing literal_indices=True or False
    """

    wavel, inten = spec
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
            index_wavel = find_wavel_index(wavel, index)
            new_indices.append(index_wavel)

    array_slice = slice(new_indices[0], new_indices[1])
    return wavel[array_slice], inten[array_slice], array_slice

