import numpy as np
from math import exp
from scipy.integrate import trapz

"""
Functions related to processing arrays and spectrums. They are primarily to be
used inside the Spectrum class, but they're also useful by themselves. Please
note that every function here expects a tuple (spec) with the wavelengths and
the intensities, in this order.
"""

def action_spec_gen(spec, from_wavel=286.0, to_wavel=400.0):
    # {{{
    """
    Simple implementation of Mckinley-Diffey's action spectrum.
    Good for use in list comprehensions.
    """

    for wavel, irrad in zip(*spec):
        if from_wavel <= wavel < 298:
            yield irrad
        elif 298 <= wavel < 328:
            yield exp(0.216 * (298 - wavel)) * irrad
        elif 328 <= wavel < to_wavel:
            yield exp(0.034 * (139 - wavel)) * irrad
        else:
            yield 0
    # }}}

def weight_irrad(spec, from_wavel=286.0, to_wavel=400.0):
# {{{
    """
    Takes in either a Spectrum object or the wavelengths and intensities arrays
    by themselves as a tuple. Returns the intensities array weighted by
    Mckinley-Diffey's action spectrum for erythema.
    """

    spec, _ = _type_check(spec)

    return np.array(
        [
            x for x in action_spec_gen(spec, from_wavel, to_wavel)
        ]
    )
# }}}

def calc_uv_index(spec, from_wavel=286.0, to_wavel=400.0):
    # {{{
    """
    Calculates the UV index based on Mckinley-Diffey's action spectra for
    erythema. Note that during this entire process spectrabuster assumes the
    intensities array has units of mW/m2/nm. Making the unit separately
    specifiable is a goal for the future.
    """

    weighted_irrad = weight_irrad(spec, from_wavel, to_wavel)

    # this 0.04 hinges on the assumption that the unit being used is mW/m2
    UV_index = round(0.04 * trapz(weighted_irrad, spec[0]), 2)

    return UV_index
    # }}}

def spectral_irrad(spec, calib, int_time=None):
# {{{
    """
    Calculates the spectral irradiance array from the intensities spec[1] and
    according to some calibration array. The calibration array must be the same
    length as the wavelengths and intensities arrays. In the future I may make
    it possible to pass a tuple of arrays for the calibration, and the
    function'll be smart enough to do the expected interpolation and slicing.


    If no integration time is passed, it assumes the intensities arrays is
    already in counts/s.
    """

    spec, int_time = _type_check(spec, int_time)

    wavel = spec[0]
    inten = count_rate(spec, int_time) if int_time is not None else spec[1]

    # Termina essa merda aí ow   
# }}}

def count_rate(spec, int_time=None):
# {{{
    """
    Returns the intensities array of spec divided by the specified integration
    time, which it assumes to be in microsseconds. 

    This function seems pretty useless now but in the future it'll justify its
    existence by supporting specifying the unit of the integration time.
    """

    spec, int_time = _type_check(spec, int_time)

    inten = spec[1]

    return inten / (int_time * 0.000001)
# }}}

