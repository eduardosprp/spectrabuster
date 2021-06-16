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

    if type(spec) in (tuple, list, np.ndarray):
        pass
    # Hacky way to avoid a circular import
    elif "Spectrum" in repr(spec):
        spec = spec.spectrum
    else:
        raise ValueError(
            "Please pass an interable type or a Spectrum instance to weight_irrad."
        )

    return np.array(
        [
            x for x in action_spec_gen(spec, from_wavel, to_wavel)
        ]
    )
# }}}

def calc_uv_index(spec, from_wavel=286.0, to_wavel=400.0):
    # {{{
    """
    Calculates the UV index based on Mckinley-Diffey's action spectra for erythema.
    """

    weighted_irrad = weight_irrad(spec, from_wavel, to_wavel)

    UV_index = round(0.04 * trapz(weighted_irrad, spec[0]), 2)

    return UV_index
    # }}}


