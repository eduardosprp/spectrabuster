# spectrabuster

Tools for simplifying the processing and storing of spectrums acquired using python-seabreeze. Basically consists of the Spectrum class, which provides an easy way to abtract away much of the overhead associated with dealing with large quantities of irradiance spectrums. It is meant primarily to be scalable and easy to use.

# Installation
```
pip3 install spectrabuster
```

# Usage
## Examples
Acquiring a new spectrum with the first spectrometer found by python-seabreeze, with integration time of 10 ms and with wavelengths between 250.0 and 800.0 nm, plot it, then save it to a gnuplot-compatible file:
```
from spectrabuster import Spectrum
from matplotlib import pyplot as plt
intenS = Spectrum(int_time=10*1000, from_index=250.0, to_index=800.0)

# intenS.spectrum returns a tuple of the wavelengths and intensities
plt.plot(*intenS.spectrum)
plt.show()

intenS.write_to_file("intenS.dat")
```

Loading spectral irradiance calibration from file, acquiring regular and dark intensities, applying the calibration and checking a specific wavelength:
```
from spectrabuster import Spectrum
intenD = Spectrum() # measures the spectrum with previously defined integration time
intenS = Spectrum() - intenD

spectral_irrad = intenS.to_spectral_irrad(calibration_file = "R.dat")

print(spectral_irrad[535.0])
```
## Documentation
Coming eventually. For the moment you can simply read the comment paragraphs explaining what each function does.

# Acknowledgements
This project was created as part of an undergraduate research project funded by FAPESP (grant n. 2019/06376-9). I'd also like to thank [Andreas Poehlmann]( https://github.com/ap--) for maintaining python-seabreeze and distributing it under a FLOSS license.
