# spectrabuster
Tools for simplifying the processing and storing of spectrums acquired using python-seabreeze. Basically consists of the Spectrum class, which provides an easy way to abtract away much of the overhead associated with dealing with large quantities of irradiance spectrums. It is meant primarily to be scalable and easy to use.

# Installation
The only requirements are [python-seabreeze](https://github.com/ap--/python-seabreeze) and numpy (which is a requirement of python-seabreeze anyway). Installing matplotlib is not required, but useful if you wish to plot your spectrums and run most of the examples.
```
pip3 install git+https://github.com/eduardosprp/spectrabuster.git
```

# Usage
## Examples
Acquiring a new spectrum with the first spectrometer found by python-seabreeze, with integration time of 10 ms and with wavelengths between 250.0 and 800.0 nm, plot it, then save it to a gnuplot-compatible file:
```
from spectrabuster import Spectrum
from matplotlib import pyplot as plt
intenS = Spectrum(int_time=10*1000, from_index=250.0, to_index=800.0)

# intenS.Spectrum returns a tuple of the wavelengths and intensities
plt.plot(*intenS.spectrum)
plt.show()

intenS.write_to_file("intenS.dat")
```

Loading spectral irradiance calibration from file, acquiring regular and dark intensities, applying the calibration and checking a specific wavelength:
```
from spectrabuster import Spectrum
R = Spectrum.from_file("R.dat")
intenD = Spectrum() # measures the spectrum with previously defined integration time
intenS = Spectrum() - intenD

spectral_irrad = [inten/R[i] for inten, i in enumerate(intenS) if intenS.wavelengths[i] == R.wavelengths[i]]

print(spectral_irrad[535.0])
```
## Documentation
Coming eventually. For the moment you can simply read the comment paragraphs explaining what each function does.

# Acknowledgements
This project was created as part of an undergraduate research project funded by FAPESP (grant n. 2019/06376-9). I'd also like to thank [Andreas Poehlmann]( https://github.com/ap--) for maintaining python-seabreeze and distributing it under a FLOSS license.
