# spectrabuster
Tools for simplifying the processing and storing of spectrums acquired using python-seabreeze. Basically consists of the Spectrum class, which provides an easy way to abtract away much of the overhead associated with dealing with large quantities of irradiance spectrums. It is meant primarily to be scalable and easy to use.

# Installation
The only requirements are [python-seabreeze](https://github.com/ap--/python-seabreeze) and numpy (which is a requirement of python-seabreeze anyway). Installing matplotlib is not required, but useful if you wish to plot your spectrums and run most of the examples.

# Usage
## Examples
Acquiring a new spectrum with the first spectrometer found by python-seabreeze, with integration time of 10 ms, plot it, then save it to a gnuplot-compatible file:
```
import spectrabuster
from matplotlib import pyplot
intenS = Spectrum(int_time=10*1000)

# intenS.Spectrum returns a tuple of the wavelengths and intensities
plt.plot(*intenS.spectrum)
plt.show()

intenS.write_to_file("intenS.dat")
```

## Documentation
Coming eventually. For the moment you can simply read the comment paragraphs explaining what each function does.

# Acknowledgements
This project was created as part of an undergraduate research project funded by FAPESP (grant n. 2019/06376-9). I'd also like to thank [Andreas Poehlmann]( https://github.com/ap--) for maintaining python-seabreeze and distributing it under a FLOSS license.
