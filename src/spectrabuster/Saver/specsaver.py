from .saver import Saver


class SpecSaver(Saver):

    """
    Overrides the save_plot method, specifically to be able to plot spectrums generated
    by the Spectrum class.
    """

    def save_plot(
        self, spec, ylabel="Spectral Irradiance (mW/m^2*nm)", xlabel="Wavelengths (nm)"
    ):
        # {{{
        """
        Stores the plot of a spectrum.
        """

        fig, ax = plt.subplots()
        ax.plot(*spec.spectrum, "b-")

        ax.set_xlim(spec.wavelengths[0], spec.wavelengths[-1])

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{spec.capture_date} {spec.capture_time} UV: {spec.uv}")
        fig.savefig(f"{self.path}{self.prefix}.png")
        plt.close(fig)


# }}}
