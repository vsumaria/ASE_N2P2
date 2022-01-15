from ase.io.jsonio import read_json
from ase.spectrum.band_structure import BandStructure
from ase.cli.main import CLIError


def read_band_structure(filename):
    bs = read_json(filename)
    if not isinstance(bs, BandStructure):
        raise CLIError(f'Expected band structure, but file contains: {bs}')
    return bs


def main(args, parser):
    import matplotlib.pyplot as plt
    bs = read_band_structure(args.calculation)
    emin, emax = (float(e) for e in args.range)
    fig = plt.gcf()
    fig.canvas.set_window_title(args.calculation)
    ax = fig.gca()
    bs.plot(ax=ax,
            filename=args.output,
            emin=emin + bs.reference,
            emax=emax + bs.reference)
    if args.output is None:
        plt.show()


class CLICommand:
    """Plot band-structure.

    Read eigenvalues and k-points from file and plot result from
    band-structure calculation or interpolate
    from Monkhorst-Pack sampling to a given path (--path=PATH).

    Example:

        $ ase band-structure bandstructure.json -r -10 10
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('calculation',
                            help='Path to output file(s) from calculation.')
        parser.add_argument('-o', '--output', help='Write image to a file')
        parser.add_argument('-r', '--range', nargs=2, default=['-3', '3'],
                            metavar=('emin', 'emax'),
                            help='Default: "-3.0 3.0" '
                            '(in eV relative to Fermi level).')

    @staticmethod
    def run(args, parser):
        main(args, parser)
