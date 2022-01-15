import time
from ase.utils import writer
from ase.io.utils import PlottingVariables, make_patch_list


class EPS(PlottingVariables):
    def __init__(self, atoms,
                 rotation='', radii=None,
                 bbox=None, colors=None, scale=20, maxwidth=500,
                 **kwargs):
        """Encapsulated PostScript writer.

        show_unit_cell: int
            0: Don't show unit cell (default).  1: Show unit cell.
            2: Show unit cell and make sure all of it is visible.
        """
        PlottingVariables.__init__(
            self, atoms, rotation=rotation,
            radii=radii, bbox=bbox, colors=colors, scale=scale,
            maxwidth=maxwidth,
            **kwargs)

    def write(self, fd):
        renderer = self._renderer(fd)
        self.write_header(fd)
        self.write_body(fd, renderer)
        self.write_trailer(fd, renderer)

    def write_header(self, fd):
        from matplotlib.backends.backend_ps import psDefs

        fd.write('%!PS-Adobe-3.0 EPSF-3.0\n')
        fd.write('%%Creator: G2\n')
        fd.write('%%CreationDate: %s\n' % time.ctime(time.time()))
        fd.write('%%Orientation: portrait\n')
        bbox = (0, 0, self.w, self.h)
        fd.write('%%%%BoundingBox: %d %d %d %d\n' % bbox)
        fd.write('%%EndComments\n')

        Ndict = len(psDefs)
        fd.write('%%BeginProlog\n')
        fd.write('/mpldict %d dict def\n' % Ndict)
        fd.write('mpldict begin\n')
        for d in psDefs:
            d = d.strip()
            for l in d.split('\n'):
                fd.write(l.strip() + '\n')
        fd.write('%%EndProlog\n')

        fd.write('mpldict begin\n')
        fd.write('%d %d 0 0 clipbox\n' % (self.w, self.h))

    def _renderer(self, fd):
        # Subclass can override
        from matplotlib.backends.backend_ps import RendererPS
        return RendererPS(self.w, self.h, fd)

    def write_body(self, fd, renderer):
        patch_list = make_patch_list(self)
        for patch in patch_list:
            patch.draw(renderer)

    def write_trailer(self, fd, renderer):
        fd.write('end\n')
        fd.write('showpage\n')


@writer
def write_eps(fd, atoms, **parameters):
    EPS(atoms, **parameters).write(fd)
