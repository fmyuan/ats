#/usr/bin/env python
"""Functions for plotting column runs of ATS."""
import sys,os
from matplotlib import pyplot as plt
import matplotlib.cm
import colors
import plot_lines

import ats_xdmf, ats_units

epilog = """
Note this plots:
* subsurface quantities (var, z) with a line for each time
* surface quantities (t, var) with a single line

Layout
------

One way of specifying what to plot is to use the layout (-x) 
option.  This option takes a potentially 2D array of lists of variables
to plot in a 2D array of axes on the figure.  The best way to understand
how to use this option is via examples:

 --layout="[pressure, surface-pressure],[temperature, surface-temperature]"

   Makes a figure with 4 axes in a 2x2 block.  Note each list is a row.

--layout="saturation_liquid & saturation_ice, temperature"

   Makes a figure with 2 axes, the first containing saturations and
   the second temperature.

 --layout=arctic

   Makes a plot for Arctic column runs.

Also, more default layouts might be reasonable to add.

Slicing
-------
Slicing in time (only for subsurface currently).  Valid choices for the
--time-slice option are:

 * A single int -- this specifies a single cycle
 * A list of comma-separated ints (no spaces) -- a list of cycles
 * A list of up to three colon-separated ints (no spaces) -- a numpy-style
   slice of cycles.

Note this also tries to guess reasonable values for axis extents.
There is no way of overriding the default values right now other than
to edit the code -- this might be an area for future development.
"""

def valid_layout(arg):
    if arg == 'arctic':
        arg = '[[saturation_liquid+saturation_ice, surface-ponded_depth],'+ \
               '[temperature, snow-depth]]'

    # whitespace never changes things -- remove it all
    arg = arg.replace(' ','')
    if not arg.startswith('[['):
        arg = '['+arg
    if not arg.startswith('[['):
        arg = '['+arg

    if not arg.endswith(']]'):
        arg = arg+']'
    if not arg.endswith(']]'):
        arg = arg+']'

    layout = list()
    assert(arg.startswith('['))
    assert(arg.endswith(']'))
    arg = arg[1:-1]
    rows = arg.split(']')
    assert(len(rows) > 1)

    empty_last = rows.pop()
    assert(empty_last == '' or empty_last == ',')
    for row in rows:
        if row.startswith(',['): # second or later row
            row = row[2:]
        elif row.startswith('['): # first row
            row = row[1:]
        else:
            assert(False)

        row_layout = list()
        columns = row.split(',')
        
        if columns[-1] == '':
            columns.pop()
        for column in columns:
            variables = column.split('&')
            row_layout.append(variables)
        layout.append(row_layout)

    n_cols = max(len(row) for row in layout)
    for row in layout:
        if len(row) != n_cols:
            row.append(list())
    return layout    


def domain_var(varname):
    if '.' in varname:
        dot_split = varname.split('.')
        suffix = '.'.join(dot_split[1:])
        varname = dot_split[0]
    else:
        suffix = ''

    vs = varname.split('-')

    if len(vs) >= 2:
        domain = vs[0]
        varname = '-'.join(vs[1:])
        if suffix != '':
            varname = varname + '.' + suffix
        
    elif len(vs) == 1:
        domain = ''
        varname = varname
        if suffix != '':
            varname = varname + '.' + suffix

    return domain, varname



def label(varname):
    return '{} [{}]'.format(varname, ats_units.get_units(varname))
    
def annotate_surface(col, ax, time_unit):
    ax.set_ylabel(','.join([label(c) for c in col]))
    ax.set_xlabel('time [{}]'.format(time_unit))

def annotate_subsurface(col, ax):
    ax.set_ylabel('z [m]')
    ax.set_xlabel(','.join([label(c) for c in col]))
    
def annotate(layout, axs, time_unit):
    axs[0][0].legend()
    for i,row in enumerate(layout):
        for j,col in enumerate(row):
            if col[0].startswith('surface') or col[0].startswith('snow'):
                annotate_surface(col, axs[i][j], time_unit)
            else:
                annotate_subsurface(col, axs[i][j])

def plot_subsurface(vis, col, ax, label, color=None, cmap=None):
    if cmap is None:
        cmap = colors.alpha_cmap(color)

    formats = ['-', '--', '-.']
    for i, (varname, form) in enumerate(zip(col, formats)):
        vis.plotLinesInTime(varname, ax=ax, colorbar_ticks=(i == 0), cmap=cmap, linestyle=form, colorbar_label=f'{varname} of {label} in time')

def animate_subsurface(vis, varnames, ax, label, colors=None):
    if type(colors) is str:
        colors = [colors,]
    elif colors is None:
        colors = [None,]*len(varnames)

    z = vis.centroids[:,2]

    formats = ['-', '--', '-.']

    datas = []
    lines = []
    for varname, color, form in zip(varnames, colors, formats):
        data = vis.getArray(varname)
        assert(len(data.shape) == 2)
        assert(data.shape[1] == len(vis.centroids))

        datas.append(data)
        lines.extend(ax.plot([], [], form, color=color, label=label))

    def animate(frm):
        for d, l in zip(datas, lines):
            l.set_data(d[frm,:], z)

        return lines

    return animate

def plot_surface(vis, col, ax, label, color):
    formats = ['-', '--', '-.']

    for varname, form in zip(col, formats):
        data = vis.getArray(varname)
        assert(data.shape[1] == 1)
        ax.plot(vis.times, data[:,0], form, color=color, label=label)

def plot(vis_objs, layout, axs, dirname, color_or_cmap, color_mode):
    for i,row in enumerate(layout):
        for j,col in enumerate(row):
            ax = axs[i][j]
            domain, _ = domain_var(col[0])
            vis = vis_objs[domain]
            if color_mode == 'runs':
                # color_or_cmap is a color
                if domain == '':
                    plot_subsurface(vis, col, ax, dirname, color_or_cmap)
                else:
                    plot_surface(vis, col, ax, dirname, color_or_cmap)
            elif color_mode == 'time':
                if domain == '':
                    plot_subsurface(vis, col, ax, dirname, None, color_or_cmap)
                else:
                    cm = colors.cm_mapper(-.1, 1, color_or_cmap)
                    color = cm(0)
                    plot_surface(vis, col, ax, dirname, color)

def layout_flattener(layout):
    for row in layout:
        for col in row:
            for v in col:
                yield v

def directory(d):
    """Argparse validator for a directory."""
    if not os.path.isdir(d):
        raise argparse.ArgumentTypeError('Nonexistent directory "{}"'.format(d))
    return d    

def time_slice(sl):
    """Argparse validator for slices."""
    if ',' in sl:
        return map(int, sl.split(','))
    elif ':' in sl:
        def s_to_i(s):
            if s == '':
                return None
            else:
                return int(s)
            
        return slice(*map(s_to_i, sl.split(':')))
    else:
        return int(sl)
                
def plotColumnData(directories,
       layout,
       color_mode='runs',
       color_sample='enumerated',
       color_map='jet',
       subsurface_time_slice=None,
       surface_time_slice=None,
       data_filename_format='ats_vis_{}_data.h5',
       mesh_filename_format='ats_vis_{}_mesh.h5',
       time_unit='d',
       fig=None,
       figsize=[5,3]):
    """Plots the column data given directories and a layout."""

    if isinstance(layout, str):
        layout = valid_layout(layout)

    if color_mode == 'runs':
        if color_sample == 'enumerated':
            color_list = colors.enumerated_colors(len(directories))
        else:
            color_list = colors.sampled_colors(len(directories),
                                               getattr(matplotlib.cm, color_map))
    elif color_mode == 'time':
        color_list = [color_map,]*len(directories)
        

    if fig is None:
        fig = plt.figure(figsize=figsize)
    axs = fig.subplots(len(layout), len(layout[0]), squeeze=False)

    domains = set([domain_var(v)[0] for v in layout_flattener(layout)])
    for dirname, color in zip(directories, color_list):
        vis_objs = dict()
        for domain in domains:
            vis = ats_xdmf.VisFile(dirname, domain,
                                   ats_xdmf.valid_data_filename(domain, data_filename_format),
                                   ats_xdmf.valid_mesh_filename(domain, mesh_filename_format),
                                   time_unit=time_unit)
            if domain == '':
                vis.loadMesh(columnar=True)
                if subsurface_time_slice is not None:
                    vis.filterIndices(subsurface_time_slice)
            else:
                vis.loadMesh()
                if surface_time_slice is not None:    
                    vis.filterIndices(surface_time_slice)

            vis_objs[domain] = vis

        plot(vis_objs, layout, axs, dirname, color, color_mode)

    annotate(layout, axs, time_unit)
    return axs

    
if __name__ == '__main__':
    import argparse
    import colors

    parser = argparse.ArgumentParser("Plot column runs of ATS", epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('directories', nargs='+', type=directory,
                        help='Directories of runs to plot.')
    parser.add_argument('--data-filename-format', type=str,
                        default='ats_vis_{}_data.h5',
                        help='Format string to generate a data hdf5 filename.')
    parser.add_argument('--mesh-filename-format', type=str,
                        default='ats_vis_{}_mesh.h5',
                        help='Format string to generate a mesh hdf5 filename.')
    parser.add_argument('--time-unit', type=str, default='d',
                        help='Unit for time plots.')

    parser.add_argument('--subsurface-time-slice', type=time_slice, 
                        help='Time slice information, see below')
    parser.add_argument('--surface-time-slice', type=time_slice, 
                        help='Time slice information, see below')
    
    parser.add_argument('-x', '--layout', type=valid_layout,
                        help='Layout in a complicated format -- see below.')
    parser.add_argument('-f', '--figsize', type=float, nargs=2, default=[5,3],
                        help='Figure size')
    parser.add_argument('--color-mode', type=str,
                        choices=['runs', 'time'], default='runs',
                        help='Choose whether colors represent different runs or time.')
    parser.add_argument('--color-sample', type=str,
                        choices=['enumerated','sampled'], default='enumerated',
                        help='Choose colors from an enumerate set or a sampled colormap.')
    parser.add_argument('--color-map', type=str,
                        default='jet',
                        help='Choose which colormap to use.')

    
    args = parser.parse_args()
    plotColumnData(args.directories, args.layout, args.color_mode, args.color_sample, args.color_map,
                   args.subsurface_time_slice, args.surface_time_slice,
                   args.data_filename_format, args.mesh_filename_format,
                   args.time_unit, args.figsize)
    plt.show()
    sys.exit(0)
    
    
