#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:11:47 2017

@author: engels
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import matplotlib
import glob




# chunk a string, regardless of whatever delimiter, after length characters,
# return a list
def chunkstring(string, length):
    return list( string[0+i:length+i] for i in range(0, len(string), length) )

def cm2inch(value):
    return value/2.54

def deg2rad(value):
    return value*np.pi/180.0

# construct a column-vector for math operatrions. I hate python.
def vct(x):
    v = np.matrix(x)
    v = v[np.newaxis]
    v = v.reshape(len(x),1)
    return v

def ylim_auto(ax, x, y):
   # ax: axes object handle
   #  x: data for entire x-axes
   #  y: data for entire y-axes
   # assumption: you have already set the x-limit as desired
   lims = ax.get_xlim()
   i = np.where( (x > lims[0]) &  (x < lims[1]) )[0]
   ax.set_ylim( y[i].min(), y[i].max() )

# set axis spacing to equal by modifying only the axis limits, not touching the
# size of the figure
def axis_equal_keepbox( fig, ax ):
    w, h = fig.get_size_inches()
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    if (x2-x1)/w > (y2-y1)/h:
        # adjust y-axis
        l_old = (y2-y1)
        l_new = (x2-x1) * h/w
        plt.ylim([ y1-(l_new-l_old)/2.0, y2+(l_new-l_old)/2.0])
    else:
        # adjust x-axis
        l_old = (x2-x1)
        l_new = (y2-y1) * w/h
        plt.xlim([ x1-(l_new-l_old)/2.0, x2+(l_new-l_old)/2.0])

# read pointcloudfile
def read_pointcloud(file):
    data = np.loadtxt(file, skiprows=1, delimiter=' ')
    if data.shape[1] > 6:
        data = np.delete( data, range(3,data.shape[1]-3) , 1)
    print(data.shape)
    return data

def write_pointcloud(file, data, header):
    write_csv_file( file, data, header=header, sep=' ')


# Read in a t-file, optionally interpolate to equidistant time vector
def load_t_file( fname, interp=False, time_out=None, return_header=False, verbose=True ):
    if verbose:
        print('reading file %s' %fname)

    # does the user want the header back?
    if return_header:
        # read header line
        f = open(fname, 'r')
        header = f.readline()
        # a header is a comment that begins with % (not all files have one)
        if "%" in header:
            # remove comment character
            header = header.replace('%',' ')
            # convert header line to list of strings
            header = chunkstring(header, 16)
            f.close()

            # format and print header
            for i in range(0,len(header)):
                # remove spaces (leading+trailing, conserve mid-spaces)
                header[i] = header[i].strip()
                # remove newlines
                header[i] = header[i].replace('\n','')
                if verbose:
                    print( 'd[:,%i] %s' % (i, header[i] ) )
        else:
            print('You requested a header, but we did not find one...')
            # return empty list
            header = []

    #------------------------
    # read the data from file
    #------------------------
    data_raw = np.loadtxt( fname, comments="%")
    nt_raw, ncols = data_raw.shape

    # retain only unique values (judging by the time stamp, so if multiple rows
    # have exactly the same time, only one of them is kept)
    dummy, unique_indices = np.unique( data_raw[:,0], return_index=True )
    data = np.copy( data_raw[unique_indices,:] )


    # info on data
    nt, ncols = data.shape
    if verbose:
        print( 'nt_unique=%i nt_raw=%i ncols=%i' % (nt, nt_raw, ncols) )

    # if desired, the data is interpolated to an equidistant time grid
    if interp:
        if time_out is None:
            # time stamps as they are in the file, possibly nont equidistant
            time_in = np.copy(data[:,0])
            # start & end time
            t1 = time_in[0]
            t2 = time_in[-1]
            # create equidistant time vector
            time_out = np.linspace( start=t1, stop=t2, endpoint=True, num=nt )
        # equidistant time step
        dt = time_out[1]-time_out[0]
        if verbose:
            print('interpolating to nt=%i (dt=%e) points' % (time_out.size, dt) )

            if data[0,0] > time_out[0] or data[-1,0] < time_out[-1]:
                print('WARNING you want to interpolate beyond bounds of data')
                print("Data: %e<=t<=%e Interp: %e<=t<=%e" % (data[0,0], data[-1,0], time_out[0], time_out[-1]))

        data = interp_matrix( data, time_out )

    # return data
    if return_header:
        return data, header
    else:
        return data



def stroke_average_matrix( d, tstroke=1.0, t1=None, t2=None ):
    # start time of data
    if t1 is None:
        t1 = d[0,0]
    # end time of data
    if t2 is None:
        t2 = d[-1,0]

    # will there be any strokes at all?
    if t2-t1 < tstroke:
        print('warning: no complete stroke present, not returning any averages')

    if t1 - np.round(t1) >= 1e-3:
        print('warning: data does not start at full stroke (tstart=%f)' % t1)

    # allocate stroke average matrix
    nt, ncols = d.shape
    navgs = np.int( np.floor((t2-t1)/tstroke) )
    D = np.zeros([navgs,ncols])
    # running index of strokes
    istroke = 0

    # go in entire strokes
    while t1+tstroke <= t2:
        # begin of this stroke
        tbegin = t1
        # end of this stroke
        tend = t1+tstroke
        # iterate
        t1 = tend
        # find index where stroke begins:
        i = 0
        while d[i,0]<tbegin:
            i = i+1
        # find index where stroke ends
        j = i
        while d[j,0]<tend and j<d.shape[0]-1:
            j = j+1

#        print('t1=%f t2=%f i1=%i i2=%i %f %f' % (tbegin, tend, i, j, d[i,0], d[j,0]))

        # actual integration. see wikipedia :)
        # the integral f(x)dx over x2-x1 is the average of the function on that
        # interval. note this script is more precise than the older matlab versions
        # as it is, numerically, higher order. the results are however very similar
        # (below 1% difference)
        for col in range(0,ncols):
            D[istroke,col] = np.trapz( d[i:j+1,col], x=d[i:j+1,0]) / (tend-tbegin)

        istroke = istroke + 1
    return D



def write_csv_file( fname, d, header=None, sep=';'):
    # open file, erase existing
    f = open( fname, 'w' )

    # if we specified a header ( a list of strings )
    # write that
    if not header == None:
        # write column headers
        if header is list:
            for name in header:
                f.write( name+sep )
        else:
            f.write(header)
        # newline after header
        f.write('\n')
        # check

    nt, ncols = d.shape

    for it in range(nt):
        for icol in range(ncols):
            f.write( '%e%s' % (d[it,icol], sep) )
        # new line
        f.write('\n')
    f.close()


def read_param(config, section, key):
    # read value
    value = config[section].get(key)
    # remove comments and ; delimiter, which flusi uses for reading.
    value = value.split(';')[0]
    return value




def read_param_vct(config, section, key):
    value = read_param(config, section, key)
    value = np.array( value.split() )
    value = value.astype(np.float)
    return value



def Fserieseval(a0,ai,bi,time):
    # evaluate the Fourier series given by a0, ai, bi at the time instant time
    # note we divide amplitude by 2
    y = a0/2.0
    for k in range( ai.size ):
        # note pythons tedious 0-based indexing, so wavenumber is k+1
        y = y + ai[k]*np.cos(2.0*np.pi*float(k+1)*time) + bi[k]*np.sin(2.0*np.pi*float(k+1)*time)
    return y



def read_kinematics_file( fname ):
    import configparser

    config = configparser.ConfigParser( inline_comment_prefixes=(';'), allow_no_value=True )
    # read the ini-file
    config.read(fname)

    if config['kinematics']:
        convention = read_param(config,'kinematics','convention')
#        print(convention)

        nfft_phi = int(read_param(config,'kinematics','nfft_phi'))
        nfft_alpha = int(read_param(config,'kinematics','nfft_alpha'))
        nfft_theta = int(read_param(config,'kinematics','nfft_theta'))

        a0_phi = float(read_param(config,'kinematics','a0_phi'))
        a0_alpha = float(read_param(config,'kinematics','a0_alpha'))
        a0_theta = float(read_param(config,'kinematics','a0_theta'))

        ai_alpha = read_param_vct(config,'kinematics','ai_alpha')
        bi_alpha = read_param_vct(config,'kinematics','bi_alpha')
        ai_theta = read_param_vct(config,'kinematics','ai_theta')
        bi_theta = read_param_vct(config,'kinematics','bi_theta')
        ai_phi = read_param_vct(config,'kinematics','ai_phi')
        bi_phi = read_param_vct(config,'kinematics','bi_phi')


        return a0_phi, ai_phi, bi_phi, a0_alpha, ai_alpha, bi_alpha, a0_theta, ai_theta, bi_theta
    else:
        print('This seems to be an invalid ini file as it does not contain the kinematics section')



def visualize_kinematics_file( fname ):
    a0_phi, ai_phi, bi_phi, a0_alpha, ai_alpha, bi_alpha, a0_theta, ai_theta, bi_theta = read_kinematics_file( fname )

    # time vector for plotting
    t = np.linspace(0,1,1000,endpoint=True)
    # allocate the lazy way
    alpha = 0.0*t.copy()
    phi = 0.0*t.copy()
    theta = 0.0*t.copy()

    for i in range(t.size):
        alpha[i]=Fserieseval(a0_alpha, ai_alpha, bi_alpha, t[i])
        phi[i]=Fserieseval(a0_phi, ai_phi, bi_phi, t[i])
        theta[i]=Fserieseval(a0_theta, ai_theta, bi_theta, t[i])

    plt.rcParams["text.usetex"] = True
    plt.close('all')
    plt.figure( figsize=(cm2inch(12), cm2inch(7)) )
    plt.subplots_adjust(bottom=0.16, left=0.14)

    plt.plot(t, phi  , label='positional')
    plt.plot(t, alpha, label='feathering')
    plt.plot(t, theta, label='deviation')

    plt.legend()
    plt.xlim([0,1])
    plt.xlabel('$t/T$')
    plt.ylabel('$[^{\circ}]$')
    ax = plt.gca()
    ax.tick_params( which='both', direction='in', top=True, right=True )
    plt.savefig( fname.replace('.ini','.pdf'), format='pdf' )
    plt.savefig( fname.replace('.ini','.png'), format='png', dpi=300 )


def Rx( angle ):
    # rotation matrix around x axis
    Rx = np.ndarray([3,3])
    Rx = [[1.0,0.0,0.0],[0.0,np.cos(angle),np.sin(angle)],[0.0,-np.sin(angle),np.cos(angle)]]
    # note the difference between array and matrix (it is the multiplication)
    Rx = np.matrix( Rx )
    return Rx


def Ry( angle ):
    # rotation matrix around y axis
    Rx = np.ndarray([3,3])
    Rx = [[np.cos(angle),0.0,-np.sin(angle)],[0.0,1.0,0.0],[+np.sin(angle),0.0,np.cos(angle)]]
    # note the difference between array and matrix (it is the multiplication)
    Rx = np.matrix( Rx )
    return Rx


def Rz( angle ):
    # rotation matrix around z axis
    Rx = np.ndarray([3,3])
    Rx = [[ np.cos(angle),+np.sin(angle),0.0],[-np.sin(angle),np.cos(angle),0.0],[0.0,0.0,1.0]]
    # note the difference between array and matrix (it is the multiplication)
    Rx = np.matrix( Rx )
    return Rx


def Rmirror( x0, n):
    # mirror by a plane through origin x0 with given normal n
    # source: https://en.wikipedia.org/wiki/Transformation_matrix#Reflection_2
    Rmirror =  np.zeros([4,4])

    a, b, c = n[0], n[1], n[2]
    d = -(a*x0[0] + b*x0[1] + c*x0[2])

    Rmirror = [ [1-2*a**2,-2*a*b,-2*a*c,-2*a*d], [-2*a*b,1-2*b**2,-2*b*c,-2*b*d], [-2*a*c,-2*b*c,1-2*c**2,-2*c*d],[0,0,0,1] ]
    # note the difference between array and matrix (it is the multiplication)
    Rmirror = np.matrix( Rmirror )

    return(Rmirror)

def visualize_wingpath_chord( fname, psi=0.0, gamma=0.0, beta=0.0, eta_stroke=0.0, equal_axis=True, DrawPath=False,
                             x_pivot_b=np.matrix([0,0,0]), x_body_g=np.matrix([0,0,0]) ):
    # read kinematics data:
    a0_phi, ai_phi, bi_phi, a0_alpha, ai_alpha, bi_alpha, a0_theta, ai_theta, bi_theta = read_kinematics_file( fname )

    # length of wing chord to be drawn. note this is not correlated with the actual
    # wing thickness at some position - it is just a marker.
    wing_chord = 0.1
    # create time vector:
    time = np.linspace( start=0.0, stop=1.0, endpoint=False, num=40)
    # wing tip in wing coordinate system
    x_tip_w = vct([0.0, 1.0, 0.0])
    x_le_w  = vct([ 0.5*wing_chord,1.0,0.0])
    x_te_w  = vct([-0.5*wing_chord,1.0,0.0])

    # body transformation matrix
    M_body = Rx(deg2rad(psi))*Ry(deg2rad(beta))*Rz(deg2rad(gamma))

    # rotation matrix from body to stroke coordinate system:
    M_stroke_l = Ry(deg2rad(eta_stroke))

    plt.figure( figsize=(cm2inch(12), cm2inch(7)) )
    plt.subplots_adjust(bottom=0.16, left=0.14)
    ax = plt.gca() # we need that to draw lines...

    # array of color (note normalization to 1 for query values)
    colors = plt.cm.jet( (np.arange(time.size) / time.size) )

    # step 1: draw the symbols for the wing section for some time steps
    for i in range(time.size):
        alpha_l = Fserieseval(a0_alpha, ai_alpha, bi_alpha, time[i])
        phi_l   = Fserieseval(a0_phi, ai_phi, bi_phi, time[i])
        theta_l = Fserieseval(a0_theta, ai_theta, bi_theta, time[i])

        # rotation matrix from body to wing coordinate system
        M_wing_l = Ry(deg2rad(alpha_l))*Rz(deg2rad(theta_l))*Rx(deg2rad(phi_l))*M_stroke_l

        # convert wing points to global coordinate system
        x_tip_g = np.transpose(M_body) * ( np.transpose(M_wing_l) * x_tip_w + x_pivot_b ) + x_body_g
        x_le_g  = np.transpose(M_body) * ( np.transpose(M_wing_l) * x_le_w  + x_pivot_b ) + x_body_g
        x_te_g  = np.transpose(M_body) * ( np.transpose(M_wing_l) * x_te_w  + x_pivot_b ) + x_body_g

        # the wing chord changes in length, as the wing moves and is oriented differently
        # note if the wing is perpendicular, it is invisible
        # so this vector goes from leading to trailing edge:
        e_chord = x_te_g - x_le_g
        e_chord[1] = [0.0]
        # normalize it to have the right length
        e_chord = e_chord * (wing_chord) / (np.linalg.norm(e_chord))
        # mark leading edge with a marker
        plt.plot( x_le_g[0], x_le_g[2], marker='o', color=colors[i], markersize=4 )

        # draw wing chord
        # see comment above why we do not draw the line TE-LE
        l = matplotlib.lines.Line2D( [x_le_g[0], x_le_g[0]+e_chord[0]], [x_le_g[2],x_le_g[2]+e_chord[2]], Linestyle='-', color=colors[i])
        ax.add_line(l)


    # step 2: draw the path of the wingtip
    if DrawPath:
        # refined time vector for drawing the wingtip path
        time = np.linspace( start=0.0, stop=1.0, endpoint=False, num=1000)
        xpath = time.copy()
        zpath = time.copy()


        for i in range(time.size):
            alpha_l = Fserieseval(a0_alpha, ai_alpha, bi_alpha, time[i])
            phi_l   = Fserieseval(a0_phi, ai_phi, bi_phi, time[i])
            theta_l = Fserieseval(a0_theta, ai_theta, bi_theta, time[i])

            # rotation matrix from body to wing coordinate system
            M_wing_l = Ry(deg2rad(alpha_l))*Rz(deg2rad(theta_l))*Rx(deg2rad(phi_l))*M_stroke_l

            # convert wing points to global coordinate system
            x_tip_g = np.transpose(M_body) * ( np.transpose(M_wing_l) * x_tip_w + x_pivot_b ) + x_body_g

            xpath[i] = (x_tip_g[0])
            zpath[i] = (x_tip_g[2])
        plt.plot( xpath, zpath, linestyle='--', color='k', linewidth=1.0 )


    # Draw stroke plane as a dashed line
    M_stroke = Ry( deg2rad(eta_stroke) )
    # we draw the line between [0,0,-1] and [0,0,1] in the stroke system
    xs1 = vct([0.0, 0.0, +1.0])
    xs2 = vct([0.0, 0.0, -1.0])
    # bring these points back to the global system
    x1 = np.transpose(M_body) * ( np.transpose(M_stroke)*xs1 + x_pivot_b ) + x_body_g
    x2 = np.transpose(M_body) * ( np.transpose(M_stroke)*xs2 + x_pivot_b ) + x_body_g

    # remember we're in the x-z plane
    l = matplotlib.lines.Line2D( [x1[0],x2[0]], [x1[2],x2[2]], color='k', linewidth=1.0, linestyle='-.')
    ax.add_line(l)

    # this is a manually set size, which should be the same as what is produced by visualize kinematics file
    plt.gcf().set_size_inches([4.71, 2.75] )
    if equal_axis:
        axis_equal_keepbox( plt.gcf(), plt.gca() )

    # annotate plot
    plt.rcParams["text.usetex"] = True
    plt.xlabel('$x^{(g)}$')
    plt.ylabel('$z^{(g)}$')

    # modify ticks in matlab-style.
    ax = plt.gca()
    ax.tick_params( which='both', direction='in', top=True, right=True )
    plt.savefig( fname.replace('.ini','_path.pdf'), format='pdf' )
    plt.savefig( fname.replace('.ini','_path.png'), format='png', dpi=300 )



def interp_matrix( d, time_new ):
    # interpolate matrix d using given time vector
    nt_this, ncols = d.shape
    nt_new = len(time_new)

    # allocate target array
    d2 = np.zeros( [nt_new, ncols] )
    # copy time vector
    d2[:,0] = time_new

    # loop over columns and interpolate
    for i in range(1,ncols):
        # interpolate this column i to equidistant data
        d2[:,i] = np.interp( time_new, d[:,0], d[:,i] )#, right=0.0 )

    return d2



def get_dset_name( fname ):
    from os.path import basename
    dset_name = basename(fname)
    dset_name = dset_name[0:dset_name.find('_')]

    return dset_name


def get_timestamp_name( fname ):
    from os.path import basename
    dset_name = basename(fname)
    dset_name = dset_name[dset_name.find('_')+1:dset_name.find('.')]

    return dset_name



def indicate_strokes( tstart=None, ifig=None, tstroke=1.0, ax=None ):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    if ifig == None:
        # get current axis
        if ax is None:
            ax = plt.gca() # we need that to draw rectangles...
    else:
        if ax is None:
            plt.figure(ifig)
            ax = plt.gca()


    # initialize empty list of rectangles
    rects = []

    # current axes extends
    t1, t2 = ax.get_xbound()
    y1, y2 = ax.get_ybound()


    # will there be any strokes at all?
    if t2-t1 < tstroke:
        print('warning: no complete stroke present, not returning any averages')

    if t1 - np.round(t1) >= 1e-3:
        print('warning: data does not start at full stroke (tstart=%f)' % t1)

    if tstart is None:
        # go in entire strokes
        while t1+tstroke <= t2:
            # begin of this stroke
            tbegin = t1
            # end of this stroke
            tend = t1 + tstroke / 2.0
            # iterate
            t1 = tbegin + tstroke
            # create actual rectangle
            r = Rectangle( [tbegin,y1], tend-tbegin, y2-y1, fill=True)
            rects.append(r)
    else:
        for tbegin in tstart:
            # end of this stroke
            tend = tbegin + tstroke / 2.0
            # create actual rectangle
            r = Rectangle( [tbegin,y1], tend-tbegin, y2-y1, fill=True)
            rects.append(r)


    # Create patch collection with specified colour/alpha
    color = [0.85,0.85,0.85]
    pc = PatchCollection(rects, facecolor=color, alpha=1.0, edgecolor=color, zorder=-2)

    # Add collection to axes
    ax.add_collection(pc)


def make_white_plot( ax ):
    # for the poster, make a couple of changes: white font, white lines, all transparent.
    legend = ax.legend()
    if not legend is None:
        frame = legend.get_frame()
        frame.set_alpha(0.0)
        # set text color to white for all entries
        for label in legend.get_texts():
            label.set_color('w')


    ax.xaxis.label.set_color('w')
    ax.tick_params(axis='x', colors='w')

    ax.yaxis.label.set_color('w')
    ax.tick_params(axis='y', colors='w')

    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['right'].set_color('w')

    ax.tick_params( which='both', direction='in', top=True, right=True, color='w' )




def hit_analysis():
    import glob

    # Take all analyis files from ./flusi --turbulence-analysis and put all
    # data in one csv file

    showed_header=False

    fid = open('complete_analysis.csv', 'w')

    for file in sorted( glob.glob('analysis_*.txt') ):
        print(file)
        # read entire file to list of lines
        with open (file, "r") as myfile:
            data = myfile.readlines()

        # remove header lines
        del data[0:2+1]
        del data[1]

        # fetch viscosity from remaining header
        header = data[0].split()
        nu = float(header[7])

        if not showed_header:
            showed_header = True
            # write header
            fid.write('name;')
            for line in data[1:]:
                fid.write("%15s; " % (line[20:-1]))
            fid.write('viscosity;\n')

        # read remaining data items
        fid.write("%s; " % (file))
        for line in data[1:]:
            fid.write("%e; " % (float(line[0:20])))

        fid.write("%e;" % (nu) )
        fid.write("\n")

    fid.close()


def plot_a_col( data, col ):
    D = stroke_average_matrix( data, tstroke=0.5 )
    plt.plot( data[:,0], data[:,col] )
    plt.plot( D[:,0], D[:,col], linestyle='None', marker='o', markerfacecolor='none', color=h[-1].get_color())



def forces_g2b( data, kinematics ):
    """ Transform timeseries data (forces.t) to body system defined by kinematics.t
    """

    # they are not necessarily at the same times t -> interpolation
    time = data[:,0]

    # interpolate kinematics to data time vector
    k = interp_matrix( kinematics, time )
    psi = k[:,4]
    beta  = k[:,5]
    gamma  = k[:,6]

    data_new = data.copy()

    for it in range(data.shape[0]):
        M_body = Rx(psi[it])*Ry(beta[it])*Rz(gamma[it])
        # usual forces
        Fg = vct( [data[it,1],data[it,2],data[it,3]] )
        Mg = vct( [data[it,7],data[it,8],data[it,9]] )
        Fb = M_body*Fg
        Mb = M_body*Mg
        data_new[it,1:3+1] = Fb.transpose()
        data_new[it,7:9+1] = Mb.transpose()
        # unsteady corrections (rarely used)
        Fg = vct( [data[it,4],data[it,5],data[it,6]] )
        Mg = vct( [data[it,10],data[it,11],data[it,12]] )
        Fb = M_body*Fg
        Mb = M_body*Mg
        data_new[it,4:6+1] = Fb.transpose()
        data_new[it,10:12+1] = Mb.transpose()

    return(data_new)


def read_flusi_HDF5( fname ):
    import h5py

    f = h5py.File(fname, 'r')


    # list all hdf5 datasets in the file - usually, we expect
    # to find only one.
    datasets = list(f.keys())
    # if we find more than one dset we warn that this is unusual
    if (len(datasets) != 1):
        print("we found more than one dset in the file (problemo)"+fname)

    else:
        # as there should be only one, this should be our dataset:
        dset_name = datasets[0]

        # get the dataset handle
        dset_id = f.get(dset_name)

        # from the dset handle, read the attributes
        time = dset_id.attrs.get('time')
        res = dset_id.attrs.get('nxyz')
        box = dset_id.attrs.get('domain_size')
        origin = dset_id.attrs.get('origin')
        if origin is None:
            origin = np.array([0,0,0])

        b = f[dset_name][:]
        data = np.array(b, dtype=float)
        # its a funny flusi convention that we have to swap axes here, and I
        # never understood why it is this way.
        data = np.swapaxes(data, 0, 2)

        if (np.max(res-data.shape)>0):
            print('WARNING!!!!!!')
            print('read_flusi_HDF5: array dimensions look funny')

        f.close()

    print("We read FLUSI file %s at time=%f" % (fname, time) )

    return time, box, origin, data



def write_flusi_HDF5( fname, time, box, data, viscosity=0.0, origin=np.array([0.0,0.0,0.0]) ):
    import h5py

    dset_name = get_dset_name( fname )

    if len(data.shape)==3:
        #3d data
        nx, ny, nz = data.shape
        print( "Writing to file=%s dset=%s max=%e min=%e size=%i %i %i " % (fname, dset_name, np.max(data), np.min(data), nx,ny,nz) )
        # i dont really know why, but there is a messup in fortran vs c ordering, so here we have to swap
        # axis
        data = np.swapaxes(data, 0, 2)
        nxyz = np.array([nx,ny,nz])
    else:
        #2d data
        nx, ny = data.shape
        print( "Writing to file=%s dset=%s max=%e min=%e size=%i %i" % (fname, dset_name, np.max(data), np.min(data), nx,ny) )
        data = np.swapaxes(data, 0, 1)
        nxyz = np.array([nx,ny])

    fid = h5py.File( fname, 'w')

    fid.create_dataset( dset_name, data=data, dtype=np.float32 )#, shape=data.shape[::-1] )
    fid.close()

    fid = h5py.File(fname,'a')
    dset_id = fid.get( dset_name )
    dset_id.attrs.create('time', time)
    dset_id.attrs.create('viscosity', viscosity)
    dset_id.attrs.create('domain_size', box )
    dset_id.attrs.create('origin', origin )
    dset_id.attrs.create('nxyz', nxyz )

    fid.close()



def load_image( infilename ):
    from PIL import Image
    import numpy as np

    img = Image.open( infilename )
    img.load()
    data = np.asarray( img , dtype=np.float32 )

    return data


def tiff2hdf( dir, outfile, dx=1, origin=np.array([0,0,0]) ):
    print('******************************')
    print('* tiff2hdf                   *')
    print('******************************')

    # first, get the list of tiff files to process
    files = glob.glob( dir+'/*.tif*' )
    files.sort()
    print("Converting dir %s (%i files) to %s" % (dir, len(files), outfile))

    nz = len(files)

    if nz>0:
        # read in first file to get the resolution
        data = load_image(files[0])
        nx, ny = data.shape

        # allocate (single precision) data
        data = np.zeros([nx,ny,nz], dtype=np.float32)

        # it is useful to now use the entire array, so python can crash here if
        # out of memory, and not after waiting a long time...
        data = data + 1.0

        print( "Data dimension is %i %i %i" % (nx,ny,nz))

        for i in range(nz):
            sheet = load_image( files[i] )
            data[:,:,i] = sheet.copy()

        write_flusi_HDF5( outfile, 0.0, [float(nx)*dx,float(ny)*dx,float(nz)*dx], data, viscosity=0.0, origin=origin )



def suzuki_error( filename, component=None ):
    """compute the error for suzukis test case"""

    reference_file = '/home/engels/Documents/Research/Insects/3D/projects/suzuki_validation/Digitize/lift_FV_new.txt'

    import insect_tools

    # read reference data, digitized from suzukis paper
    dref = insect_tools.load_t_file( reference_file )

    # read actual data
    data = insect_tools.load_t_file( filename )

    import numpy as np

    # interpolate actual data on ref data points
    data = insect_tools.interp_matrix( data, dref[:,0] )


    # Suzuki et al. eq. in appendix B.5.2
    L = 0.833
    c = 0.4167
    rho = 1
    utip = 2*np.pi*(80*np.pi/180)*(0.1667+0.833)/1
    fcoef = 0.5*rho*(utip**2)*(L*c)

    data[:,3] /= fcoef



    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(data[:,0], data[:,3], dref[:,0], dref[:,1])
    plt.title(filename)

    err = np.trapz( abs(data[:,3]-dref[:,1]), x=data[:,0] ) / np.trapz( abs(dref[:,1]), x=data[:,0] )

    return err