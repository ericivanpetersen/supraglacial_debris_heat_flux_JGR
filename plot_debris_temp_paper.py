import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
import matplotlib.dates as mdates
import matplotlib.animation as ani
from aws_tools import *
import os, sys

### Plotting font size settings:
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.rcParams["font.family"] = "verdana"        

def neighbor_mean(data):
    """ Calculates the mean between neighboring 
    data series elements, assuming data is a 
    pandas Series."""

    output = (data + data.shift(-1))/2

    return output

def sensor_neighbor_mean(data):
    output = (data + data.shift(-1, axis=1))/2

    return output

class debris_temp_profile(object):

    """
    Class to store data and metadata on a 
    debris temperature profile. Also methods 
    for plotting and analyzing that data.

    2021_08_02: Am cleaning up this class, removing
    un-used methods & appendices (these will be preserved in
    "plot_debris_temp_archive_2021_08_02.py").
    Deleted the following methods:
        "estimate_G"
        "plot_EB_compare"
    'estimate_Qd' is slightly alterred with an addition from
        'plot_EB_compare' to be kept.
    """

    def __init__(self, directory, period=None, mask=None, smooth=6, downsample=None, plotdir=None):
        """
        Initiation of Class, Starts with:
        Profile info file in directory, contains names of
        #   temperature sensor files and their associated 
        #   depth in the profile, in order from top to bottom.
            If the sensor dummy name "base" is provided, it is 
            assumed to indicate the depth of the debris-ice 
            interface, where there is no sensor but the value of 
            temperature is assumed to be a constant 0 C.
        """
        #Plot saving directory option:
        if plotdir is not None:
            self.pd=1
            self.plotdir=plotdir
        else:
            self.pd=0
        # Read & Check Profile Info file:
        print("Reading Debris Temperature Profile From {}".format(directory))
        index_file = directory + '/profile_info.csv'
        index = pd.read_csv(index_file)
        print("Sensors:")
        print(index)

        # Start reading in data
        data = pd.DataFrame()
        sensor_names = index['Name']
        sensor_depth = index['Depth']
        for name in sensor_names:
            if name == "Base":
                data[name] = data[sensor_names[0]] * 0
            else:
                hobo = read_hobo(directory + str(name) +'.csv')
                data[name] = hobo[hobo.columns[2]]
        
        # Downsample if desired:
        if downsample is not None:
            data = downsample_AWS(data, downsample)
        
        # Define data for class:
        self.data = data
        self.sensor_names = sensor_names
        self.sensor_depth = sensor_depth

        self.calc_derivatives(period, mask, smooth)

        return

    def calc_derivatives(self, period, mask, smooth):

        self.datetime = self.data.index
        self.dt = (self.datetime[1] - self.datetime[0]).total_seconds()
        self.num_sensors = len(self.sensor_names)
        self.sensor_depth.rename('Depth') # Needed for linear fit routine later
        # Create Sensor depth string for plotting:
        depth_str = []
        for n in range(self.num_sensors):
            depth_str.append('{} cm'.format(self.sensor_depth[n]))
        self.depth_str = depth_str
        #Determine sensor spacing (will be needed for differentiation):
        self.spacings = self.sensor_depth.diff()/100 # Spacing in meters between sensors
        self.dz2 = neighbor_mean(self.spacings) # dz2 for second spatial derivative

        # Clip data if desired:
        if period is not None:
            self.clip_data(period)
        
        # Mask data if desired:
        if mask is not None:
            for sensor_name in self.sensor_names:
                self.data[sensor_name][mask[0]:mask[1]] = None
        
        # Some final metadata:
        self.length = self.data.index.size
        self.median_temp = pd.Series(data=np.median(self.data,axis=0), name='Median_Temp')

        print("Time Step = {} seconds".format(self.dt))
        print("Number of Records = {}".format(self.length))

        # Calculate Differentials:
        if smooth is not None: 
            smooth_data = self.data.rolling(smooth).mean() # Produce smoothed data:
        else: smooth_data = self.data
        # Produce appropriately sized dz & dz2 matrices:
        spacing_mat = np.ones((self.length, 1))*self.spacings.to_numpy()
        self.dz2_mat = np.ones((self.length, 1))*self.dz2.to_numpy()
        # Calculate differentials:
        self.dTdt = smooth_data.diff() / self.dt # dT/dt in K/sec
        self.dTdz = smooth_data.diff(axis=1) / spacing_mat # dT/dz in K/meter
        self.dTdz_z = self.sensor_depth[:-1].to_numpy() + self.spacings[1:].to_numpy()*50
        self.dTdz2 = -self.dTdz.diff(axis=1,periods=-1) / self.dz2_mat # dT/dz2 in K/m2
        Qstr = []
        Qcncstr = []
        for z_value in self.dTdz_z:
            Qstr.append('{:.1f} cm'.format(z_value))
            Qcncstr.append(r'$Q_{c}$'+', {:.1f} cm'.format(z_value))
            Qcncstr.append(r'$Q_{nc}$'+', {:.1f} cm'.format(z_value))
        self.Qstr = Qstr
        self.Qcncstr = Qcncstr

        return
    
    def clip_data(self, period):
        self.data = self.data[period[0]:period[-1]]
        self.datetime = self.data.index 

    def plot_debris_temp(self, leg=None, period=None, dt="day", animate=False, show_plot=True):
        """ Plots both a time series plot for the desired time
        period, as well as a box-whisker plot for temperature
        statistics as a function of depth."""

        if dt=="month":
            major_tick = mdates.MonthLocator()
            dt_format = mdates.DateFormatter('%B')
        if dt=="day":
            dt_format = mdates.DateFormatter('%b-%d')
        
        if leg is None:
            leg = self.depth_str

        # Timeseries plot:
        plt.figure(figsize=(13,5))
        ax = plt.subplot()
        if period is not None:
            plot_timeseries(self.data, self.sensor_names, legend=leg, ylbl='T '+r'($^\circ$C)', period=period)
        else: 
            plot_timeseries(self.data, self.sensor_names, legend=leg, ylbl='T '+r'($^\circ$C)')
        if dt=="month":
            ax.xaxis.set_major_locator(major_tick)
        ax.xaxis.set_major_formatter(dt_format)
        plt.grid(True,axis='x')
        plt.ylim([-5,30])

        if self.plotdir is not None:
            plt.savefig(self.plotdir+'temp_timeseries.png', bbox_inches='tight')

        # Vertical Box-Whiskers Plot:
        plt.figure(dpi=72, figsize=(6.5,5))
        ax = plt.subplot()
        plt.boxplot(self.data.to_numpy(),positions=-self.sensor_depth,notch=True,vert=False,widths=2,manage_ticks=False)
        plt.plot(self.median_temp, -self.sensor_depth,'k-')
        plt.ylim(top=0,bottom=-np.max(self.sensor_depth)-4)
        plt.xlim([-5,30])
        plt.ylabel('Depth (cm)')
        plt.xlabel('T '+r'($^\circ$C)')

        # Calculate debris temperature gradient:
        grad_mod = transfer_function(-self.sensor_depth, self.median_temp)
        grad_mod.LinearModel()
        txt0 = "N = {}".format(self.length)
        txt1 = r'R$^2$'+' = {:.2f}'.format(grad_mod.R2)
        txt2 = 'Gradient = {:.2f}'.format(grad_mod.model.coef_[0][0]) + r'$^\circ$C/cm'
        txt3 = "Surf. Temp (Fit) = {:.1f}".format(grad_mod.model.intercept_[0]) + r'$^\circ$C '
        plt.text(0.4,0.28,txt0,transform=ax.transAxes,fontsize=16)
        plt.text(0.4,0.21,txt1,transform=ax.transAxes,fontsize=16)
        plt.text(0.4,0.14,txt2,transform=ax.transAxes,fontsize=16)
        plt.text(0.4,0.07,txt3,transform=ax.transAxes,fontsize=16)
        print(txt1)
        print(txt2)
        print(txt3)

        if self.plotdir is not None:
            plt.savefig(self.plotdir+'sensor_stats.png', bbox_inches='tight')

        if show_plot:
            plt.show()

    def estimate_k_from_melt(self, melt_period, cum_melt, k_meas, show_plot=True):
        """
        Takes as input the melt period and cumulative melt to consider,
        then sums dTdz for that period and, dividing by cumulative melt,
        thus determines the effective thermal conductivity of the debris.
        
        Melt period: list of datetime strings
        cum_melt: cumulative melt in meters w.e. 
        """

        mpi = ((self.datetime > melt_period[0]) & (self.datetime < melt_period[1])) # melt period index

        dTdz_ice = self.dTdz[self.sensor_names[self.num_sensors-1]]
        dTdz_ice = dTdz_ice[mpi]
        cum_dTdz = np.cumsum(dTdz_ice)[-1]

        rho_i = 900
        L_m = 334000

        k = rho_i * L_m * cum_melt / (cum_dTdz * self.dt)
        txt1 = r"$k_e$"+" = {:.3f} W/(K m)".format(-k)
        txt2 = "Temp Fit, k = {:.3f} W/(K m)".format(k_meas)
        print(txt1)

        Melt = k * dTdz_ice * self.dt / (rho_i * L_m) * 0.9
        Cum_Melt_timeseries = np.cumsum(Melt)

        Melt_mod = -k_meas * dTdz_ice * self.dt / (rho_i * L_m) *0.9
        Cum_Melt_mod_timeseries = np.cumsum(Melt_mod)

        if show_plot:
            dt_format = mdates.DateFormatter('%b-%d')
            #plt.figure()
            # ax = plt.subplot(211)
            # plt.plot(dTdz_ice)
            # plt.ylabel(r'$\partial T/\partial z$ (K/m)')
            # ax.xaxis.set_major_formatter(dt_format)

            plt.figure(figsize=(12,3), dpi=100)
            ax=plt.subplot(111)
            plt.plot(Melt * 3600 * 24 / self.dt * 100, linestyle='--')
            #plt.plot(Melt_mod * 3600 * 24/ self.dt * 100, linestyle='--')
            plt.legend([txt1], loc='upper left')
            plt.ylabel('Melt Rate\n(cm w.e./day)')
            ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            ax2 = ax.twinx()
            ax2.plot(self.datetime[mpi], Cum_Melt_timeseries * 100, c=colors[0])
            #ax2.plot(self.datetime[mpi], Cum_Melt_mod_timeseries * 100, c=colors[1])
            ax2.plot(self.datetime[mpi][-1], cum_melt * 90, 'k*',markersize=12)
            
            plt.ylabel('Cum. Melt (cm w.e.)')
            # t = plt.text(0.02,0.85,txt,transform=ax.transAxes, backgroundcolor='0.7',alpha=1)
            # t.set_bbox(dict(facecolor='0.75',alpha=0.8,edgecolor='k'))
            ax.xaxis.set_major_formatter(dt_format)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))

            if self.pd:
                plt.savefig(self.plotdir+'melt_rates.png', bbox_inches='tight')

        return k

    def estimate_ke_Q(self, dt = 5, period=None, smooth=6, saveas=None, display_hr=False, animate=False, ke_ideal=None, hours_of_day=None, show_diff_ts=False, rho=1800, c=900):
        """ Calculates dT/dt and d2T/dz2, plots them, and performs 
        a linear regression to estimate the best fit effective thermal 
        diffusivity ke.
        Then uses the effective thermal diffusivity to calculate the residual in
        the model, here referred to as "dqdz," and integrates up from the debris-ice
        interface to estimate Q_nc.
        hours_of_day = [morning_hour, evening_hour], where the desired 
            sampling time is overnight between the evening & morning hours
        """

        # Pull differentials out of the program:
        dTdt = self.dTdt
        dTdz2 = self.dTdz2

        # Initiate dqdz, Q_nc, Q_c:
        dqdz = self.dTdz2 * 0
        Q_nc = self.dTdz * 0
        Q_c = self.dTdz * 0

        hr = self.datetime.hour # Calculate Hour of Day for Plotting purposes
        # Mask for data into hours of most conductive behavior if provided:
        if hours_of_day is not None:
            hi = ( (hr>= hours_of_day[0]) | (hr<= hours_of_day[1])) # hi = mask to highlight desired data
        else:
            hi = (hr>=0)
        # Initiate Loop through middle sensors:
        NN = len(self.sensor_names)-1
        R2 = np.zeros((NN+1))
        ke = np.zeros((NN+1))
        ke_z2 = np.zeros((NN+1))
        x_mod = {}
        y_mod = {}
        for nn in range(1,NN):
            # Perform linear regression on dT/dt vs d2T/dz2:
            model = transfer_function(dTdz2[self.sensor_names[nn]][hi], dTdt[self.sensor_names[nn]][hi]*1000) # performing linear regression on mK/sec
            model.LinearModel(intercept=False)
            x_mod[self.sensor_names[nn]] = [[dTdz2[self.sensor_names[nn]].min()], [dTdz2[self.sensor_names[nn]].max()]]
            y_mod[self.sensor_names[nn]] = model.model.predict(x_mod[self.sensor_names[nn]])

            # Determine dqdz:
            dqdz[self.sensor_names[nn]] = rho * c * (self.dTdt[self.sensor_names[nn]] - model.model.coef_[0][0]/1000 * self.dTdz2[self.sensor_names[nn]]) / 100 # in [W/m2]/[cm]
            ke[nn] = model.model.coef_[0][0]/1000

            # Plot Timeseries:
            if show_diff_ts:
                plt.figure()
                plt.subplot(211)
                plt.plot(dTdz2[self.sensor_names[nn]])
                plt.title('Depth = {} cm'.format(self.sensor_depth[nn]))
                plt.ylabel(r'$\partial^2 T / \partial z^2$'+' '+r'(K/m$^2$)')
                plt.subplot(212)
                plt.plot(dTdt[self.sensor_names[nn]])
                plt.ylabel(r'$\partial T / \partial t$'+' (mK/s)')

            # Plot Scatterplot:
            leg_str = ['Fit', 'All Data']
            fig,ax = plt.subplots()
            if display_hr:
                plt.scatter(dTdz2[self.sensor_names[nn]], dTdt[self.sensor_names[nn]]*1000, c=hr, s=5)
                plt.colorbar()
            else:
                plt.scatter(dTdz2[self.sensor_names[nn]], dTdt[self.sensor_names[nn]]*1000, s=5)
            if hours_of_day is not None:
                plt.scatter(dTdz2[self.sensor_names[nn]][hi], dTdt[self.sensor_names[nn]][hi]*1000, c='k', marker='x', alpha=0.3)
                leg_str.append('Fit Data')
            plt.plot(x_mod[self.sensor_names[nn]],y_mod[self.sensor_names[nn]],'r--')
            plt.title('Depth = {:.1f} cm'.format(self.sensor_depth[nn]))
            #plt.xlim(model.x_lim1, model.x_lim2)
            #plt.ylim(model.y_lim1, model.y_lim2)
            plt.xlabel(r'$\partial^2 T / \partial z^2$'+' '+r'(K/m$^2$)')
            plt.ylabel(r'$\partial T / \partial t$'+' (mK/s)')
            txt1 = r'R$^2$'+' = {:.3f}'.format(model.R2)
            txt2 = r'$\kappa_e$'+' = {:.2e} '.format(model.model.coef_[0][0]/1000) + r'(m$^2$ s$^{-1}$)'
            plt.text(0.02,0.9,txt2,transform=ax.transAxes)
            plt.text(0.02,0.8,txt1,transform=ax.transAxes)
            plt.legend(leg_str, loc='lower right')

            if ke_ideal is not None:
                y_mod_ideal = np.array(x_mod) * ke_ideal * 1000
                plt.plot(x_mod, y_mod_ideal, 'k')
                txt3 = r'Ideal $\kappa_e$ = '+'{:.3e}'.format(ke_ideal)
                plt.text(0.02,0.85,txt3,transform=ax.transAxes)
                plt.legend(['Fit','Ideal','Data'], loc='lower right')
            if self.plotdir is not None:
                plt.savefig(self.plotdir+'diff_scatter_'+str(nn)+'.png',bbox_inches='tight')

        # Calculate Q_c:
        for nn in range(1,NN+1):
            if nn==1:
                ke_z2[nn] = ke[nn]
            elif nn==NN:
                ke_z2[nn] = ke[nn-1]
            else:
                ke_z2[nn] = (ke[nn] + ke[nn-1])/2

            Q_c[self.sensor_names[nn]] = ke_z2[nn] * rho * c * -self.dTdz[self.sensor_names[nn]]

        # Determine Q_nc (assuming Q_nc at ice-debris boundary = 0)
        Q_nc = (dqdz)  * self.dz2_mat * 100
        Q_nc[self.sensor_names[self.num_sensors-1]] = self.data[self.sensor_names[self.num_sensors-1]] * 0
        N = self.num_sensors
        for n in range(2, N-1):
            Q_nc[self.sensor_names[N-n-1]] = Q_nc[self.sensor_names[N-n-1]] + Q_nc[self.sensor_names[N-n]]

        Q_nc_c = Q_nc+Q_c # sum of Q, conductive & non-conductive
        # Animate the plot, if desired:
        if animate:
            fig = plt.figure(figsize=(12,8))
            plt.subplots_adjust(wspace=0.25, hspace=0.25)
            # Subplots for T, dqdz, Qnc:
            sub1 = fig.add_subplot(1,5,1)
            sub2 = fig.add_subplot(1,5,2)
            sub3 = fig.add_subplot(1,5,3)
            # Subplots for differentials:
            subs = {}
            for n in range(1,NN):
                subs[self.sensor_names[n]] = fig.add_subplot(NN-1,3,(n-1)*3+3)
            # Animated 

            def ani_frame_ke_dqdz_q(i):
                sub1.clear()
                sub1.plot(self.data.loc[self.datetime[i],:], -self.sensor_depth, 'k*-')
                sub1.set_xlabel('Temp ('+r'$^\circ$C)')
                sub1.set_ylabel('Depth (cm)')
                sub1.set_xlim([0, 30])
                sub1.set_ylim([-np.max(self.sensor_depth)-4, 0])

                sub2.clear()
                sub2.plot([0, 0], [-np.max(self.sensor_depth)-4, 0], 'k--',alpha=0.5)
                sub2.plot(dqdz[self.sensor_names[1:len(self.sensor_names)-1]].loc[self.datetime[i],:], -self.sensor_depth[1:len(self.sensor_names)-1], 'k*-')
                sub2.set_xlim(np.nanmin(dqdz), np.nanmax(dqdz))
                sub2.set_ylim([-np.max(self.sensor_depth)-4, 0])
                sub2.set_xlabel(r'$\partial Q_{nc} / \partial z$ $(W/m^2)/cm$')

                sub3.clear()
                sub3.plot([0, 0], [-np.max(self.sensor_depth)-4, 0], 'k--',alpha=0.5)
                sub3.plot(Q_c[self.sensor_names[1:len(self.sensor_names)]].loc[self.datetime[i],:], -self.dTdz_z, 'b*--')
                sub3.plot(Q_nc[self.sensor_names[1:len(self.sensor_names)]].loc[self.datetime[i],:], -self.dTdz_z, 'g*--')
                sub3.plot(Q_nc_c[self.sensor_names[1:len(self.sensor_names)]].loc[self.datetime[i],:], -self.dTdz_z, 'k*--')
                sub3.legend(['0',r'$Q_{c}$',r'$Q_{nc}$',r'$\Sigma Q$'])
                sub3.set_ylim([-np.max(self.sensor_depth)-4, 0])
                sub3.set_xlim(np.nanmin(Q_nc), np.nanmax(Q_nc_c))
                sub3.set_xlabel(r'$Q_{nc}$ $(W/m^2)$')

                for nn in range(1,NN):
                    sub = subs[self.sensor_names[nn]]
                    sub.clear()
                    # plot first the full dataset up until this point in color, with transparency
                    sub.plot(x_mod[self.sensor_names[nn]],y_mod[self.sensor_names[nn]],'r--') # linear fit for determination of dqdz
                    sub.scatter(dTdz2[self.sensor_names[nn]][:i], dTdt[self.sensor_names[nn]][:i]*1000, c=hr[:i], marker='.', s=5, alpha=0.7)
                    # plot the dataset of the day so far in dark grey:
                    idate = self.datetime[i]
                    iday = datetime(idate.year, idate.month, idate.day)
                    ind = (self.datetime >= iday) & (self.datetime <= idate)
                    sub.scatter(dTdz2[self.sensor_names[nn]][iday:idate], dTdt[self.sensor_names[nn]][iday:idate]*1000, c='k', marker='.', s=5, alpha=0.8)
                    # plot the current datapoint in bold, black:
                    sub.scatter(dTdz2[self.sensor_names[nn]][i], dTdt[self.sensor_names[nn]][i]*1000, c='k', marker = 'o')
                    # ensure consistent framing:
                    sub.set_xlabel(r'$\partial^2 T / \partial z^2$'+' '+r'(K/m$^2$)')
                    sub.set_ylabel(r'$\partial T / \partial t$'+' (mK/s)')
                    #sub2.set_xlim(model.x_lim1.to_numpy(), model.x_lim2.to_numpy())
                    sub.set_xlim(dTdz2[self.sensor_names[nn]].min(), dTdz2[self.sensor_names[nn]].max())
                    sub.set_ylim(dTdt[self.sensor_names[nn]].min()*1000, dTdt[self.sensor_names[nn]].max()*1000)
                    txt = self.depth_str[nn]
                    sub.text(0.05,0.95,txt,transform=sub.transAxes)
                plt.suptitle(idate)
            
            animator = ani.FuncAnimation(fig, ani_frame_ke_dqdz_q, interval=16, save_count=np.size(self.datetime))
            if pd:
                animator.save(self.plotdir+'debris_temp_animation.mp4')
            else:
                animator.save(r'/Users/ericpetersen/Desktop/debris_temp_animation.mp4')

        #plt.show()
        self.dqdz = dqdz
        self.Q_nc = Q_nc
        self.Q_c = Q_c
        self.Q_nc_c = Q_nc+Q_c
        self.ke = ke
        self.ke_z2 = ke_z2
        #self.Q_nc_c[(self.Q_c<15)] = np.nan

        return

    def estimate_dqdz(self, kappa_e, rho, c, show_plot=False, Q_0=0, period=None):
        """ Routine to estimate dqdz = residual dT/dt,
        assuming an input kappa_e. 
        
        NOTE: This routine requires "estimate_ke_Q" to have
        been run so that dT/dt and d2T/dz2 are saved in 
        the program. 
        
        Q_0 = the boundary condition on non-conductive heat
        fluxes experienced at the ice-debris interface.
        
        "estimate_ke_Q" currently derives dqdz/dqdz and q_nc itself, 
        however this method allows more control over input kappa_e values
        as well as prescribing the Q_0 at the debris-ice interface."""

        dqdz = rho * c * (self.dTdt - kappa_e * self.dTdz2) / 100 # in [W/m2]/[cm]

        Q_nc = dqdz * self.dz2_mat * 100
        N = self.num_sensors
        for n in range(2, N-1):
            Q_nc[self.sensor_names[N-n-1]] = Q_nc[self.sensor_names[N-n-1]] + Q_nc[self.sensor_names[N-n]]

        if show_plot:
            self.plot_dqdz_Q()

        self.dqdz = dqdz
        self.Q_nc = Q_nc

    def plot_dqdz_Q(self, period=None):

        """
        Plots dqdz, Qnc, and Qc stored in the project
        (requires 'estimate_ke_Q' to have been run)
        """

        # set plotdir 
        if self.pd==1:
            plotdir=self.plotdir
        pd = self.pd

        # Time series plots:
        # Determine ylimits:
        Qc_max = self.Q_c.max(skipna=True).max()
        Qnc_max = self.Q_nc.max(skipna=True).max()
        Q_max = np.max([Qc_max,Qnc_max])
        Qc_min = self.Q_c.min(skipna=True).min()
        Qnc_min = self.Q_nc.min(skipna=True).min()
        Q_min = np.min([Qc_min,Qnc_min])
        # dqdz and Q_nc subplots:
        plt.figure(figsize=(12.8,4.8), dpi=100)
        ax=plt.subplot(211)
        plt.plot(self.dqdz[self.sensor_names[1:len(self.sensor_names)-1]])
        plt.legend(self.depth_str[1:len(self.sensor_names)-1], loc='lower right', prop={'size':18})
        plt.plot(self.Q_nc[self.sensor_names[len(self.sensor_names)-1]],c=[0.5,0.5,0.5],linestyle='--') # grey lines at 0
        plt.ylabel(r'$\partial Q_{nc} / \partial z$'+'\n'+r'$(W/m^2)/(cm)$')
        dt_format = mdates.DateFormatter('%b-%d')
        ax.xaxis.set_major_formatter(dt_format)
        ax.tick_params(axis='both', which='major', labelsize=18)
        if period is not None:
            plt.xlim(period)
        
        # ax=plt.subplot(212)
        # plt.plot(self.Q_nc[self.sensor_names[1:len(self.sensor_names)-1]])
        # plt.legend(self.Qstr[:-1], loc='lower right')
        # plt.plot(self.Q_nc[self.sensor_names[len(self.sensor_names)-1]],c=[0.5,0.5,0.5],linestyle='--') # grey lines at 0
        # plt.ylabel(r'$Q_{nc}$ $(W/m^2)$')
        # ax.xaxis.set_major_formatter(dt_format)
        # if period is not None:
        #     plt.xlim(period)
        
        if pd:
            plt.savefig(plotdir+'timeseries_dqdz_Qnc.png', bbox_inches='tight')

        # Q_nc, Q_c, and Sum (Q) Subplots:
        plt.figure(figsize=(12.8,8.7), dpi=100)
        ax=plt.subplot(411)
        plt.plot(self.dqdz[self.sensor_names[1:len(self.sensor_names)-1]])
        plt.legend(self.depth_str[1:len(self.sensor_names)-1], loc='lower right')
        plt.plot(self.Q_nc[self.sensor_names[len(self.sensor_names)-1]],c=[0.5,0.5,0.5],linestyle='--') # grey lines at 0
        plt.ylabel(r'$\partial Q_{nc} / \partial z$'+'\n'+r'$(W/m^2)/(cm)$')
        dt_format = mdates.DateFormatter('%b-%d')
        ax.xaxis.set_major_formatter(dt_format)
        ax.tick_params(axis='both', which='major')
        if period is not None:
            plt.xlim(period)

        ax=plt.subplot(412)
        plt.plot(self.Q_c[self.sensor_names[1:len(self.sensor_names)]])
        plt.legend(self.Qstr[:], loc='lower right')
        plt.plot(self.Q_nc[self.sensor_names[len(self.sensor_names)-1]],c=[0.5,0.5,0.5],linestyle='--') # grey lines at 0
        plt.ylabel(r'$Q_{c}$'+'\n'+r'$(W/m^2)$')
        plt.ylim([Q_min,Q_max])
        ax.xaxis.set_major_formatter(dt_format)
        if period is not None:
            plt.xlim(period)

        ax=plt.subplot(413)
        plt.plot(self.Q_nc[self.sensor_names[1:len(self.sensor_names)-1]])
        plt.legend(self.Qstr[:-1], loc='lower right')
        plt.plot(self.Q_nc[self.sensor_names[len(self.sensor_names)-1]],c=[0.5,0.5,0.5],linestyle='--') # grey lines at 0
        plt.ylabel(r'$Q_{nc}$'+'\n'+r'$(W/m^2)$')
        plt.ylim([Q_min,Q_max])
        ax.xaxis.set_major_formatter(dt_format)
        if period is not None:
            plt.xlim(period)

        ax=plt.subplot(414)
        plt.plot(self.Q_nc_c[self.sensor_names[1:len(self.sensor_names)]])
        plt.legend(self.Qstr[:], loc='lower right')
        plt.plot(self.Q_nc[self.sensor_names[len(self.sensor_names)-1]],c=[0.5,0.5,0.5],linestyle='--') # grey lines at 0
        plt.ylabel(r'$Q_{total}$'+'\n'+r'(W/m$^2$)')
        ax.xaxis.set_major_formatter(dt_format)
        if period is not None:
            plt.xlim(period)
        
        if pd:
            plt.savefig(plotdir+'timeseries_dqdz_Qc_Qnc_Q.png', bbox_inches='tight')

        # Q_c and Q_nc at each level:
        plt.figure(figsize=(12.8,4.8), dpi=100)
        ax=plt.subplot(211)
        for nn in range(1,len(self.sensor_names)-1):
            plt.plot(self.Q_c[self.sensor_names[nn]],c=colors[nn-1])
            plt.plot(self.Q_nc[self.sensor_names[nn]],c=colors[nn-1], linestyle='--')
        #plt.plot(self.Q_c[self.sensor_names[len(self.sensor_names)]])
        ax.xaxis.set_major_formatter(dt_format)
        if period is not None:
            plt.xlim(period)
        plt.legend(self.Qcncstr[:-2], loc="lower right")
        plt.plot(self.Q_nc[self.sensor_names[len(self.sensor_names)-1]],c=[0.5,0.5,0.5],linestyle='--') # grey lines at 0
        plt.ylabel(r'$Q$ '+r'(W/m$^2$)')

        ax=plt.subplot(212)
        plt.plot(self.Q_nc_c[self.sensor_names[1:len(self.sensor_names)]])
        plt.legend(self.Qstr[:], loc='lower right')
        plt.plot(self.Q_nc[self.sensor_names[len(self.sensor_names)-1]],c=[0.5,0.5,0.5],linestyle='--') # grey lines at 0
        plt.ylabel(r'$Q_{total}$'+r' (W/m$^2$)')
        ax.xaxis.set_major_formatter(dt_format)
        if period is not None:
            plt.xlim(period)

        if pd:
            plt.savefig(plotdir+'timeseries_Qcnc_Q.png', bbox_inches='tight')

        # Q_nc/Q_c at each level:
        plt.figure(figsize=(12.8,4.8), dpi=100)
        ax=plt.subplot(111)
        # Make dataframe for Qnc/Qc ratio, plot:
        self.Q_nc_c_ratio = self.Q_nc.copy()
        for nn in range(1,len(self.sensor_names)-1):
            self.Q_nc_c_ratio[self.sensor_names[nn]] = self.Q_nc[self.sensor_names[nn]]/self.Q_c[self.sensor_names[nn]]
            plt.plot(self.Q_nc_c_ratio[self.sensor_names[nn]],c=colors[nn-1])
        #plt.plot(self.Q_c[self.sensor_names[len(self.sensor_names)]])
        plt.ylabel(r'$Q_{nc}$/$Q_{c}$')
        plt.legend(self.Qstr[:], loc='lower right')
        ax.xaxis.set_major_formatter(dt_format)
        if period is not None:
            plt.xlim(period)

        if pd:
            plt.savefig(plotdir+'timeseries_Qnc_c_ratio.png', bbox_inches='tight')

        # Q_nc/Q_c histograms:
        for nn in range(1,len(self.sensor_names)-1):
            print("Mean Qnc/Qc = {}".format(self.Q_nc_c_ratio[self.sensor_names[nn]].mean()))
            print("Median Qnc/Qc = {}".format(self.Q_nc_c_ratio[self.sensor_names[nn]].median()))
            print("Max Qnc/Qc = {}".format(self.Q_nc_c_ratio[self.sensor_names[nn]].max()))
            plt.figure()
            plt.hist(self.Q_nc_c_ratio[self.sensor_names[nn]], bins=np.arange(-5,5,0.2))
            plt.xlabel(r'$Q_{nc}$/$Q_{c}$')
            plt.ylabel('Count')
            plt.title(self.Qstr[nn-1])

        #plt.ylim([-2,3])
        # Mean Diurnal Cycle Plots:
        # Calculate and plot avg. diurnal cycles:
        # dqdz:
        plt.figure()
        for n in np.arange(1,len(self.sensor_names)-1):
            dqdz_di, dqdz_err = calc_diurnal(self.datetime, self.dqdz[self.sensor_names[n]].to_numpy())
            plt.plot(dqdz_di)
        plt.legend(self.depth_str[1:len(self.sensor_names)-1], loc='upper right')
        plt.xlabel('Hour of Day')
        plt.ylabel(r'$\partial Q_{nc} / \partial z $ $(W/m^2)/(cm)$')
        plt.plot([0,24],[0,0],'k--')
        plt.xlim([0,24])

        if pd:
            plt.savefig(plotdir+'diurnal_dqdz.png', bbox_inches='tight')

        # Q_c and Q_nc diurnal:
        plt.figure()
        for n in np.arange(1,len(self.sensor_names)-1):
            Qnc_di, Qnc_err = calc_diurnal(self.datetime, self.Q_nc[self.sensor_names[n]].to_numpy())
            Qc_di, Qc_err = calc_diurnal(self.datetime, self.Q_c[self.sensor_names[n]].to_numpy())
            plt.plot(Qc_di,c=colors[n-1])
            plt.plot(Qnc_di,c=colors[n-1],linestyle='--')
        plt.legend(self.Qcncstr, loc='upper left')
        plt.xlabel('Hour of Day')
        plt.plot([0,24],[0,0],'k--')
        plt.ylabel(r'$Q$ '+r'(W/m$^2$)')
        plt.xlim([0,24])
        plt.ylim([-40,140])

        if pd:
            plt.savefig(plotdir+'diurnal_Qc_Qnc.png', bbox_inches='tight')

        #Q_nc/Q_c diurnal
        plt.figure()
        for n in np.arange(1,len(self.sensor_names)-1):
            Qnc_di, Qnc_err = calc_diurnal(self.datetime, self.Q_nc[self.sensor_names[n]].to_numpy())
            Qc_di, Qc_err = calc_diurnal(self.datetime, self.Q_c[self.sensor_names[n]].to_numpy())
            plt.plot(Qnc_di/Qc_di,c=colors[n-1])
        plt.legend(self.Qstr, loc='upper right')
        plt.xlabel('Hour of Day')
        plt.plot([0,24],[0,0],'k--')
        plt.ylabel(r'$Q_{nc}/Q_c$')
        plt.xlim([0,24])
        if pd:
            plt.savefig(plotdir+'diurnal_QncQc.png', bbox_inches='tight')

        plt.figure()
        #for n in np.arange(1,len(self.sensor_names)):
        for nn in range(1,len(self.sensor_names)-1):
            sensor = self.sensor_names[nn]
            plt.scatter(self.Q_c[sensor], self.Q_nc[sensor], marker='.', c=colors[nn-1])
        plt.legend(self.Qstr)
        plt.xlabel(r'$Q_{c}$ $(W/m^2)$')
        plt.ylabel(r'$Q_{nc}$ $(W/m^2)$')
        if pd:
            plt.savefig(plotdir+'Qc_vs_Qnc.png', bbox_inches='tight')

        return

class leif_debris_temp_profile(debris_temp_profile):

    """
    Class is same as debris temp profile, but loads data
    from Leif Anderson's file format.
    """
    def __init__(self, datafile, depthfile, period=None, mask=None, smooth=None, downsample=None, plotdir=None):
        """
        Initiation of Class

        datafile: File containing debris temp profile as Leif formatted
            Tab-delimited
            DATE temp1 temp2 temp3 temp4
            DATE is in mm/DD/YY HH:MM format
        depthfile: contains depths at which the sensors rest.
            single depth on each line
        """

        if plotdir is not None:
            self.pd = 1
            self.plotdir = plotdir
        else:
            self.pd = 0
        data = pd.read_csv(datafile, sep='\t', header=None, index_col=0, parse_dates=True)
        sensor_depth = pd.read_csv(depthfile, header=None)
        sensor_depth = sensor_depth[0]

        #Assign sensor names based on sensor_depth:
        sensor_names = []
        for depth in sensor_depth:
            sensor_names.append(str(depth)+ ' cm')

        print(sensor_names)
        print(sensor_depth)
        
        data.columns = sensor_names

        # Assign!
        self.data = data
        self.sensor_depth = sensor_depth
        self.sensor_names = sensor_names

        self.calc_derivatives(period, mask, smooth)

        return


def main():
    data_dir = './Debris_Temp/'
    P_2020a_dir = data_dir + 'P_2020a/'
    P_2020b_dir = data_dir + 'P_2020b/'

    summer_period = ['2020-07-27 00:00:00', '2020-09-06 12:25:00']
    plot_period = ['2020-07-28 00:00:00', '2020-08-05 00:00:00']
    pd0 = './Figures/'

    P_2020a = debris_temp_profile(P_2020a_dir, smooth=6, period=summer_period, plotdir=pd0+'P_2020a/')#, period=period2)#, mask=remove_period)
    P_2020a.plot_debris_temp(period=plot_period)
    P_2020a.estimate_ke_Q(display_hr=True, animate=False, hours_of_day=[24,7])
    P_2020a.plot_dqdz_Q(period=plot_period)

    P_2020b = debris_temp_profile(P_2020b_dir, smooth=6, period=summer_period, plotdir=pd0+'P_2020b/')
    P_2020b.plot_debris_temp(period=plot_period, dt='day')
    P_2020b.estimate_k_from_melt(['2020-07-26 12:25:00','2020-09-06 12:25:00'], .4, 0.624)
    P_2020b.estimate_ke_Q(display_hr=True, animate=False, hours_of_day=None)
    P_2020b.plot_dqdz_Q(period=plot_period)

    P_2011 = leif_debris_temp_profile(data_dir+'P_2011/Kennicott_red.txt', data_dir+'P_2011/Kennicott_red_depth.txt', smooth=6, plotdir=pd0+'P_2011/')
    P_2011.plot_debris_temp(dt="day")
    P_2011.estimate_ke_Q(display_hr=True, animate=False, hours_of_day=[23,2])
    P_2011.plot_dqdz_Q()

    plt.show()

if __name__ == "__main__":

    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    main()