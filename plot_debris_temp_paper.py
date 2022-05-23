import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 
import matplotlib.dates as mdates
import matplotlib.animation as ani
from aws_tools import *
import os, sys 
from sklearn import linear_model

### Plotting font size settings:
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'k','#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
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
        self.hr = self.datetime.hour # Calculate Hour of Day for Plotting purposes
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
        # Estimate dT/dz for sensor locations:
        self.dTdz_sensors = self.dTdz.copy()
        for n in range(len(self.sensor_names)-1):
            self.dTdz_sensors[self.sensor_names[n]] = (self.dTdz[self.sensor_names[n]] + self.dTdz[self.sensor_names[n+1]])/2
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

    def plot_debris_temp(self, leg=None, period=None, dt="day", animate=False, show_plot=False):
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
            plot_timeseries(self.data, self.sensor_names, legend=leg, ylbl='T '+r'($^\circ$C)', period=period, colors=colors)
        else: 
            plot_timeseries(self.data, self.sensor_names, legend=leg, ylbl='T '+r'($^\circ$C)', colors=colors)
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
        txt2 = 'Gradient = {:.2f}'.format(grad_mod.model.coef_[0][0]) + r'$^\circ$C cm$^{-1}$'
        txt3 = "Surf. Temp (Fit) = {:.1f}".format(grad_mod.model.intercept_[0]) + r'$^\circ$C '
        plt.text(0.38,0.28,txt0,transform=ax.transAxes,fontsize=16)
        plt.text(0.38,0.21,txt1,transform=ax.transAxes,fontsize=16)
        plt.text(0.38,0.14,txt2,transform=ax.transAxes,fontsize=16)
        plt.text(0.38,0.07,txt3,transform=ax.transAxes,fontsize=16)
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

    def estimate_ke_Q(self, dt = 5, period=None, smooth=6, saveas=None, display_hr=False, animate=False, ke_ideal=None, hours_of_day=None, show_diff_ts=False, rho=1800, c=900, plotdir=None):
        """ Calculates dT/dt and d2T/dz2, plots them, and performs 
        a linear regression to estimate the best fit effective thermal 
        diffusivity ke.
        Then uses the effective thermal diffusivity to calculate the residual in
        the model, here referred to as "dqdz," and integrates up from the debris-ice
        interface to estimate Q_nc.
        hours_of_day = [morning_hour, evening_hour], where the desired 
            sampling time is overnight between the evening & morning hours
        """

        #Plot saving directory option:
        if plotdir is not None:
            self.pd=1
            if not os.path.exists(plotdir):
                os.makedirs(plotdir)
        if plotdir is None and self.pd==1:
            plotdir = self.plotdir

        # Pull differentials out of the program:
        dTdt = self.dTdt
        dTdz2 = self.dTdz2

        # Initiate dqdz, Q_nc, Q_c:
        dqdz = self.dTdz2.copy() * 0
        Q_nc = self.dTdz.copy() * 0
        Q_c = self.dTdz.copy() * 0

        hr = self.datetime.hour # Calculate Hour of Day for Plotting purposes
        # Mask for data into hours of most conductive behavior if provided:
        if hours_of_day is not None:
            if hours_of_day[0] > hours_of_day[1]:
                hi = ( (hr>= hours_of_day[0]) | (hr<= hours_of_day[1])) # hi = mask to highlight desired data
            else:
                hi = ( (hr>= hours_of_day[0]) & (hr<= hours_of_day[1]))
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
            R2[nn] = model.R2

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
            plt.xlabel(r'$\partial^2 T / \partial z^2$'+' '+r'(K m$^{-2}$)')
            plt.ylabel(r'$\partial T / \partial t$'+' '+r'(mK s$^{-1}$)')
            txt1 = r'R$^2$'+' = {:.2f}'.format(model.R2)
            txt2 = r'$\kappa$'+' = {:.2e} '.format(model.model.coef_[0][0]/1000) + r'(m$^2$ s$^{-1}$)'
            plt.text(0.02,0.9,txt2,transform=ax.transAxes)
            plt.text(0.02,0.8,txt1,transform=ax.transAxes)
            plt.legend(leg_str, loc='lower right')

            if ke_ideal is not None:
                y_mod_ideal = np.array(x_mod) * ke_ideal * 1000
                plt.plot(x_mod, y_mod_ideal, 'k')
                txt3 = r'Ideal $\kappa_e$ = '+'{:.3e}'.format(ke_ideal)
                plt.text(0.02,0.85,txt3,transform=ax.transAxes)
                plt.legend(['Fit','Ideal','Data'], loc='lower right')
            if plotdir is not None:
                plt.savefig(plotdir+'diff_scatter_'+str(nn)+'.png',bbox_inches='tight')

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
                sub3.plot(Q_nc[self.sensor_names[1:len(self.sensor_names)]].loc[self.datetime[i],:], -self.dTdz_z, 'k*-')
                #sub3.plot(Q_nc_c[self.sensor_names[1:len(self.sensor_names)]].loc[self.datetime[i],:], -self.dTdz_z, 'k*--')
                sub3.legend(['0',r'$Q_{c}$',r'$Q_{nc}$'])#,r'$\Sigma Q$'])
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
                animator.save(plotdir+'debris_temp_animation.mp4')
            else:
                animator.save(r'/Users/ericpetersen/Desktop/debris_temp_animation.mp4')

        #plt.show()
        self.dqdz = dqdz
        self.Q_nc = Q_nc
        self.Q_c = Q_c
        self.Q_nc_c = Q_nc+Q_c
        self.ke = ke
        self.ke_z2 = ke_z2
        self.R2 = R2
        self.hr = hr
        #self.Q_nc_c[(self.Q_c<15)] = np.nan

        return

    def animate_plot(self):

        fig = plt.figure(figsize=(12,8))
        plt.subplots_adjust(wspace=0.25, hspace=0.25)
        # Subplots for T, dqdz, Qnc:
        sub1 = fig.add_subplot(1,5,1)
        sub2 = fig.add_subplot(1,5,2)
        sub3 = fig.add_subplot(1,5,3)
        # Subplots for differentials:
        subs = {}
        for n in range(1,self.num_sensors-1):
            subs[self.sensor_names[n]] = fig.add_subplot(self.num_sensors-2,3,(n-1)*3+3)
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
            sub2.plot(self.dqdz[self.sensor_names[1:len(self.sensor_names)-1]].loc[self.datetime[i],:], -self.sensor_depth[1:len(self.sensor_names)-1], 'k*-')
            sub2.set_xlim(np.nanmin(self.dqdz), np.nanmax(self.dqdz))
            sub2.set_ylim([-np.max(self.sensor_depth)-4, 0])
            sub2.set_xlabel(r'$\partial Q_{nc} / \partial z$ $(W/m^2)/cm$')

            sub3.clear()
            sub3.plot([0, 0], [-np.max(self.sensor_depth)-4, 0], 'k--',alpha=0.5)
            sub3.plot(self.Q_c[self.sensor_names[1:len(self.sensor_names)]].loc[self.datetime[i],:], -self.dTdz_z, 'b*--')
            sub3.plot(self.Q_nc[self.sensor_names[1:len(self.sensor_names)]].loc[self.datetime[i],:], -self.dTdz_z, 'k*-')
            #sub3.plot(Q_nc_c[self.sensor_names[1:len(self.sensor_names)]].loc[self.datetime[i],:], -self.dTdz_z, 'k*--')
            sub3.legend(['0',r'$Q_{c}$',r'$Q_{nc}$'])#,r'$\Sigma Q$'])
            sub3.set_ylim([-np.max(self.sensor_depth)-4, 0])
            sub3.set_xlim(np.nanmin(self.Q_nc), np.nanmax(self.Q_nc_c))
            sub3.set_xlabel(r'$Q_{nc}$ $(W/m^2)$')

            for nn in range(1,self.num_sensors-1):
                sub = subs[self.sensor_names[nn]]
                sub.clear()
                # plot first the full dataset up until this point in color, with transparency
                x_mod = [[self.dTdz2[self.sensor_names[nn]].min()], [self.dTdz2[self.sensor_names[nn]].max()]]
                y_mod = np.array(x_mod) * self.ke[nn] * 1000
                sub.plot(x_mod,y_mod,'r--') # linear fit for determination of dqdz
                sub.scatter(self.dTdz2[self.sensor_names[nn]][:i], self.dTdt[self.sensor_names[nn]][:i]*1000, c=self.hr[:i], marker='.', s=5, alpha=0.7)
                # plot the dataset of the day so far in dark grey:
                idate = self.datetime[i]
                iday = datetime(idate.year, idate.month, idate.day)
                ind = (self.datetime >= iday) & (self.datetime <= idate)
                sub.scatter(self.dTdz2[self.sensor_names[nn]][iday:idate], self.dTdt[self.sensor_names[nn]][iday:idate]*1000, c='k', marker='.', s=5, alpha=0.8)
                # plot the current datapoint in bold, black:
                sub.scatter(self.dTdz2[self.sensor_names[nn]][i], self.dTdt[self.sensor_names[nn]][i]*1000, c='k', marker = 'o')
                # ensure consistent framing:
                sub.set_xlabel(r'$\partial^2 T / \partial z^2$'+' '+r'(K/m$^2$)')
                sub.set_ylabel(r'$\partial T / \partial t$'+' (mK/s)')
                #sub2.set_xlim(model.x_lim1.to_numpy(), model.x_lim2.to_numpy())
                sub.set_xlim(self.dTdz2[self.sensor_names[nn]].min(), self.dTdz2[self.sensor_names[nn]].max())
                sub.set_ylim(self.dTdt[self.sensor_names[nn]].min()*1000, self.dTdt[self.sensor_names[nn]].max()*1000)
                txt = self.depth_str[nn]
                sub.text(0.05,0.95,txt,transform=sub.transAxes)
            plt.suptitle(idate)
        
        animator = ani.FuncAnimation(fig, ani_frame_ke_dqdz_q, interval=16, save_count=np.size(self.datetime))
        if pd:
            animator.save(self.plotdir+'debris_temp_animation.mp4')
        else:
            animator.save(r'/Users/ericpetersen/Desktop/debris_temp_animation.mp4')

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

        self.ke = kappa_e
        ke_z2 = np.zeros(self.num_sensors)
        for nn in range(1,self.num_sensors):
            if nn==1:
                ke_z2[nn] = kappa_e[nn]
            elif nn==self.num_sensors-1:
                ke_z2[nn] = kappa_e[nn-1]
            else:
                ke_z2[nn] = (kappa_e[nn] + kappa_e[nn-1])/2
        self.ke_z2 = ke_z2
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
        plt.ylabel(r'$\partial Q_{nc} / \partial z$'+'\n'+' (W '+r'$m^{-2} cm^{-1}$)')
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
        plt.ylabel(r'$\partial Q_{nc} / \partial z$'+'\n'+' (W '+r'$m^{-2} cm^{-1}$)')
        dt_format = mdates.DateFormatter('%b-%d')
        ax.xaxis.set_major_formatter(dt_format)
        ax.tick_params(axis='both', which='major')
        if period is not None:
            plt.xlim(period)

        ax=plt.subplot(412)
        plt.plot(self.Q_c[self.sensor_names[1:len(self.sensor_names)]])
        plt.legend(self.Qstr[:], loc='lower right')
        plt.plot(self.Q_nc[self.sensor_names[len(self.sensor_names)-1]],c=[0.5,0.5,0.5],linestyle='--') # grey lines at 0
        plt.ylabel(r'$Q_{c}$'+'\n'+'(W '+r'$m^{-2})$')
        plt.ylim([Q_min,Q_max])
        ax.xaxis.set_major_formatter(dt_format)
        if period is not None:
            plt.xlim(period)

        ax=plt.subplot(413)
        plt.plot(self.Q_nc[self.sensor_names[1:len(self.sensor_names)-1]])
        plt.legend(self.Qstr[:-1], loc='lower right')
        plt.plot(self.Q_nc[self.sensor_names[len(self.sensor_names)-1]],c=[0.5,0.5,0.5],linestyle='--') # grey lines at 0
        plt.ylabel(r'$Q_{nc}$'+'\n'+'(W '+r'$m^{-2})$')
        plt.ylim([Q_min,Q_max])
        ax.xaxis.set_major_formatter(dt_format)
        if period is not None:
            plt.xlim(period)

        ax=plt.subplot(414)
        plt.plot(self.Q_nc_c[self.sensor_names[1:len(self.sensor_names)]])
        plt.legend(self.Qstr[:], loc='lower right')
        plt.plot(self.Q_nc[self.sensor_names[len(self.sensor_names)-1]],c=[0.5,0.5,0.5],linestyle='--') # grey lines at 0
        plt.ylabel(r'$Q_{total}$'+'\n'+'(W '+r'$m^{-2}$)')
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
        plt.ylabel(r'$Q$ '+'(W '+r'$m^{-2}$)')

        ax=plt.subplot(212)
        plt.plot(self.Q_nc_c[self.sensor_names[1:len(self.sensor_names)]])
        plt.legend(self.Qstr[:], loc='lower right')
        plt.plot(self.Q_nc[self.sensor_names[len(self.sensor_names)-1]],c=[0.5,0.5,0.5],linestyle='--') # grey lines at 0
        plt.ylabel(r'$Q_{total}$'+r' (W m$^{-2}$)')
        ax.xaxis.set_major_formatter(dt_format)
        if period is not None:
            plt.xlim(period)

        if pd:
            plt.savefig(plotdir+'timeseries_Qcnc_Q.png', bbox_inches='tight')

        # Mean Diurnal Cycle Plots:
        # Calculate and plot avg. diurnal cycles:
        # dqdz:
        plt.figure()
        for n in np.arange(1,len(self.sensor_names)-1):
            dqdz_di, dqdz_err = calc_diurnal(self.datetime, self.dqdz[self.sensor_names[n]].to_numpy())
            plt.plot(dqdz_di)
        plt.legend(self.depth_str[1:len(self.sensor_names)-1], loc='upper right')
        plt.xlabel('Hour of Day')
        plt.ylabel(r'$\partial Q_{nc} / \partial z $ $(W m^{-2} cm^{-1})$')
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
        plt.ylabel(r'$Q$ '+r'(W m^${-2}$)')
        plt.xlim([0,24])
        plt.ylim([-40,140])

        if pd:
            plt.savefig(plotdir+'diurnal_Qc_Qnc.png', bbox_inches='tight')


        # Q_nc/Q_c at each level:
        ratio = 0
        if ratio:
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
            plt.xlabel(r'$Q_{c}$ $(W m^{-2})$')
            plt.ylabel(r'$Q_{nc}$ $(W m^{-2})$')
            if pd:
                plt.savefig(plotdir+'Qc_vs_Qnc.png', bbox_inches='tight')

        return

    def plot_Q_diurnal(self, period=None, sunny_file=None, Q_lim=None):
        """
        This sub-routine plots diurnal cycles in heat fluxes.
        If a "sunny_file" is produced, the diurnal cycles are
        additionally separated out into diurnal cycles for sunny days 
        and cloudy days.
        """

        Qc_ds_str = []
        Qnc_ds_str = []
        for z_value in self.dTdz_z:
            Qc_ds_str.append('{:.1f} cm'.format(z_value)+', Sunny')
            Qc_ds_str.append('{:.1f} cm'.format(z_value)+', Cloudy')
            Qnc_ds_str.append('{:.1f} cm'.format(z_value)+', Sunny')
            Qnc_ds_str.append('{:.1f} cm'.format(z_value)+', Cloudy')

        leg_size=12
        # Q_c and Q_nc diurnal:
        plt.figure()
        for n in np.arange(1,len(self.sensor_names)-1):
            Qnc_di, Qnc_err = calc_diurnal(self.datetime, self.Q_nc[self.sensor_names[n]].to_numpy())
            Qc_di, Qc_err = calc_diurnal(self.datetime, self.Q_c[self.sensor_names[n]].to_numpy())
            plt.plot(Qc_di,c=colors[n-1])
            plt.plot(Qnc_di,c=colors[n-1],linestyle='--')
        plt.legend(self.Qcncstr, loc='upper left', prop={'size': leg_size})
        plt.xlabel('Hour of Day')
        plt.plot([0,24],[0,0],'k--')
        plt.ylabel(r'$Q$ '+'(W '+r'$m^{-2}$)')
        plt.xlim([0,24])
        if Q_lim is not None:
            plt.ylim(Q_lim)

        if self.pd:
            plt.savefig(self.plotdir+'diurnal_Qc_Qnc.png', bbox_inches='tight')
        
        # Sunny vs. cloudy days, if sunny_file provided:
        if sunny_file is not None:
            sunny = pd.read_csv(sunny_file, parse_dates=[0], index_col=0)
            # Return series with only dates for debris temp information:
            suni = self.data[self.data.columns[1]] * 0 - 1
            for nn in range(len(sunny)):
                temp_di = self.datetime.date == sunny.index.date[nn]
                suni[temp_di] = sunny['Sunny'][nn]
            suni = suni.astype(bool)
            cldi = np.invert(suni)
        
            # Q_c and Q_nc diurnal for sunny vs. cloudy:
            plt.figure()
            for n in np.arange(1,len(self.sensor_names)-1):
                Qc_di_sun, Qnc_err_sun = calc_diurnal(self.datetime[suni], self.Q_c[self.sensor_names[n]][suni].to_numpy())
                Qc_di_cld, Qc_err_sun = calc_diurnal(self.datetime[cldi], self.Q_c[self.sensor_names[n]][cldi].to_numpy())
                plt.plot(Qc_di_sun,c=colors[n-1])
                plt.plot(Qc_di_cld,c=colors[n-1],linestyle='--')
            plt.legend(Qc_ds_str, loc='upper left', prop={'size': leg_size})
            plt.xlabel('Hour of Day')
            plt.plot([0,24],[0,0],'k--')
            plt.ylabel(r'$Q_c$ '+'(W '+r'$m^{-2}$)')
            plt.xlim([0,24])
            if Q_lim is not None:
                plt.ylim(Q_lim)

            if self.pd:
                plt.savefig(self.plotdir+'diurnal_Qc_sunnycloudy.png', bbox_inches='tight')

            plt.figure()
            for n in np.arange(1,len(self.sensor_names)-1):
                Qnc_di_sun, Qnc_err_sun = calc_diurnal(self.datetime[suni], self.Q_nc[self.sensor_names[n]][suni].to_numpy())
                Qnc_di_cld, Qc_err_sun = calc_diurnal(self.datetime[cldi], self.Q_nc[self.sensor_names[n]][cldi].to_numpy())
                plt.plot(Qnc_di_sun,c=colors[n-1])
                plt.plot(Qnc_di_cld,c=colors[n-1],linestyle='--')
            plt.legend(Qnc_ds_str, loc='upper left', prop={'size': leg_size})
            plt.xlabel('Hour of Day')
            plt.plot([0,24],[0,0],'k--')
            plt.ylabel(r'$Q_{nc}$ '+'(W '+r'$m^{-2}$)')
            plt.xlim([0,24])
            if Q_lim is not None:
                plt.ylim(Q_lim)

            if self.pd:
                plt.savefig(self.plotdir+'diurnal_Qnc_sunnycloudy.png', bbox_inches='tight')

            return

    def estimate_dkdz_problem(self, rho=2040, c=750):
        """
        Given the original heat transfer equation, determine the 
        dk/dz required such that dk/dz dT/dz = dQnc/dz that I estimate
        """

        dkdz_mod = self.dTdz.copy() * 0
        print(self.spacings)
        dkdz = np.diff(self.ke,prepend=True)/(self.spacings*100)*rho * c
        print(self.ke_z2)
        print(dkdz)
        plt.figure()
        for nn in range(1,len(self.sensor_names)-1):
            self.dTdz_sensors[self.sensor_names[nn]][(np.abs(self.dTdz_sensors[self.sensor_names[nn]])<0.01)] = np.nan
            dkdz_mod[self.sensor_names[nn]] = self.dqdz[self.sensor_names[nn]]/self.dTdz_sensors[self.sensor_names[nn]]
            plt.plot(dkdz_mod[self.sensor_names[nn]])
        plt.legend(self.Qstr)
        plt.ylabel(r'$\partial k / \partial z$ '+r'(W/(m K))/cm')

        # Diurnal:
        plt.figure()
        for n in np.arange(1,len(self.sensor_names)-1):
            dkdz_di, dkdz_err = calc_diurnal(self.datetime, dkdz_mod[self.sensor_names[n]].to_numpy())
            plt.plot(dkdz_di,c=colors[n-1])
        #plt.legend(self.Qcncstr, loc='upper left', prop={'size': leg_size})
        plt.xlabel('Hour of Day')
        plt.plot([0,24],[0,0],'k--')
        plt.ylabel(r'$\partial k / \partial z$ '+r'(W/(m K))/cm')
        plt.xlim([0,24])
        #plt.ylim([-40,200])

        # make synthetic dQdz
        dqdz_mod = self.dTdz.copy() * np.nan
        plt.figure()
        for n in np.arange(1,len(self.sensor_names)-1):
            dqdz_mod[self.sensor_names[n]] = self.dTdz_sensors[self.sensor_names[n]] * dkdz[2]
            #if n>1:
            plt.plot(dqdz_mod[self.sensor_names[n]], c=colors[n-1])
            plt.plot(self.dqdz[self.sensor_names[n]], c=colors[n-1], linestyle='--')
        #plt.legend(self.Qcncstr, loc='upper left', prop={'size': leg_size})
        plt.ylabel(r'$\partial Q_{nc} / \partial z$'+'\n'+r'$(W/m^2)/(cm)$')

        # Plot dTdt vs dTdz_sensors:
        hr = self.datetime.hour # Calculate Hour of Day for Plotting purposes
        for n in np.arange(1,len(self.sensor_names)-1):
            fig=plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.dTdz_sensors[self.sensor_names[n]], self.dTdz2[self.sensor_names[n]], self.dTdt[self.sensor_names[n]]*1000, c=hr, s=5)
            #plt.colorbar()
            plt.title('Depth = {:.1f} cm'.format(self.sensor_depth[n]))
            #plt.xlim(model.x_lim1, model.x_lim2)
            #plt.ylim(model.y_lim1, model.y_lim2)
            plt.xlabel(r'$\partial T / \partial z$'+' '+r'(K/m)')
            plt.ylabel(r'$\partial T / \partial t$'+' (mK/s)')

        plt.show()

    def estimate_k_dkdz(self, rho=1630, c=750):
        """
        Performs a multiple linear regression on 
        dTdt as a function of dTdz and d2Tdz2 in
        order to constrain k and dkdz, taking then
        the residual as dQdz. This is theoretically
        an improvement on estimate_ke_Q.
        """

        # Initiate dqdz, Q_nc, Q_c:
        dqdz = self.dTdz2.copy() * 0
        Q_nc = self.dTdz.copy() * 0
        Q_c = self.dTdz.copy() * 0

        R2 = np.ones((self.num_sensors)) * np.nan
        kappa = np.ones((self.num_sensors))  * np.nan
        ke_z2 = np.ones((self.num_sensors)) * np.nan
        dkappadz = np.ones((self.num_sensors)) * np.nan
        x_mod = {}
        y_mod = {}
        #Loop through middle sensors:
        for n in range(1,self.num_sensors-1):
            # set up linear fit:
            X = pd.DataFrame(columns=['dTdt','dTdz','d2Tdz2'])
            X['dTdt'] = self.dTdt[self.sensor_names[n]].copy()
            X['dTdz'] = self.dTdz_sensors[self.sensor_names[n]].copy()
            X['d2Tdz2'] = self.dTdz2[self.sensor_names[n]].copy()
            # Remove nans:
            X = X.dropna()
            y = X['dTdt']
            X = X.drop(columns=['dTdt'])
            hr_mod = X.index.hour

            # Make regression:
            regr = linear_model.LinearRegression(fit_intercept=False)
            regr.fit(X, y)
            print(regr.coef_)
            kappa[n] = regr.coef_[1]
            dkappadz[n] = regr.coef_[0]
            R2[n] = regr.score(X, y)
            print(R2[n])

            # Predict data:
            dTdt_mod = regr.predict(X)
            dqdz[self.sensor_names[n]] = rho * c * (y-dTdt_mod) / 100 # (W/m2) / cm

            #Plots:
            # Data vs. Model:
            fig,ax = plt.subplots()
            plt.scatter(dTdt_mod*1000, y*1000, c='k', marker='.', alpha=0.7, s=7)
            plt.ylabel(r'$[\partial T / \partial t]_{data}$'+' (mK '+r's$^{-1}$)')
            plt.xlabel(r'$[\partial T / \partial t]_{model}$'+' (mK '+r's$^{-1}$)')
            plt.title('Depth = {:.1f} cm'.format(self.sensor_depth[n]))
            txt1 = r'R$^2$'+' = {:.2f}'.format(R2[n])
            txt2 = r'$\kappa$'+' = {:.2e} '.format(kappa[n]) + r'(m$^2$ s$^{-1}$)'
            txt3 = r'$\partial \kappa/\partial z$'+' = {:.2e} '.format(dkappadz[n]/100) + r'(m$^2$ s$^{-1}$ cm$^{-1}$)'
            plt.text(0.02,0.9,txt3,transform=ax.transAxes)
            plt.text(0.02,0.8,txt2,transform=ax.transAxes)
            plt.text(0.02,0.7,txt1,transform=ax.transAxes)
            if self.plotdir is not None:
                plt.savefig(self.plotdir+'data_model_dTdt_'+str(n)+'.png',bbox_inches='tight')

            # 2D plot with dTdz as point color:
            plt.figure()
            plt.scatter(self.dTdz2[self.sensor_names[n]], self.dTdt[self.sensor_names[n]]*1000, c=self.dTdz_sensors[self.sensor_names[n]], s=5)
            plt.xlabel(r'$\partial^2 T / \partial z^2$'+' (K '+r'm$^{-2}$)')
            plt.ylabel(r'$\partial T / \partial t$'+' (mK '+r's$^{-1}$)')
            cbar = plt.colorbar()
            cbar.set_label(r'$\partial T / \partial z$'+' (K '+r'm$^{-1}$)', rotation=270, labelpad=30)
            plt.title('Depth = {:.1f} cm'.format(self.sensor_depth[n]))
            plt.grid()
            if self.plotdir is not None:
                plt.savefig(self.plotdir+'diff_scatter_'+str(n)+'.png',bbox_inches='tight')

            # 3D plot:
            fig=plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.dTdz_sensors[self.sensor_names[n]], self.dTdz2[self.sensor_names[n]], self.dTdt[self.sensor_names[n]], c=hr_mod, s=5)
            ax.scatter(X['dTdz'], X['d2Tdz2'], dTdt_mod, color='k', s=1)
            #plt.colorbar()
            plt.title('Depth = {:.1f} cm'.format(self.sensor_depth[n]))
            #plt.xlim(model.x_lim1, model.x_lim2)
            #plt.ylim(model.y_lim1, model.y_lim2)
            plt.xlabel(r'$\partial T / \partial z$'+' (K '+r'm$^{-1})')
            plt.ylabel(r'$\partial^2 T / \partial z^2$'+' (K '+r'm$^{-2}$)')
            ax.set_zlabel(r'$\partial T / \partial t$'+' (K '+r's$^{-1}$)')

        # Calculate Q_c:
        # Extrapolate boundary k from dkdz:
        kappa[0] = kappa[1] - dkappadz[1] * self.spacings[1] # k at first sensor location
        kappa[self.num_sensors-1] = kappa[self.num_sensors-2] + dkappadz[self.num_sensors-2] * self.spacings[self.num_sensors-1] # k at bottom sensor location
        
        # calculate average ke between sensors in order to calculate Q_c:
        ke_z2[0] = np.nan
        for nn in range(1,self.num_sensors):
            ke_z2[nn] = (kappa[nn] + kappa[nn-1])/2

            Q_c[self.sensor_names[nn]] = ke_z2[nn] * rho * c * -self.dTdz[self.sensor_names[nn]]

        # Plot kappa and k profiles:
        plt.figure(figsize=(3,5.5), dpi=100)
        plt.plot(kappa[0:2], -self.sensor_depth[0:2], 'k--',)
        plt.plot(kappa[1:-1], -self.sensor_depth[1:-1], 'k*-', markersize=10)
        plt.plot(kappa[-2:], -self.sensor_depth[-2:], 'k--')
        plt.xlabel(r'$\kappa$ (m$^2$ s$^{-1}$)')
        plt.ylabel('Depth (cm)')
        if self.plotdir is not None:
            plt.savefig(self.plotdir+'kappa_profile.png',bbox_inches='tight')

        plt.figure(figsize=(3.5,5), dpi=100)
        plt.plot(kappa[0:2] * rho * c, -self.sensor_depth[0:2], 'k--')
        plt.plot(kappa[1:-1] * rho * c, -self.sensor_depth[1:-1], 'k*-', markersize=10)
        plt.plot(kappa[-2:] * rho * c, -self.sensor_depth[-2:], 'k--')
        plt.ylim([-25,0])
        plt.xlim([0,2])
        plt.xlabel('k (W '+r'm$^{-1}$'+' '+r'K$^{-1}$)')
        plt.ylabel('Depth (cm)')
        if self.plotdir is not None:
            plt.savefig(self.plotdir+'k_profile.png',bbox_inches='tight')
        print(kappa)
        print(ke_z2)
        # Determine Q_nc (assuming Q_nc at ice-debris boundary = 0)
        Q_nc = (dqdz.copy())  * self.dz2_mat * 100
        Q_nc[self.sensor_names[self.num_sensors-1]] = self.data[self.sensor_names[self.num_sensors-1]] * 0
        N = self.num_sensors
        for n in range(2, N-1):
            Q_nc[self.sensor_names[N-n-1]] = Q_nc[self.sensor_names[N-n-1]] + Q_nc[self.sensor_names[N-n]]

        self.dqdz = dqdz
        self.Q_nc = Q_nc
        self.Q_c = Q_c
        self.Q_nc_c = Q_nc+Q_c # sum of Q, conductive & non-conductive
        self.ke = kappa
        self.ke_z2 = ke_z2
        self.R2 = R2

        return
        
    def plot_EB_model_comparison(self, kd, td_model, period=None, dt="month", hide_surf=0):
            """
            Compares Temp, Q_c, Q_nc derived from data with outputs from
            from Crank-Nicolson scheme by Rounce Energy Balance Model, producing
            a "Flux Deficit" Qd and the potential evaporation rate from such.

            NOTE: this method is a combination of two others, may require some
            work (has not been used recently)
            """
            G = self.Q_c # Take G as Q_c previously derived
            
            Gstr = []

            for z_value in self.dTdz_z:
                Gstr.append(str(z_value)+ ' cm')

            # Read in modeled debris temperature and 
            Td = pd.read_csv(td_model, parse_dates=True, index_col=0)
            Td.describe()
            Td_dt = (Td.index[1] - Td.index[0]).total_seconds()
            # Wrangle modeled temps to same timestamps as measured temps:
            Td = conform_records(Td, self.data)
            Td.describe()

            # Produce array of depth values for modeled temps:
            N = len(Td.columns)
            Td_depth = np.zeros(N)
            for n in range(N):
                Td_depth[n] = float(Td.columns[n]) * 100 # Depth in cm
            
            # select the modeled layers closest to 
            #   the measurement locations.
            Td_comp = pd.DataFrame()
            Td_comp_depth = np.zeros(self.num_sensors)
            for m in range(self.num_sensors):
                i = np.argmin(np.abs(self.sensor_depth[m] - Td_depth))
                if (i == len(Td_depth)-1) & (self.sensor_names[m] != "Base"):
                    i=i-1
                Td_comp[self.sensor_names[m]] = Td[Td.columns[i]]
                Td_comp_depth[m] = Td_depth[i]
            # Calculate differential and Qc from Td_comp:
            Td_comp_spacings = np.diff(Td_comp_depth, prepend=np.nan)/100
            Td_comp_spacing_mat = np.ones((self.length, 1))*Td_comp_spacings
            dTdz_model = Td_comp.diff(axis=1) / Td_comp_spacing_mat
            G_model = kd * dTdz_model

            #### NOTE: Right now, I haven't written anything to automatically
            #           register the modeled debris temperature to the same 
            #           timestamps of measured debris temps. By manually 
            #           averaging measured values to hourlies, they should be
            #           comparable. But if one doesn't do that, the code won't
            #           really work...
            # Calculate Qd:
            Qd = G_model - G

            dQdz = -Qd.diff(axis=1,periods=-1) / self.dz2_mat / 100 # dT/dz2 in (W/m2)/cm

            dTdz_dz2_avg = self.dTdz
            for n in np.arange(1,self.num_sensors-1):
                dTdz_dz2_avg[self.sensor_names[n]] = (self.dTdz[self.sensor_names[n]] + self.dTdz[self.sensor_names[n+1]])/2

            dkdzdTdz = self.dqdz - dQdz

            dkdz = dkdzdTdz / (dTdz_dz2_avg)

            # Calculate E_p (potential evaporation)
            E_p = -Qd / (1000 * 2260000) # rate in m/s
            E_p = E_p * 1000 * 3600 # rate in mm/hr
            
            Melt = -G[self.sensor_names[len(self.sensor_names)-1]] * self.dt / (900 * 334000)
            Cum_Melt = np.cumsum(Melt)

            Melt_model = -G_model[self.sensor_names[len(self.sensor_names)-1]] * self.dt / (900 * 334000)
            Cum_Melt_model = np.cumsum(Melt_model)
            
            # Adjust cumulative melt to make it look good in the plot:
            Cum_Melt = Cum_Melt - Cum_Melt[(Cum_Melt.index>period[0])].min() # Start cum melt at start of plotting period
            plt_Cum_Melt = Cum_Melt[period[0]:period[1]]
            Cum_Melt_model = Cum_Melt_model - Cum_Melt_model[(Cum_Melt.index>period[0])].min() # Start cum melt at start of plotting period
            plt_Cum_Melt_model = Cum_Melt_model[period[0]:period[1]]

            if dt=="month":
                major_tick = mdates.MonthLocator()
                dt_format = mdates.DateFormatter('%B')
                ax.xaxis.set_major_locator(major_tick)
            if dt=="day":
                dt_format = mdates.DateFormatter('%b-%d')

            # Plot comparison of debris temperatures themselves:
            plt.figure()
            plt.title('Temperature at Depth')
            for n in range(self.num_sensors):
                ax = plt.subplot(self.num_sensors, 1, n+1)
                plt.plot(self.data[self.sensor_names[n]])
                plt.plot(Td_comp[self.sensor_names[n]]-273.15)
                if period is not None:
                    plt.xlim(period)
                plt.ylabel(r'$^\circ$C')
                txt = str(self.sensor_depth[n]) + ' cm'
                t=plt.text(0.02,0.8,txt,transform=ax.transAxes, backgroundcolor='0.7',alpha=1)
                t.set_bbox(dict(facecolor='0.75',alpha=0.8,edgecolor='k'))
                ax.xaxis.set_major_formatter(dt_format)
            plt.legend(['Data','Model'], loc='lower right')

            # Plot direct comparison of G:
            plt.figure()
            plt.title(r'$Q_c$ at Depth')
            for n in range(self.num_sensors):
                if n==0:
                    continue
                ax = plt.subplot(self.num_sensors-1, 1, n)
                plt.plot(G[self.sensor_names[n]])
                plt.plot(G_model[self.sensor_names[n]])
                plt.ylabel(r'$W/m^2$')
                if period is not None:
                    plt.xlim(period)
                t=plt.text(0.02,0.8,Gstr[n-1],transform=ax.transAxes, backgroundcolor='0.7',alpha=1)
                t.set_bbox(dict(facecolor='0.75',alpha=0.8,edgecolor='k'))
                ax.xaxis.set_major_formatter(dt_format)
                plt.grid(True)
            plt.legend(['Data','Model'], loc='lower right')

            # Plot comparison of Melt, etc:
            plt.figure()

            dqdz_leg=[]
            for n in np.arange(1,len(self.sensor_names)-1):
                dqdz_leg.append(str(self.sensor_depth[n]) + ' cm')

            blarg = 0
            if blarg:
                ax = plt.subplot(414)
                plt.plot(self.dqdz[self.sensor_names[1:len(self.sensor_names)-1]])
                plt.legend(dqdz_leg, loc='lower right')
                plt.ylabel(r'$\partial Q_{nc} / \partial z$ $(W/m^2)/(cm)$')
                if period is not None:
                    plt.xlim(period)
                plt.grid(True)
                ax.xaxis.set_major_formatter(dt_format)

            ax = plt.subplot(312)
            if period is not None:
                plot_timeseries(Qd, Qd.columns[1+hide_surf:], legend=Gstr[hide_surf:], ylbl=r'$Q_d$ '+r'(W/m$^2$)', period=period, loc='lower right')
            else: 
                plot_timeseries(Qd, Qd.columns[1+hide_surf:], legend=Gstr[hide_surf:], ylbl=r'$Q_d$ '+r'(W/m$^2$)', loc='lower right')
            if dt=="month":
                major_tick = mdates.MonthLocator()
                dt_format = mdates.DateFormatter('%B')
                ax.xaxis.set_major_locator(major_tick)
            if dt=="day":
                dt_format = mdates.DateFormatter('%b-%d')
            plt.ylim([-100,50])
            ax.xaxis.set_major_formatter(dt_format)

            ax=plt.subplot(311)
            plt.plot(Melt*100*3600/self.dt, 'b')
            plt.plot(Melt_model*100*3600/Td_dt, 'g')
            plt.legend(['T Data','Model'], loc='lower right')
            plt.ylabel('Melt Rate \n(cm/hr)')
            plt.xlim(period)
            ax.xaxis.set_major_formatter(dt_format)
            plt.grid(True)

            ax2 = ax.twinx()
            ax2.plot(plt_Cum_Melt*100, 'b--')
            ax2.plot(plt_Cum_Melt_model*100, 'g--')
            plt.ylabel('Melt (cm)')
            ax.xaxis.set_major_formatter(dt_format)

            ax = plt.subplot(313)
            plot_timeseries(E_p, Qd.columns[1+hide_surf:], legend=Gstr[hide_surf:], ylbl=r'$E_p$'+' (mm/hr)', period=period, loc='lower right')
            plt.ylabel(r'$E_p$'+' (mm/hr)')
            plt.xlim(period)
            plt.grid(True)
            ax.xaxis.set_major_formatter(dt_format)

            #ax2 = ax.twinx()
            #ax2.plot(100*(E_p[E_p.columns[len(E_p.columns)-1]]+E_p[E_p.columns[len(E_p.columns)-2]]/(10*Melt*100*3600/self.dt)),'k--')
            #plt.ylabel(r'$E_p$/(Melt)'+' (%)')

            # Plot comparison of higher-order derivatives:
            plt.figure()
            N = len(self.sensor_names)-2

            for n in range(N):
                ax = plt.subplot(N,1,n+1)
                plt.plot(self.dqdz[self.sensor_names[n+1]])
                plt.plot(dQdz[self.sensor_names[n+1]])
                #plt.plot(dkdzdTdz[self.sensor_names[n+1]])
                plt.ylabel(r'$(W/m^2)/cm$')
                plt.legend([r'$\partial Q_{nc} / \partial z$', r'$\partial Q_d / \partial z$'], loc='lower right')
                t = plt.text(0.02, 0.9, Gstr[n], transform=ax.transAxes, backgroundcolor='0.7',alpha=1)
                t.set_bbox(dict(facecolor='0.75',alpha=0.8,edgecolor='k'))
                ax.xaxis.set_major_formatter(dt_format)
                if period is not None:
                    plt.xlim(period)

            # Calculate and plot avg. diurnal cycles:

            # Q_d:
            plt.figure()
            for n in np.arange(1,len(self.sensor_names)):
                Qd_di, Qd_err = calc_diurnal(self.datetime, Qd[self.sensor_names[n]].to_numpy())
                plt.plot(Qd_di)
            plt.legend(Gstr, loc='lower right')
            plt.xlabel('Hour of Day')
            plt.plot([0,24],[0,0],'k--')
            plt.ylabel(r'$Q_d$ '+r'(W/m$^2$)')

            # Scatter plots, ported over from a different method (so may
            #   need some work to get working)
            scat =1
            if scat:
                # Scatter plot of G vs. upper Q_nc:
                plt.figure()
                plt.scatter(eb['G'],hrly_Qnc[self.sensor_names[2]], 2, c='k')
                plt.xlabel('G')
                plt.ylabel(r'$Q_{nc}$')

                # Scatter plot of Rad vs. upper Q_nc:
                plt.figure()
                plt.scatter(eb['Rad'],hrly_Qnc[self.sensor_names[2]], 2, c='k')
                plt.xlabel('Rad')
                plt.ylabel(r'$Q_{nc}$')

                # Scatter plot of LE vs. upper Q_nc:
                plt.figure()
                plt.scatter(eb['LE'],hrly_Qnc[self.sensor_names[2]], 2, c='k')
                plt.xlabel('LE')
                plt.ylabel(r'$Q_{nc}$')

                # Scatter plot of Melt vs. upper Q_nc:
                plt.figure()
                plt.scatter(eb['Melt'],hrly_Qnc[self.sensor_names[2]], 2, c='k')
                plt.xlabel('Melt')
                plt.ylabel(r'$Q_{nc}$')

                plt.figure()
                plt.plot(hrly_Qnc[self.sensor_names[2]]/eb['LE'])

                plt.figure()
                plt.scatter(self.dTdz[self.sensor_names[1]],self.dqdz[self.sensor_names[1]], 2, c='k')
                plt.xlabel(r'$\partial T / \partial z$')
                plt.ylabel(r'$\partial Q_{nc} / \partial z$')

            return 

    def estimate_ke_Q_HOD_analy(self, direc=None):

        if direc == None:
            direc = self.plotdir+'/HOD/'
        
        self.estimate_ke_Q(display_hr=True, animate=False, plotdir=direc+'0_0/') #Data for all hours
        R2_mat = {}
        kappa_mat = {}
        for nn in range(1,self.num_sensors-1):
            R2_mat[self.sensor_names[nn]] = np.empty((24,24))
            kappa_mat[self.sensor_names[nn]]= np.empty((24,24))

        for hr1 in range(0,24):
            for hr2 in range(0,24):
                if hr1 == hr2:
                    for nn in range(1,self.num_sensors-1):
                        R2_mat[self.sensor_names[nn]][hr1,hr2] = np.nan
                        kappa_mat[self.sensor_names[nn]][hr1,hr2] = np.nan
                    continue
                self.estimate_ke_Q(display_hr=True, animate=False, hours_of_day=[hr1,hr2], plotdir=direc+str(hr1)+'_'+str(hr2)+'/')
                for nn in range(1,self.num_sensors-1):
                    R2_mat[self.sensor_names[nn]][hr1,hr2] = self.R2[nn]
                    kappa_mat[self.sensor_names[nn]][hr1,hr2] = self.ke[nn]

        # Determine best fits:
        best_hrs = {}
        best_R2 = np.zeros(len(self.sensor_names))
        best_kappa = np.zeros(len(self.sensor_names))
        for nn in range(1,self.num_sensors-1):
            best_R2[nn] = np.nanmax(R2_mat[self.sensor_names[nn]])
            best_hrs[self.sensor_names[nn]] = np.argwhere(R2_mat[self.sensor_names[nn]] == best_R2[nn])
            print(best_hrs[self.sensor_names[nn]])
            best_kappa[nn] = kappa_mat[self.sensor_names[nn]][best_hrs[self.sensor_names[nn]][0][0],best_hrs[self.sensor_names[nn]][0][1]]
        

        # Plot R2 & Kappa Matrices:
        for nn in range(1,self.num_sensors-1):
            plt.figure()
            plt.matshow(R2_mat[self.sensor_names[nn]], vmin=0, vmax=1)
            plt.xlabel('End Hour')
            plt.ylabel('Start Hour')
            plt.title('Depth = {:.1f} cm'.format(self.sensor_depth[nn]))
            plt.colorbar()
            plt.savefig(direc+'R2_mat_'+str(nn)+'.png',bbox_inches='tight')

            plt.figure()
            plt.matshow(kappa_mat[self.sensor_names[nn]], vmin=0, vmax=1e-06)
            plt.xlabel('End Hour')
            plt.ylabel('Start Hour')
            plt.title('Depth = {:.1f} cm'.format(self.sensor_depth[nn]))
            plt.colorbar()
            plt.savefig(direc+'kappa_mat_'+str(nn)+'.png',bbox_inches='tight')

            # Save to csv files:
            np.savetxt(direc+'R2_mat' + str(nn) +'.csv',R2_mat[self.sensor_names[nn]], delimiter=',')
            np.savetxt(direc+'kappa_mat' + str(nn) +'.csv',kappa_mat[self.sensor_names[nn]], delimiter=',')

        ke = best_kappa
        ke_z2 = np.zeros(len(self.sensor_names))

         # Calculate Q_c:
        Q_c = self.dTdz * 0
        for nn in range(1,len(self.sensor_names)):
            if nn==1:
                ke_z2[nn] = ke[nn]
            elif nn==len(self.sensor_names)-1:
                ke_z2[nn] = ke[nn-1]
            else:
                ke_z2[nn] = (ke[nn] + ke[nn-1])/2

            Q_c[self.sensor_names[nn]] = ke_z2[nn] * 1800 * 806 * -self.dTdz[self.sensor_names[nn]]
        
        self.Q_c = Q_c
        self.ke = ke
        self.ke_z2 = ke_z2
        self.R2 = best_R2
        self.R2_mat = R2_mat
        self.kappa_mat = kappa_mat

        # Determine dqdz, Qnc:

        self.estimate_dqdz(self.ke, rho=1800, c=806, show_plot=True)
        return

class iacs_debris_temp_profile(debris_temp_profile):

    """
    Class is same as debris temp profile, but loads data
    from 
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
            self.plotdir = None 
        data = pd.read_csv(datafile, sep='\t', header=None, index_col=0, parse_dates=True)
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
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

        # Downsample if desired:
        if downsample is not None:
            data = downsample_AWS(data, downsample)

        self.calc_derivatives(period, mask, smooth)

        return

def main():
    data_dir = './Debris_Temp/'
    P_2020a_dir = data_dir + 'P_2020a/'
    P_2020b_dir = data_dir + 'P_2020b/'

    summer_period = ['2020-07-27 00:00:00', '2020-09-06 12:25:00']
    plot_period = ['2020-07-28 00:00:00', '2020-08-05 00:00:00']
    plot_period_2011 = ['2011-08-08', '2011-08-17']
    pd0 = './Figures/'

    P_2020a = debris_temp_profile(P_2020a_dir, smooth=6, period=summer_period, plotdir=pd0+'P_2020a/')#, period=period2)#, mask=remove_period)
    P_2020a.plot_debris_temp(period=plot_period, show_plot=False)
    P_2020a.estimate_k_dkdz()
    P_2020a.plot_dqdz_Q(period=plot_period)
    P_2020a.plot_Q_diurnal(sunny_file=data_dir+'sky_conditions_2020.csv', Q_lim=[-40,100])

    P_2020b = debris_temp_profile(P_2020b_dir, smooth=6, period=summer_period, plotdir=pd0+'P_2020b/')
    P_2020b.plot_debris_temp(period=plot_period, dt='day', show_plot=False)
    P_2020b.estimate_k_from_melt(['2020-07-26 12:25:00','2020-09-06 12:25:00'], .4, 0.624)
    P_2020b.estimate_k_dkdz()
    P_2020b.plot_dqdz_Q(period=plot_period)

    P_2011 = iacs_debris_temp_profile(data_dir+'P_2011/Kennicott_red.txt', data_dir+'P_2011/Kennicott_red_depth.txt', smooth=6, plotdir=pd0+'P_2011/')
    P_2011.plot_debris_temp(period=plot_period_2011, dt="day", show_plot=False)
    P_2011.estimate_k_dkdz()
    P_2011.plot_dqdz_Q(period=plot_period_2011)
    P_2011.plot_Q_diurnal(sunny_file=data_dir+'sky_conditions_2011.csv', Q_lim=[-40,100])

    #plt.show()

if __name__ == "__main__":

    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    main()