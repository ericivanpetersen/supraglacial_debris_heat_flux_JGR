import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt 
from datetime import datetime
from datetime import timedelta
import os
from angular_functions import *

def conform_records(df1, df0):
    """
    Conforms data records in df1 to have the same 
    timestamps and cropped extent as those in df0
    """

    df_comb = pd.merge(df1, df0, left_index=True, right_index=True, how="right")
    df1_conf = df_comb[df1.columns]

    return df1_conf

def combine_aws_recs(filelist):
    """Program that reads in multiple data files from the same weather station,
    written in TOA5 standard as output by Campbell Software, and compiles them
    into the same dataframe."""

    data = {}
    n=0
    for filename in filelist:
        data[filename] = pd.read_csv(filename,header=[1],skiprows=[2,3])
        if n==0:
            cat_data = data[filename]
        else:
            cat_data = pd.concat([cat_data, data[filename]])
        n=n+1

    strfmt = '%Y-%m-%d %H:%M:%S'
    dtstring = cat_data[cat_data.columns[0]] # Pick out datettime string
    dt = np.asarray([datetime.strptime(element, strfmt) for element in dtstring]) # convert to datetime object
    dt_complete = pd.date_range(start=dt[0], end=dt[-1], freq=(dt[1]-dt[0]))
    dummy_dt = pd.DataFrame(data={'A': np.full(np.size(dt_complete), 0)}, index=dt_complete)
    cat_data.index = dt # Set datetime as index
    cat_data_filled = pd.merge(left=dummy_dt, right=cat_data, how='left', left_index=True, right_index=True)

    return cat_data_filled

def calc_temp_blackbody(rad, LW_downwell=0, eps=1):
    """
    rad = radiation in W/m^2
    eps = emmissivity, 1 by default

    returns temp in degrees C
    """
    sigma = 5.670373 * (10**-8) #Stefan-Boltzmann constant

    temp = ( (rad - (1-eps)* LW_downwell)/(sigma*eps))**(1/4)-273.15

    return temp

def calc_specific_humidity(RH, P, T):
    """Calculate specific humidity q 
    from relative humidity, pressure in Pa,
    and temperature in degrees C"""

    T_K = T + 273.15

    #es = mixing_ratio_from_relative_humidity(RH,T_K,P)
    #q = specific_humidity_from_mixing_ratio(es)
    q = RH * np.exp(17.67 * (T_K - 273.16)/(T_K - 29.65)) / (0.263 * P)

    return q

def correct_sonic_temp(T_sonic, q):
    """Correct sonic temp using input
    specific humidity"""

    T_corr = T_sonic * (1 / (1+0.51*q))
    return T_corr

def calc_emissivity(LW_upwell,LW_downwell,ground_temp,plot=1):
    """
    Determines the emissivity from 
    measurements of ground temperature in C 
    and Longwave upwelling radiation in W/m^2
    """

    sigma = 5.670373 * (10**-8) #Stefan-Boltzmann constant

    eqn = pd.DataFrame()

    eqn['RHS'] = sigma * (ground_temp+273.15)**4 - LW_downwell
    eqn['LHS'] = LW_upwell - LW_downwell


    emissivity_proj = transfer_function(eqn['RHS'], eqn['LHS'])
    emissivity_proj.LinearModel(intercept=False)

    eps = emissivity_proj.model.coef_[0][0]
    r2 = emissivity_proj.R2

    temp_rad = calc_temp_blackbody(LW_upwell,LW_downwell=LW_downwell, eps=eps)

    if plot:
        plt.figure()
        plt.plot(ground_temp)
        plt.plot(temp_rad)
        plt.legend(['Ground Temp', 'Temp from '+r'$LW_{upwelling}$'+' '+r'($\epsilon=${:.3f})'.format(eps)])
        plt.ylabel(r'$\circ$C')
        plt.xlabel('Date')

        plt.figure()
        plt.scatter(emissivity_proj.X, emissivity_proj.Y)
        plt.xlabel(r'$\sigma T^4-LW_{downwelling}$' +' ' + r'$(W/m^2)$')
        plt.ylabel(r'$LW_{upwelling}-LW_{downwelling}$'+ ' ' + r'$(W/m^2)$')
        label_string = r'$\epsilon$ = {:.3f}; $R^2$ = {:.3f}'.format(eps, r2)
        plt.text(np.min(emissivity_proj.X),np.max(emissivity_proj.Y), label_string)
        plt.show()

    return eps

def plot_timeseries(dataframe, fields, ylbl, legend=None, period=None, loc='upper right', colors=None):

    n = 0
    for field in fields:
        if colors is not None:
            plt.plot(dataframe[field], c=colors[n])
        else:
            plt.plot(dataframe[field])
        n=n+1
    plt.ylabel(ylbl)
    if period is not None:
        plt.xlim(period)

    if len(fields) > 1:
        if legend is not None:
            plt.legend(legend, loc=loc)
        else:
            plt.legend(fields, loc=loc)

def calc_diurnal(datetime, value):
    """Calculate diurnal cycle of values in timeseries with
    provided datetime

    :param list datetime: list of datetime objects for the time series given
    :param float value: value desired to diurnally average
    :output float diurnal_value: diurnally averaged value
    """
    hours = np.asarray([dat.hour for dat in datetime])
    diurnal_value = np.empty(24)
    diurnal_std = np.empty(24)
    for nn in range(24):
        ind = np.where((hours == float(nn)))
        diurnal_value[nn] = np.nanmean(value[ind])
        diurnal_std[nn] = np.nanstd(value[ind])

    return diurnal_value, diurnal_std

def calc_monthly(datetime, value):
    """Calculate monthly averages of values in timeseries with 
    provided datetime

    :param list datetime: list of datetime objects
    :param float value: value desired to monthly average
    :output float monthly_value: monthly average value
    :output float monthly_std: monthly stdev
    """
    months = np.asarray([dat.month for dat in datetime])
    monthly_value = np.empty(12)
    monthly_std = np.empty(12)
    for nn in range(12):
        ind = np.where((months == float(nn+1)))
        monthly_value[nn] = np.mean(value[ind])
        monthly_std[nn] = np.std(value[ind])

    return monthly_value, monthly_std

def calc_pdds(dt, temp):

    year = np.array([dat.year for dat in dt])
    month = np.array([dat.month for dat in dt])
    day = np.array([dat.day for dat in dt])
	
    NN = len(dt)
    pdd = []
    dates = []
    nn = 0
    while nn < NN-1:
        dind = (year == dt[nn].year) & (month == dt[nn].month) & (day == dt[nn].day)
        daytemp = np.mean( temp[dind] )
        if daytemp > 0:
            pdd.append( daytemp )
        else:
            pdd.append( 0 )
        dates.append(datetime.date(dt[nn]))#.year, dt[nn].month, dt[nn].day)
        nn = int(max(np.argwhere(dind))+1)
    pdd = np.asarray(pdd)
    return pdd, dates

def downsample_AWS(AWS_orig, downsample_freq, angular_fields=None):
    """
    Function to downsample AWS data to 
    desired timescale using averaging, on
    each measurement made.

    angular_fields is used to indicate, in a list of strings,
    the fields/columns in AWS_orig which are expressed as 
    degrees, i.e. wind direction, thus need to be calculated as 
    an angular/vectorial mean.
    """

    AWS_downsampled = pd.DataFrame()
    for column in AWS_orig.columns:
        if AWS_orig[column].dtype=='float':
            if angular_fields is not None:
                if column in angular_fields:
                    AWS_downsampled[column] = AWS_orig[column].resample(downsample_freq).apply(angular_mean)
                else: AWS_downsampled[column] = AWS_orig[column].resample(downsample_freq).mean()
            else: AWS_downsampled[column] = AWS_orig[column].resample(downsample_freq).mean()

    return AWS_downsampled

def upsample_AWS(AWS_orig, upsample_freq):
    """
    Function to upsample AWS data and
    interpolate the values
    """

    AWS_upsampled = pd.DataFrame()
    for column in AWS_orig.columns:
        if AWS_orig[column].dtype=='float64':
            AWS_upsampled[column] = AWS_orig[column].resample(upsample_freq).interpolate(method='linear')
    
    return AWS_upsampled

class transfer_function(object):
    """
    Function to manage variables and 
    produce a transfer function for 
    time-series data. 
    """

    def __init__(self, X, Y, modelfile=None):
        """
        :param dataframe X: X variable(s) values
        :param dataframe Y: Y variable values

        both X & Y are pandas dataframes
        with datetime indices (best to average to 
        a standard time interval such as Hourly
        so that they are merged correctly)
        """

        # Manage X & Y variables
        # Drop NaN values:
        self.X = X.dropna()
        self.Y = Y.dropna()
        XY = pd.merge(X.dropna(), Y.dropna(), left_index=True, right_index=True) #Merge X with Y/get referenced to same time
        #self.X = XY[XY.columns[:-1]]
        #self.Y = XY[XY.columns[-1]]
        self.X_len = XY.shape[1]-1 # Number of X variables
        if self.X_len > 1:
            print("Loading {} X Variables: {}".format(self.X_len, X.columns))
        self.X_fit = XY[XY.columns[:-1]].values.reshape(-1,self.X_len) # X array for fitting
        self.Y_fit = XY[XY.columns[-1]].values.reshape(-1,1) # Y array for fitting

        # Stats for nice plotting, etc:
        self.x_max = np.max(self.X)
        self.x_min = np.min(self.X)
        self.y_max = np.max(self.Y)
        self.y_min = np.min(self.Y)
        self.x_inc = (self.x_max - self.x_min)/20
        self.y_inc = (self.y_max - self.y_min)/20
        self.x_lim1 = self.x_min - self.x_inc
        self.x_lim2 = self.x_max + self.x_inc
        self.y_lim1 = self.y_min - self.y_inc
        self.y_lim2 = self.y_max + self.y_inc
        #self.y_lim1 = np.mean(self.Y) - 4*np.std(self.Y)
        #self.y_lim2 = np.mean(self.Y) + 4*np.std(self.Y)

        if modelfile:
            self.model = joblib.load(modelfile)
            self.Y_pred_dt = self.X_fit.index
            self.Y_pred = self.model.predict(X_fit)

    def RandomForest(self, ts=0.2, rs=123, ne=100, shuffle_on=False):
        """
        Preform Random Forest Regression
        on data and produce predictions
        """
        # Split data into training & test sets:
        X_train, X_test, y_train, y_test = train_test_split(self.X_fit, self.Y_fit, test_size=ts, shuffle=shuffle_on, random_state=rs)

        #if shuffle_on is False:
            # Display time periods for test and train data:
            #print("Training Period: {} to {}".format(X_train.index[0], X_train.index[-1]))
            #print("Test Period: {} to {}".format(X_test.index[0], X_test.index[-1]))

        # Normalize X variables:
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create processing pipeline and fit the model:
        pipeline = Pipeline([('scalar', preprocessing.StandardScaler()), ('clf', RandomForestRegressor(n_estimators=ne))])
        #pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=ne))
        hyperparameters = { 'clf__max_features' : ['auto', 'sqrt', 'log2'], 'clf__max_depth' : [None, 5, 3, 1]}
        clf = GridSearchCV(pipeline, hyperparameters, cv=10) #split dataset into 10 subsets for cross-validation
        clf.fit(X_train, y_train)

        Y_pred_test = clf.predict(X_test)
        Y_train_pred = clf.predict(X_train)
        print("R2 = {:.2f}".format( r2_score(y_test, Y_pred_test)))
        print("MSE = {:.2f}".format(mean_squared_error(y_test, Y_pred_test)))

        self.y_train = y_train
        self.Y_train_pred = Y_train_pred
        self.y_test = y_test
        self.Y_pred_test = Y_pred_test
        self.model = clf
        self.Y_pred = clf.predict(self.X)
        self.Y_pred_dt = self.X.index
        self.R2 = r2_score(y_test, Y_pred_test)
        self.MSE = mean_squared_error(y_test, Y_pred_test)

    def plot_results(self):
        R2_train = r2_score(self.y_train, self.Y_train_pred)
        # Plot Predicted vs Data:
        plt.figure()
        ax=plt.subplot(1,2,1)
        plt.title('Trainer')
        plt.plot(self.y_train, self.Y_train_pred, 'k.')
        txt1 = r'$R^2$' + ' = {:.3f}'.format(R2_train)
        plt.text(0.05, 0.95, txt1, transform=ax.transAxes)
        plt.ylabel('Predicted')
        plt.xlabel('Data')
        ax=plt.subplot(1,2,2)
        plt.title('Test')
        plt.plot(self.y_test, self.Y_pred_test, 'k.')
        txt2 = r'$R^2$' + ' = {:.3f}'.format(self.R2)
        plt.text(0.05, 0.95, txt2, transform=ax.transAxes)
        plt.xlabel('Data')
        #plt.show()

    def LinearModel(self, intercept=True):
        """
        Perform Multiple Linear Regression on
        data and produce predictions
        """

        reg = LinearRegression(fit_intercept=intercept).fit(self.X_fit, self.Y_fit) # Linear Regression
        
        # Assign variables to self:
        self.model = reg # Fit
        self.R2 = reg.score(self.X_fit, self.Y_fit) #R2 score 
        self.Y_pred = reg.predict(self.X.values.reshape(-1,self.X_len))
        self.Y_pred_dt = self.X.index
        self.Y_pred_fit = reg.predict(self.X_fit)

        print("R2 = {}".format(self.R2))
        print("Coeffs = {}".format(self.model.coef_))
        print("Intercepts = {}".format(self.model.intercept_))

    def save_model(self, modelfile):
        """ Saves model to pickle"""

        joblib.dump(self, modelfile)

def read_hobo(hobo_file, strfmt='%m/%d/%y %H:%M:%S'):
    """
    Function to read hobo sensor data
    stored in csv format, to pandas dataframe
    and manage date/time
    """

    data = pd.read_csv(hobo_file, skiprows=1) # read file to dataframe
    dtstring = data[data.columns[1]] # Pick out datettime string
    try: 
        dt = np.asarray([datetime.strptime(element, strfmt) for element in dtstring]) # convert to datetime object
    except ValueError:
        strfmt='%m/%d/%y %I:%M:%S %p'
        dt = np.asarray([datetime.strptime(element, strfmt) for element in dtstring]) # convert to datetime object
    except:
        print("Error parsing date/time using default format")
    data.index = dt # Set datetime as index

    return data

def transfer_AWS_data(aws_full, aws_partial, proj_dir, period=None):
    """
    Method to develop a transfer function between
    all meteorological measurements at one aws
    (aws_full) and each metereological measurement
    at another (aws_partial), then produce a 
    synthetic aws record for aws_partial

    :param dataframe aws_full:
    :param dataframe aws_partial:
    """

    # Create Project Directories:
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)
        os.makedirs(proj_dir + '/model_pickles/')    
        os.makedirs(proj_dir + '/figures/')

    models = {}
    aws_synth = pd.DataFrame()
    R2_MSE = pd.DataFrame() # R2 and MSE dataframe
    FI = pd.DataFrame() # Feature importance dataframe
    for column in aws_partial.columns:
        print("Assessing {}".format(column))
        pkl_file = proj_dir + '/model_pickles/' + column + '.pkl'
        if os.path.exists(pkl_file):
            print("    Pickled Model exists, reading in {}.pkl".format(column))
            tmp_mod = joblib.load(pkl_file)
        else:
            print("    Performing Regression...")
            tmp_mod = transfer_function(aws_full, aws_partial[column])
            tmp_mod.RandomForest()
            # Save Model Information:
            tmp_mod.save_model(pkl_file)     

        # Plot results:
        tmp_mod.plot_results()
        plt.suptitle(column)
        plt.savefig(proj_dir + '/figures/scatter_' + column + '.png')
        aws_synth[column] = pd.Series(tmp_mod.Y_pred, tmp_mod.Y_pred_dt)

        FI[column] = tmp_mod.model.best_estimator_.named_steps['clf'].feature_importances_
        R2_MSE[column] = [tmp_mod.R2, tmp_mod.MSE]

    FI.index = aws_full.columns
    R2_MSE.index = ['R2','MSE']
    R2_MSE = R2_MSE.T #transpose

    aws_synth.to_csv(proj_dir + '/synthetic_record.csv') # Save fully synthetic data record
    # Produce in-filled data record
    aws_partial['Orig_Data'] = np.ones(len(aws_partial))
    aws_synth['Orig_Data'] = np.zeros(len(aws_synth))
    aws_infill = aws_partial.combine_first(aws_synth)
    aws_infill.to_csv(proj_dir + '/infilled_record.csv') # Save infilled data record
    R2_MSE.to_csv(proj_dir + '/R2_MSE.csv') # Save R2 & MSE scores
    FI.to_csv(proj_dir + '/FI.csv')
    print("Saved Data")
    #plt.show()

    return

def extrap_MCKA2():
    proj_dir = '/Users/ericpetersen/Documents/Kennicott_Project/Field_Data/AWS_Data/'
    aws_debris_file = proj_dir + 'AWS_Debris_2021/CR1000/09_11/EKT_MetData_2021-09-11T21-28.dat'
    MCKA2_file = proj_dir + 'May_Creek/raw_downloads/MCKA2_2021.csv'

    aws_debris = pd.read_csv(aws_debris_file, header=1, skiprows=[2,3], index_col=0, parse_dates=True)
    aws_debris.index = aws_debris.index.tz_localize('America/Anchorage')
    mcka2 = pd.read_csv(MCKA2_file, header=[6], skiprows=[7], index_col=1, parse_dates=True)
    
    aws_debris = downsample_AWS(aws_debris, 'H')
    mcka2 = downsample_AWS(mcka2, 'H')
    debris_T = aws_debris['AirTemp_C_2m_Avg']
    mcka2_T = (mcka2['air_temp_set_1'] - 32) * 5/9 #convert to celsius
    print(debris_T)
    print(mcka2_T)

    transfer = transfer_function(mcka2_T, debris_T)
    transfer.LinearModel()
    aws_synth = pd.DataFrame()
    aws_part = pd.DataFrame()
    aws_part['T'] = pd.Series(debris_T, index=debris_T.index)
    aws_synth['T'] = pd.Series(transfer.Y_pred.flatten(), index=transfer.Y_pred_dt)
    aws_part['Orig_Data'] = np.ones(len(aws_part))
    aws_synth['Orig_Data'] = np.zeros(len(aws_synth))
    print(aws_synth)
    aws_filled = aws_part.combine_first(aws_synth)
    aws_filled.to_csv(proj_dir + 'AWS_Debris_2021/Temp_Record_MCKA2_Fill.csv')

    plt.figure()
    plt.plot(mcka2_T)
    plt.plot(aws_synth['T'])
    plt.plot(aws_part['T'])
    plt.legend(['May Creek','Filled','Orig'])
    plt.show()
    

#def main():
    

if __name__ == "__main__":
    extrap_MCKA2()