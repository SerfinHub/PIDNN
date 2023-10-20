# LSTM PID implementation

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tqdm import tqdm  # Progress bar

# For scaling, feature selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

# For LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from keras.models import load_model

# For Gekko and TCLab
import tclab
from gekko import GEKKO

tclab_hardware = False
speedup = 100
mlab = tclab.setup(connected=False, speedup=speedup)  # Emulator


# Data for model to train
def generate_step_test_data():
    # generate step test data on TCLab
    global mlab
    filename = 'tclab_data.csv'

    n = 500
    tm = np.linspace(0, n * 2, n + 1)
    T1 = np.zeros(n + 1)
    T2 = np.zeros(n + 1)

    # heater steps
    Q1d = np.zeros(n + 1)
    Q1d[10:150] = 80
    Q1d[150:300] = 20
    Q1d[300:450] = 70
    Q1d[450:] = 50

    Q2d = np.zeros(n + 1)
    Q2d[50:150] = 35
    Q2d[150:250] = 95
    Q2d[250:350] = 25
    Q2d[350:] = 100

    p1 = 1 if tclab_hardware else 100
    # Connect to TCLab
    with mlab() as lab:
        # run step test
        i = 0
        for t in tclab.clock(tm[-1] + 1, 2):
            # set heater values
            lab.Q1(Q1d[i])
            lab.Q2(Q2d[i])
            T1[i] = lab.T1
            T2[i] = lab.T2
            if i % p1 == 0:
                print('Time: ' + str(2 * i) + \
                      ' Q1: ' + str(Q1d[i]) + \
                      ' Q2: ' + str(Q2d[i]) + \
                      ' T1: ' + str(round(T1[i], 2)) + \
                      ' T2: ' + str(round(T2[i], 2)))
            i += 1

    # write data to file
    fid = open(filename, 'w')
    fid.write('Time,Q1,Q2,T1,T2\n')
    for i in range(n + 1):
        fid.write(str(tm[i]) + ',' + str(Q1d[i]) + ',' + str(Q2d[i]) + ',' \
                  + str(T1[i]) + ',' + str(T2[i]) + '\n')
    fid.close()

    # read data file
    data = pd.read_csv(filename)

    # plot measurements
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(data['Time'], data['Q1'], 'r-', label='Heater 1')
    plt.plot(data['Time'], data['Q2'], 'b--', label='Heater 2')
    plt.ylabel('Heater (%)')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(data['Time'], data['T1'], 'r-.', label='Temperature 1')
    plt.plot(data['Time'], data['T2'], 'b-.', label='Temperature 2')
    plt.ylabel('Temperature (degC)')
    plt.legend(loc='best')
    plt.xlabel('Time (sec)')
    plt.savefig('tclab_data.png')
    plt.show()


def identify_model():
    #########################################################
    # Initialize Model
    #########################################################
    # load data and parse into columns
    data = pd.read_csv('tclab_data.csv')
    t = data['Time']
    u = data[['Q1', 'Q2']]
    y = data[['T1', 'T2']]

    # generate time-series model
    m = GEKKO(remote=False)

    ##################################################################
    # system identification
    na = 2
    nb = 2  # use 2nd order model
    print('Identify model')
    yp, p, K = m.sysid(t, u, y, na, nb, objf=10000, scale=False, diaglevel=0, pred='model')

    ##################################################################
    # plot sysid results
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, u)
    plt.legend([r'$Q_1$', r'$Q_2$'])
    plt.ylabel('MVs')
    plt.subplot(2, 1, 2)
    plt.plot(t, y)
    plt.plot(t, yp)
    plt.legend([r'$T_{1meas}$', r'$T_{2meas}$', r'$T_{1pred}$', r'$T_{2pred}$'])
    plt.ylabel('CVs')
    plt.xlabel('Time')
    plt.savefig('sysid.png')

    return yp, p, K


# Model predictive control
def mpc(m, T1, T1sp, T2, T2sp):
    # Insert measurements
    m.TC1.MEAS = T1
    m.TC2.MEAS = T2

    # Adjust setpoints
    db1 = 1.0  # dead-band
    m.TC1.SP = T1sp
    m.TC1.SPHI = T1sp + db1
    m.TC1.SPLO = T1sp - db1

    db2 = 0.2
    m.TC2.SP = T2sp
    m.TC2.SPHI = T2sp + db2
    m.TC2.SPLO = T2sp - db2

    # Adjust heaters with MPC
    m.solve(disp=False)

    if m.options.APPSTATUS == 1:
        # Retrieve new values
        Q1 = m.Q1.NEWVAL
        Q2 = m.Q2.NEWVAL
    else:
        # Solution failed
        Q1 = 0.0
        Q2 = 0.0
    return [Q1, Q2]


def initialize_controller(p):
    ##################################################################
    # create control ARX model
    print("Initialize controller")
    m = GEKKO(remote=False)
    m.y = m.Array(m.CV, 2)
    m.u = m.Array(m.MV, 2)
    m.arx(p, m.y, m.u)

    # rename CVs
    m.TC1 = m.y[0]
    m.TC2 = m.y[1]

    # rename MVs
    m.Q1 = m.u[0]
    m.Q2 = m.u[1]

    # steady state initialization
    m.options.IMODE = 1
    m.solve(disp=False)

    # set up MPC
    m.options.IMODE = 6  # MPC
    m.options.CV_TYPE = 2  # Objective type
    m.options.NODES = 2  # Collocation nodes
    m.options.SOLVER = 1  # APOPT
    m.time = np.linspace(0, 60, 31)

    # Manipulated variables
    m.Q1.STATUS = 1  # manipulated
    m.Q1.FSTATUS = 0  # not measured
    m.Q1.DMAX = 100.0
    m.Q1.DCOST = 2.0
    m.Q1.UPPER = 100.0
    m.Q1.LOWER = 0.0

    m.Q2.STATUS = 0  # manipulated, turn off Q2
    m.Q2.FSTATUS = 1  # use measured value
    m.Q2.DMAX = 100.0
    m.Q2.DCOST = 2.0
    m.Q2.UPPER = 100.0
    m.Q2.LOWER = 0.0
    m.Q2.MEAS = 0  # set Q2=0

    # Controlled variables
    m.TC1.STATUS = 1  # drive to set point
    m.TC1.FSTATUS = 1  # receive measurement
    m.TC1.TAU = 8  # response speed (time constant)
    m.TC1.TR_INIT = 2  # reference trajectory
    m.TC1.TR_OPEN = 5

    m.TC2.STATUS = 0  # drive to set point
    m.TC2.FSTATUS = 1  # receive measurement
    m.TC2.TAU = 8  # response speed (time constant)
    m.TC2.TR_INIT = 2  # dead-band
    m.TC2.TR_OPEN = 1  # for CV_TYPE=1


def generate_data_for_training_lstm():
    ##### Set up run parameters #####
    global mlab
    run_time = 90.0  # minutes

    loops = int(30.0 * run_time + 1)  # cycles (2 sec each)

    # arrays for storing data
    T1 = np.zeros(loops)  # measured T (degC)
    T2 = np.zeros(loops)  # measured T (degC)
    Q1 = np.zeros(loops)  # Heater values
    Q2 = np.zeros(loops)  # Heater values
    tm = np.linspace(0, 2 * (loops - 1), loops)  # Time

    # Temperature set point (degC)
    with mlab() as lab:
        Tsp1 = np.ones(loops) * lab.T1
        Tsp2 = np.ones(loops) * lab.T2

    # vary temperature setpoint
    end = 2  # leave first couple cycles of temp set point as room temp
    while end <= loops:
        start = end
        # keep new temp set point value for anywhere from 3 to 5 min
        end += random.randint(90, 150)
        Tsp1[start:end] = random.randint(30, 70)
    Tsp1[-120:] = Tsp1[0]  # last 4 minutes at room temperature

    if tclab_hardware:
        # print every cycle with hardware
        p1 = 10
        p2 = 1
    else:
        # print 20x less with emulator
        p1 = 200
        p2 = 20

        # Plot
    plt.plot(tm, Tsp1, 'b.-')
    plt.xlabel('Time', size=14)
    plt.ylabel(r'Temp SP ($^oC$)', size=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.savefig('SP_profile.png')

    return T1, T2, Q1, Q2, tm, Tsp1, Tsp2, p1, p2


def data_collection(tm, T1, T2, Q1, Q2, m, Tsp1, Tsp2, p1, p2):
    # Data collection
    global mlab
    with mlab() as lab:
        # Find current T1, T2
        print('Temperature 1: {0:0.2f} °C'.format(lab.T1))
        print('Temperature 2: {0:0.2f} °C'.format(lab.T2))

        i = 0
        for t in tclab.clock(tm[-1] + 1, 2):
            # Read temperatures in Celcius
            T1[i] = lab.T1
            T2[i] = lab.T2

            # Calculate MPC output every 2 sec
            try:
                [Q1[i], Q2[i]] = mpc(m, T1[i], Tsp1[i], T2[i], Tsp2[i])
            except:
                Q1[i] = 0
                Q2[i] = 0  # catch any failure to converge
            # Write heater output (0-100)
            lab.Q1(Q1[i])
            lab.Q2(Q2[i])

            if i % p1 == 0:
                print('  Time_____Q1___Tsp1_____T1______Q2____Tsp2_____T2')
            if i % p2 == 0:
                print('{:6.1f} {:6.2f} {:6.2f} {:6.2f}  {:6.2f}  {:6.2f} {:6.2f}'.format(tm[i], Q1[i], Tsp1[i], T1[i],
                                                                                         Q2[i], Tsp2[i], T2[i]))
            i += 1

    return i


def save_csv_file(tm, Q1, T1, Tsp1, i):
    # Save csv file
    df = pd.DataFrame()
    df['time'] = tm[:i]
    df['Q1'] = Q1[:i]
    df['T1'] = T1[:i]
    df['Tsp'] = Tsp1[:i]
    df.set_index('time', inplace=True)
    df.to_csv('MPC_train_data.csv', index=False)

    # Plot
    df[['Q1', 'T1', 'Tsp']].plot()
    plt.savefig('MPC_train.png')
    plt.show()
    df.head()


def create_new_feature(df):
    # Create new feature: setpoint error
    df['err'] = df['Tsp'] - df['T1']

    # Load possible features
    X = df[['T1', 'Tsp', 'err']]
    y = np.ravel(df[['Q1']])

    # SelectKBest feature selection
    bestfeatures = SelectKBest(score_func=f_regression, k='all')
    fit = bestfeatures.fit(X, y)
    plt.bar(x=X.columns, height=fit.scores_)


def feature_selection(df):
    X = df[['Tsp', 'err']].values
    y = df[['Q1']].values

    # Scale data
    s_x = MinMaxScaler()
    Xs = s_x.fit_transform(X)

    s_y = MinMaxScaler()
    ys = s_y.fit_transform(y)

    # Each input uses last 'window' number of Tsp and err to predict the next Q1
    window = 15
    X_lstm = []
    y_lstm = []
    for i in range(window, len(df)):
        X_lstm.append(Xs[i - window:i])
        y_lstm.append(ys[i])

    # Reshape data to format accepted by LSTM
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    # Split into train and test
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)

    return Xtrain, Xtest, ytrain, ytest


def keras_lstm(Xtrain, ytrain):
    # Keras LSTM model
    model = Sequential()

    # First layer specifies input_shape and returns sequences
    model.add(LSTM(units=100, return_sequences=True,
                   input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    # Dropout layer to prevent overfitting
    model.add(Dropout(rate=0.1))

    # Last layer doesn't return sequences (middle layers should return sequences)
    model.add(LSTM(units=100))
    model.add(Dropout(rate=0.1))

    # Dense layer to return prediction
    model.add(Dense(1))

    # Compile model; adam optimizer, mse loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    result = model.fit(Xtrain, ytrain, verbose=0, validation_split=0.2,
                       callbacks=[TqdmCallback(verbose=1)],  # es
                       batch_size=100,
                       epochs=300)

    # Plot loss and save model
    epochs = es.stopped_epoch
    plt.semilogy(result.history['loss'], label='loss')
    plt.semilogy(result.history['val_loss'], label='val_loss')
    plt.legend()

    model.save('lstm_control.h5')


def predict_using_lstm(model, Xtest, ytest, s_x, s_y):
    # Predict using LSTM
    yp_s = model.predict(Xtest)

    # Unscale data
    Xtest_us = s_x.inverse_transform(Xtest[:, -1, :])
    ytest_us = s_y.inverse_transform(ytest)
    yp = s_y.inverse_transform(yp_s)

    # Derive Tsp (sp) and T1 (pv) from X data
    sp = Xtest_us[:, 0]
    pv = Xtest_us[:, 0] + Xtest_us[:, 1]

    # Plot SP, MPC response, and LSTM response
    tm = np.linspace(0, 2 * (len(sp) - 1) / 60, len(sp))
    plt.plot(tm, sp, 'k-', label='$SP$ $(^oC)$')
    plt.plot(tm, pv, 'r-', label='$T_1$ $(^oC)$')
    plt.plot(tm, ytest_us, 'b-', label='$Q_{MPC}$ (%)')
    plt.plot(tm, yp, 'g-', label='$Q_{LSTM}$ (%)')
    plt.legend()
    plt.xlabel('Time (min)')
    plt.ylabel('Value')
    plt.show()


def generate_sp_data_for_test(window):
    # Run time in minutes
    run_time = 45.0

    # Number of cycles
    loops = int(30.0 * run_time)

    # arrays for storing data
    T1 = np.zeros(loops)  # measured T (degC)
    T2 = np.zeros(loops)
    Q1mpc = np.zeros(loops)  # Heater values for MPC controller
    Q2mpc = np.zeros(loops)
    Qlstm = np.zeros(loops)  # Heater values for LSTM controller
    tm = np.linspace(0, 2 * (loops - 1), loops)  # Time

    # Temperature set point (degC)
    with mlab() as lab:
        Tsp1 = np.ones(loops) * lab.T1
        Tsp2 = np.ones(loops) * lab.T2

    # vary temperature setpoint
    end = window + 5  # leave 1st window + 10 seconds of temp set point as room temp
    while end <= loops:
        start = end
        # keep new temp set point value for anywhere from 3 to 5 min
        end += random.randint(90, 150)
        Tsp1[start:end] = random.randint(30, 70)

    # leave last 120 seconds as room temp
    Tsp1[-60:] = Tsp1[0]
    plt.plot(Tsp1)
    plt.show()


def lstm_controller(s_x, s_y, model):
    # LSTM Controller
    def lstm(T1_m, Tsp_m):
        # Calculate error (necessary feature for LSTM input)
        err = Tsp_m - T1_m

        # Format data for LSTM input
        X = np.vstack((Tsp_m, err)).T
        Xs = s_x.transform(X)
        Xs = np.reshape(Xs, (1, Xs.shape[0], Xs.shape[1]))

        # Predict Q for controller and unscale
        Q1c_s = model.predict(Xs)
        Q1c = s_y.inverse_transform(Q1c_s)[0][0]

        # Ensure Q1c is between 0 and 100
        Q1c = np.clip(Q1c, 0.0, 100.0)
        return Q1c


def run_test(T1, T2, tm, Q1mpc, Q2mpc, m, Tsp1, Tsp2, p1, p2, window, Qlstm, lstm):
    # Run test
    with mlab() as lab:
        # Find current T1, T2
        print('Temperature 1: {0:0.2f} °C'.format(lab.T1))
        print('Temperature 2: {0:0.2f} °C'.format(lab.T2))

        i = 0
        for t in tclab.clock(tm[-1] + 1, 2):

            # Read temperatures in Celcius
            T1[i] = lab.T1
            T2[i] = lab.T2

            # Calculate MPC output every 2 sec
            try:
                [Q1mpc[i], Q2mpc[i]] = mpc(m, T1[i], Tsp1[i], T2[i], Tsp2[i])
            except:
                Q1mpc[i] = 0
                Q2mpc[i] = 0
            # Write heater output (0-100)
            lab.Q1(Q1mpc[i])
            lab.Q2(Q2mpc[i])

            if i % p1 == 0:
                print('  Time_____Q1___Tsp1_____T1')
            if i % p2 == 0:
                print(('{:6.1f} {:6.2f} {:6.2f} {:6.2f}').format( \
                    tm[i], Q1mpc[i], Tsp1[i], T1[i]))

            # Run LSTM model to get Q1 value for control
            if i >= window:
                # Load data for model
                T1_m = T1[i - window:i]
                Tsp_m = Tsp1[i - window:i]
                # Predict and store LSTM value for comparison
                Qlstm[i] = lstm(T1_m, Tsp_m)
            i += 1


def plot_test_a(tm, Tsp1, T1, Q1mpc, Qlstm, i):
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(tm[:i], Tsp1[:i], 'k-', label='SP $(^oC)$')
    plt.plot(tm[:i], T1[:i], 'r-', label='$T_1$ $(^oC)$')
    plt.legend(loc='upper right', fontsize=14)
    plt.ylim((0, 100))
    plt.xlabel('Time (s)', size=14)
    plt.subplot(2, 1, 2)
    plt.plot(tm[:i], Q1mpc[:i], 'b-', label='$Q_{MPC}$ (%)')
    plt.plot(tm[:i], Qlstm[:i], 'g-', label='$Q_{LSTM}$ (%)')
    plt.legend(loc='upper right', fontsize=14)
    plt.ylim((0, 100))
    plt.xlabel('Time (s)', size=14)
    plt.show()


def run_lstm(tm, T1, T2, window, Tsp1, Qlstm, lstm, p1, p2):
    # Run test
    global mlab
    with mlab() as lab:
        # Find current T1, T2
        print('Temperature 1: {0:0.2f} °C'.format(lab.T1))
        print('Temperature 2: {0:0.2f} °C'.format(lab.T2))

        i = 0
        for t in tclab.clock(tm[-1] + 1, 2):
            # Read temperatures in Celcius
            T1[i] = lab.T1
            T2[i] = lab.T2

            # Run LSTM model to get Q1 value for control
            if i >= window:
                # Load data for model
                T1_m = T1[i - window:i]
                Tsp_m = Tsp1[i - window:i]
                # Predict and store LSTM value for comparison
                Qlstm[i] = lstm(T1_m, Tsp_m)

            if i % p1 == 0:
                print('  Time_____Q1___Tsp1_____T1')
            if i % p2 == 0:
                print('{:6.1f} {:6.2f} {:6.2f} {:6.2f}'.format(tm[i], Qlstm[i], Tsp1[i], T1[i]))

            # Write heater output (0-100)
            lab.Q1(Qlstm[i])
            i += 1


def print_lsm(tm, Tsp1, T1, Qlstm, i):
    plt.figure(figsize=(10, 4))
    plt.plot(tm[:i], Tsp1[:i], 'k-', label='SP $(^oC)$')
    plt.plot(tm[:i], T1[:i], 'r-', label='$T_1$ $(^oC)$')
    plt.plot(tm[:i], Qlstm[:i], 'g-', label='$Q_{LSTM}$ (%)')
    plt.legend()
    plt.ylim((0, 100))
    plt.xlabel('Time (s)')
    plt.grid()
    plt.show()


def main():
    # generate_step_test_data()
    yp, p, K = identify_model()
    # initialize_controller(p)
    T1, T2, Q1, Q2, tm, Tsp1, Tsp2, p1, p2 = generate_data_for_training_lstm()


if __name__ == "__main__":
    main()
