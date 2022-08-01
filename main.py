import models
import pickle
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import random as rd
import random
import gc
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sn

# --------- LORENZ MODEL 3 parameters -----------------------------------

pars = {'N' : 20,
        'K' : 2,
        'F' : 30
        }

dt = 1e-3
t_stop = 1e3

# init cond after 1e4 time for N=20,K=2,F=30
init_cond = [-17.16677811 ,  2.77684494,   5.32918288,   4.06720654,  -4.50697691,
              5.28285585  , 0.25784428 ,  5.83645619 , 16.70014973 ,  4.11380002,
             -1.95477235  ,-5.53403114 , -2.23405133 , 11.30964113 , -7.40045119,
              8.4124894   ,22.06908642 , 13.63659989 , 16.6942645  , -7.06811886]


# PRINT INIT COND AS THE LAST POINT OF SOME EVAL
# with open('data/LM05_{}.pkl'.format(pars), 'rb') as inp:
#     L = pickle.load(inp) # Lorenz_model_3_evaluated
#     L.print_parameters()
#
#     eval_path_num = -1
#     eval_path = L.evaluation_paths[eval_path_num]
#
#     with open(eval_path, 'rb') as LM05_inp_eval:
#         ev = pickle.load(LM05_inp_eval)
#         print(ev.y.T[-1])
#         del LM05_inp_eval

# Init cond for N=20,K=3,I=1 (last point after 1000 time units of integration)
#init_cond = [-6.87498985, -7.73416789,  6.53112349, 15.33917886,  0.61888726,  3.09367241, 13.01554729,
#             4.22892966,  -3.24957584, -0.71020495,  0.31950248,  1.35676813,  2.02681408,  5.0922673 ,
#             12.74492152, 17.25554894,  4.75553264,  0.5567377,   3.22637389,  1.34066901]


# with open("data/eval_[240, 24, 30, 0.001, 100]_f6a0c5a21a53e9dce0bcb42d48e497e5863ff7ac.pkl", 'rb') as LM05_inp_eval:
#     ev = pickle.load(LM05_inp_eval)
#     print(ev.y.T[-1])
#     del LM05_inp_eval

# init cond after 1e3 time for N=40,K=4,F=30
# init_cond = [  0.48318681,   2.95424609,   6.52282199,   7.20527187,
#         -3.96403755,  -5.47662056,   1.92791392,   4.32702182,
#          7.82995316,   8.69806238,   3.92827945,   2.92003026,
#          0.54267392,  -5.21726865,  -5.19664442,  -2.08799543,
#          1.20924886,   3.1653867 ,   1.75996026,  -8.86433605,
#        -16.77190089, -14.60236073,  -9.11398247, -14.37423936,
#        -17.31203336,  -7.19224751,   5.85510574,  11.08874374,
#          8.2646984 ,  10.94253367,  22.50197604,  28.77939118,
#         21.85693002,  14.40939591,   7.82256211,   1.69014723,
#          1.63308663,   6.82762937,   6.55133002,   0.94571574]





# init_cond after 1e3 time for N=100,K=10,F=30
# init_cond = [ -2.45536445,  -5.03078802,  -4.33404545,  -1.25141839,
#          2.43206433,   4.95360807,   5.33790065,   3.71648245,
#          0.9822884 ,  -1.77954178,  -3.45122017,  -3.07917425,
#         -0.50961663,   3.27759871,   6.79970615,   8.98705982,
#          9.40849095,   8.08113685,   5.51032987,   2.65664567,
#          0.48744448,  -0.44842964,  -0.04918513,   1.43830612,
#          3.50973783,   5.65259791,   7.53430131,   8.92961109,
#          9.67702253,   9.76191952,   9.33660169,   8.70309577,
#          8.33483282,   8.74993559,  10.06368927,  11.60288508,
#         12.21735306,  11.17156961,   8.66438707,   5.49594243,
#          2.41030749,  -0.30597065,  -2.87634664,  -5.77636041,
#         -9.1859454 , -12.5463623 , -14.75784347, -14.96657564,
#        -13.2788412 , -10.68920001,  -8.18008602,  -5.94976885,
#         -3.77079265,  -1.87746859,  -0.93525666,  -1.26812961,
#         -2.62968625,  -4.68645208,  -7.29769699, -10.1717435 ,
#        -12.42344327, -12.67389595,  -9.804031  ,  -3.84983827,
#          3.82490861,  11.34801522,  17.27601908,  20.94115385,
#         22.41158998,  22.19391499,  20.71898947,  18.10297359,
#         14.52292676,  10.79413418,   8.33074553,   8.33165269,
#         11.00298196,  15.35365375,  19.46366238,  21.32167332,
#         20.02854762,  16.27486995,  11.45039454,   6.32222582,
#          0.79940471,  -5.03929832, -10.10320467, -12.79609034,
#        -12.18567515,  -8.42529763,  -2.40277833,   4.61851371,
#         11.41960358,  17.25161363,  21.5945879 ,  23.56012798,
#         22.1117202 ,  17.17309048,  10.13364291,   3.0106164 ]

# init cond after 100 time for N=240,K=24,F=30
# init_cond = [ 1.35563085e+01,  1.31116444e+01,  1.22419789e+01,  1.10029118e+01,
#               9.46602660e+00,  7.71444609e+00,  5.84128086e+00,  3.95188204e+00,
#               2.16658191e+00,  6.18002733e-01, -5.61033048e-01, -1.25712952e+00,
#              -1.39647046e+00, -9.58160735e-01,  2.24101742e-02,  1.45843364e+00,
#               3.22077101e+00,  5.14866202e+00,  7.06307096e+00,  8.78606257e+00,
#               1.01669905e+01,  1.11090004e+01,  1.15840830e+01,  1.16286980e+01,
#               1.13228933e+01,  1.07640756e+01,  1.00459108e+01,  9.24636252e+00,
#               8.42332395e+00,  7.61467382e+00,  6.84057333e+00,  6.10701616e+00,
#               5.40993509e+00,  4.73886845e+00,  4.07916987e+00,  3.41257012e+00,
#               2.71722776e+00,  1.96916513e+00,  1.14648666e+00,  2.36430899e-01,
#              -7.56038690e-01, -1.80048091e+00, -2.83735346e+00, -3.78199609e+00,
#              -4.53794199e+00, -5.01736733e+00, -5.16244883e+00, -4.96012512e+00,
#              -4.44524949e+00, -3.69167646e+00, -2.79481140e+00, -1.85101935e+00,
#              -9.39100200e-01, -1.07744103e-01,  6.28679637e-01,  1.28536214e+00,
#               1.89494544e+00,  2.49242598e+00,  3.10321549e+00,  3.73921436e+00,
#               4.40313453e+00,  5.09633758e+00,  5.82385568e+00,  6.59281974e+00,
#               7.40480138e+00,  8.24574521e+00,  9.07808092e+00,  9.83866810e+00,
#               1.04443058e+01,  1.08042608e+01,  1.08370251e+01,  1.04868545e+01,
#               9.73542553e+00,  8.60578599e+00,  7.15909149e+00,  5.48764405e+00,
#               3.70854192e+00,  1.96008592e+00,  3.99031927e-01, -8.06590664e-01,
#              -1.49420836e+00, -1.53332460e+00, -8.55379900e-01,  5.24139819e-01,
#               2.50052691e+00,  4.89393929e+00,  7.47780754e+00,  1.00129877e+01,
#               1.22799174e+01,  1.41042291e+01,  1.53739471e+01,  1.60471358e+01,
#               1.61487048e+01,  1.57561698e+01,  1.49771840e+01,  1.39248489e+01,
#               1.26977785e+01,  1.13696934e+01,  9.98901726e+00,  8.58470978e+00,
#               7.17268562e+00,  5.75886848e+00,  4.33916608e+00,  2.90018866e+00,
#               1.42425977e+00, -1.01728666e-01, -1.67812298e+00, -3.28971716e+00,
#              -4.90916719e+00, -6.50267189e+00, -8.03235127e+00, -9.45257778e+00,
#              -1.07029728e+01, -1.17052757e+01, -1.23713733e+01, -1.26242048e+01,
#              -1.24247944e+01, -1.17932122e+01, -1.08130917e+01, -9.61690885e+00,
#              -8.35701289e+00, -7.17137369e+00, -6.15323869e+00, -5.33244063e+00,
#              -4.67412776e+00, -4.09738067e+00, -3.51003399e+00, -2.84788634e+00,
#              -2.10133344e+00, -1.31606078e+00, -5.66762717e-01,  8.36198243e-02,
#               6.20495584e-01,  1.09492992e+00,  1.61563241e+00,  2.31892232e+00,
#               3.32666258e+00,  4.70540330e+00,  6.43921487e+00,  8.42362126e+00,
#               1.04804104e+01,  1.23868038e+01,  1.39103894e+01,  1.48426022e+01,
#               1.50258251e+01,  1.43714291e+01,  1.28689103e+01,  1.05890427e+01,
#               7.68393661e+00,  4.38263850e+00,  9.75905190e-01, -2.21540247e+00,
#              -4.88508986e+00, -6.78634251e+00, -7.76122995e+00, -7.74765112e+00,
#              -6.76701490e+00, -4.90230206e+00, -2.27664515e+00,  9.60703973e-01,
#               4.63919353e+00,  8.56634238e+00,  1.25295764e+01,  1.63082392e+01,
#               1.96975985e+01,  2.25381724e+01,  2.47389224e+01,  2.62856634e+01,
#               2.72336396e+01,  2.76890724e+01,  2.77852348e+01,  2.76566019e+01,
#               2.74140617e+01,  2.71261567e+01,  2.68130529e+01,  2.64573994e+01,
#               2.60283837e+01,  2.55066485e+01,  2.48960796e+01,  2.42168475e+01,
#               2.34863387e+01,  2.27013892e+01,  2.18331510e+01,  2.08383475e+01,
#               1.96821686e+01,  1.83615663e+01,  1.69159712e+01,  1.54179552e+01,
#               1.39476656e+01,  1.25646728e+01,  1.12914444e+01,  1.01138252e+01,
#               8.99369157e+00,  7.88505839e+00,  6.74784265e+00,  5.55830891e+00,
#               4.31747358e+00,  3.05737595e+00,  1.84204551e+00,  7.58604062e-01,
#              -1.02814444e-01, -6.74442043e-01, -9.30385640e-01, -8.91794235e-01,
#              -6.18353015e-01, -1.94025093e-01,  2.85342911e-01,  7.20521368e-01,
#               1.01640843e+00,  1.09210653e+00,  8.93595183e-01,  4.04209800e-01,
#              -3.53514623e-01, -1.32684665e+00, -2.44798029e+00, -3.64770274e+00,
#              -4.86347819e+00, -6.04262506e+00, -7.14251050e+00, -8.12810263e+00,
#              -8.96664880e+00, -9.62144084e+00, -1.00493073e+01, -1.02054489e+01,
#              -1.00534594e+01, -9.57316745e+00, -8.76075921e+00, -7.62311148e+00,
#              -6.17308343e+00, -4.43035673e+00, -2.42716109e+00, -2.15214427e-01,
#               2.12973334e+00,  4.50835255e+00,  6.80483907e+00,  8.90037685e+00,
#               1.06890586e+01,  1.20897242e+01,  1.30494142e+01,  1.35403615e+01,]
#


# ---------------- Evaluated model integration ---------------------------

# Create an object

#init_cond = [np.random.uniform() * pars['F'] for i in range(pars['N'])]


if False:

    for K in [1]:

        pars = {'N': 4,
                'K': K,
                'F': 30
                }

        dt = 0.5


        L = models.L05_2(N=pars['N'],
                         K=K,
                         F=pars['F'])

        L.verbose = True


        #init_cond = [np.random.uniform() * pars['F'] for i in range(pars['N'])]
        #init_cond = L.evaluate(init_cond=init_cond, dt=0.1, t_stop=100, save=False)[1] # Evolve to get to attractor
        #L.evaluate(init_cond=init_cond, dt=dt, t_stop=1e3, save=True)

        #eval_path = L.evaluation_paths[-1]
        eval_path = "data/eval_{'N': 4, 'K': 1, 'F': 30, 'dt': 0.5, 't1': 1000.0}_2a5b7a155f03d5638010039a15c6c85abd7cb418_1658665161.6465442.pkl"

        # with open(eval_path, 'rb') as L05_2_inp_eval:
        #     ev = pickle.load(L05_2_inp_eval)
        #     del L05_2_inp_eval
        # plt.plot(ev.t, ev.y.T[:,0])
        # plt.plot(ev.t, ev.y.T[:,1])
        # plt.plot(ev.t, ev.y.T[:,2])
        # plt.plot(ev.t, ev.y.T[:,3])
        # plt.show()


        # L.calculate_mutual_information(eval_path, time_delays=np.arange(0, 2, 0.05), partition_num=100)
        # plt.plot(L.mutual_information[-1]['time_delays'], L.mutual_information[-1]['mutual_info'])
        # plt.grid(True)
        # plt.show()

        L.calculate_FNN_v1(eval_path, T=11, Dt=dt, norm='l2', R_tol=10, A_tol=1, indices_apart=0)

        L.print_parameters()


        # Save object
        with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:
            pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)

            del L05_2_outp
            del L



if False:

    L = models.L05_2(N=pars['N'],
                     K=pars['K'],
                     F=pars['F'])
    #L.get_terms()

    L.print_parameters()

    #L.evaluation_paths.append('data/eval_[20, 2, 30, 0.001, 10000]_d70a3db09d46e5f3472bf12b4303610c042d5df4.pkl')

    L.verbose = True

    # Save object
    with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:

        pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)

        del L05_2_outp
        del L

# Load object, perform integration
if False:

    with open('data/L05_2_{}.pkl'.format(pars), 'rb') as L05_2_inp:

        L = pickle.load(L05_2_inp)
        del L05_2_inp

        L.verbose = True

        L.stringify_ODE()
        L.print_parameters()

        #init_cond = L.evaluate(init_cond=init_cond, dt=0.1, t_stop=900, save=False)[1] # Evolve to get init_cond on attractor

        # eval_path='data/eval_[240, 24, 30, 0.001, 100]_f000e6eedc55ad89c24eee37944fc2e10a66fc25.pkl'
        # with open(eval_path, 'rb') as L05_2_inp_eval:
        #     ev = pickle.load(L05_2_inp_eval)
        #     init_cond = ev.y.T[-1]
        #     del L05_2_inp_eval, ev

        # print(L.evaluate(init_cond=init_cond, dt=dt, t_stop=t_stop))
        #
        # L.print_parameters()
        #
        # with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:
        #
        #     pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)
        #     del L05_2_outp
        #     del L

# Load object, calculate max lyapunov exponent, save object
if False:

    eval_path_num = -1

    with open('data/L05_2_{}.pkl'.format(pars), 'rb') as L05_2_inp:

        L = pickle.load(L05_2_inp)  # Lorenz_model_3_evaluated
        del L05_2_inp

        #L.print_parameters()

        eval_path = L.evaluation_paths[eval_path_num]  # Get (eval_path_num)'th eval_path

        L.calculate_lyapunov_max_v1(eval_path)

        with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:

            pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)
            del L05_2_outp
            del L

# Load object, calculate lyapunov spectrum, save object
if False:

    with open('data/L05_2_{}.pkl'.format(pars), 'rb') as L05_2_inp:

        L = pickle.load(L05_2_inp)  # Lorenz_model_2_evaluated
        del L05_2_inp

        #L.print_parameters()
        L.verbose = True

        spec = L.calculate_lyapunov_spectrum(init_cond=init_cond, T=4, max_iter_num=10, N=30)

        print(spec)

        with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:

            pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)
            del L05_2_outp
            del L

# Load object, calculate psd, save object
if False:

    eval_path_num = -2

    with open('data/L05_2_{}.pkl'.format(pars), 'rb') as L05_2_inp:
        L = pickle.load(L05_2_inp)  # Lorenz_model_3_evaluated
        del L05_2_inp

        L.print_parameters()

        #eval_path = L.evaluation_paths[eval_path_num]
        eval_path = 'data/eval_[20, 2, 30, 0.001, 10000]_d70a3db09d46e5f3472bf12b4303610c042d5df4.pkl'
        #eval_path = "data/eval_{'N': 20, 'K': 2, 'F': 30, 'dt': 0.001, 't1': 1000.0}_b57effa05d091343c13dc4f79b36aa6c9dd28dc7_1658215342.635584.pkl"


        # window = 'hanning'
        # nperseg = 1e6
        # L.calculate_PSD(eval_path, nperseg=nperseg, window=window)

        # window = 'boxcar'
        # nperseg = 1e7
        # L.calculate_PSD(eval_path, nperseg=nperseg, window=window)
        #
        # window = 'triang'
        # nperseg = 1e5
        # L.calculate_PSD(eval_path, nperseg=nperseg, window=window)
        #
        # window = 'hanning'
        # nperseg = 1e5
        # L.calculate_PSD(eval_path, nperseg=nperseg, window=window)
        #
        # window = 'boxcar'
        # nperseg = 1e5
        # L.calculate_PSD(eval_path, nperseg=nperseg, window=window)
        #
        # window = 'triang'
        # nperseg = 1e4
        # L.calculate_PSD(eval_path, nperseg=nperseg, window=window)
        #
        # window = 'hanning'
        # nperseg = 1e4
        # L.calculate_PSD(eval_path, nperseg=nperseg, window=window)
        #
        # window = 'boxcar'
        # nperseg = 1e4
        # L.calculate_PSD(eval_path, nperseg=nperseg, window=window)

        window = 'hanning'
        nperseg = 1e4

        eval_path = 'data/eval_[20, 2, 30, 0.001, 10000]_d70a3db09d46e5f3472bf12b4303610c042d5df4.pkl'
        L.calculate_PSD(eval_path, nperseg=nperseg, window=window)

        eval_path = "data/eval_{'N': 20, 'K': 2, 'F': 30, 'dt': 0.01, 't1': 1000.0}_b57effa05d091343c13dc4f79b36aa6c9dd28dc7_1658220073.2848322.pkl"
        L.calculate_PSD(eval_path, nperseg=nperseg, window=window)

        eval_path = "data/eval_{'N': 20, 'K': 2, 'F': 30, 'dt': 0.001, 't1': 1000.0}_b57effa05d091343c13dc4f79b36aa6c9dd28dc7_1658215342.635584.pkl"
        L.calculate_PSD(eval_path, nperseg=nperseg, window=window)

        eval_path = "data/eval_{'N': 20, 'K': 2, 'F': 30, 'dt': 0.0001, 't1': 1000.0}_b57effa05d091343c13dc4f79b36aa6c9dd28dc7_1658219044.2730567.pkl"
        L.calculate_PSD(eval_path, nperseg=nperseg, window=window)

        for i in range(len(L.PSD)):
            PSD_res = L.PSD[i]

            freqs = PSD_res['freqs']
            psd = PSD_res['psd']

            label = 'win={}, nperseg={}, dt={}, '.format(PSD_res['window'],int(PSD_res['npesreg']), PSD_res['dt']) + r'$t_{stop}$=' + str(int(round(PSD_res['t_stop'],0)))

            if PSD_res['window'] == 'boxcar' and PSD_res['npesreg'] == 1e7:
                plt.loglog(freqs, psd, label=label, zorder=-1000,alpha=0.5)
            else:
                plt.loglog(freqs, psd,label=label)

        plt.title('PSD: N={}, K={}, F={}'.format(pars['N'], pars['K'], pars['F']))
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()


        with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:

            pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)
            del L05_2_outp
            del L

# Load object, calculate periodic moments, save object
if False:

    eval_path_num = -1

    with open('data/L05_2_{}.pkl'.format(pars), 'rb') as L05_2_inp:
        L = pickle.load(L05_2_inp)  # Lorenz_model_3_evaluated
        del L05_2_inp

        L.verbose =True

        #eval_path = L.evaluation_paths[eval_path_num]
        eval_path = "data/eval_[20, 2, 30, 0.001, 10000]_d70a3db09d46e5f3472bf12b4303610c042d5df4.pkl"
        #eval_path = "data/eval_[20, 2, 30, 0.001, 100]_4299de3d3cb0f6d35708b262c6b159cebd7dc394.pkl"

        L.calculate_periodic_moments(eval_path, k_arr=10**np.linspace(-2,0,100))

        with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:

            pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)
            del L05_2_outp
            del L

# Load object, calculate mutual information, save object
if False:

    eval_path_num = -1

    with open('data/L05_2_{}.pkl'.format(pars), 'rb') as L05_2_inp:
        L = pickle.load(L05_2_inp)  # Lorenz_model_3_evaluated
        del L05_2_inp

        eval_path  = 'data/eval_[20, 2, 30, 0.001, 10000]_d70a3db09d46e5f3472bf12b4303610c042d5df4.pkl'

        L.verbose =True
        print(L.calculate_mutual_information(eval_path, time_delays=np.arange(0, 2, 0.05), partition_num=100))

        with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:

            pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)
            del L05_2_outp
            del L

# Load object, calculate FNN, save object
if False:

    eval_path_num = -1

    with open('data/L05_2_{}.pkl'.format(pars), 'rb') as L05_2_inp:
        L = pickle.load(L05_2_inp)  # Lorenz_model_3_evaluated
        del L05_2_inp

        L.verbose =True

        eval_path = 'data/eval_[20, 2, 30, 0.001, 10000]_d70a3db09d46e5f3472bf12b4303610c042d5df4.pkl'
        #eval_path = L.evaluation_paths[eval_path_num]

        L.calculate_FNN_v1(eval_path, T=1, Dt=0.1, norm='l2', R_tol=10, A_tol=1)#, td_dim_span=(30, 32))
        #L.calculate_FNN_v3(eval_path, T=10, Dt=5, norm='l2')

        with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:

            pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)
            del L05_2_outp
            del L

# Load object, calculate BCD, save object
if False:

    eval_path_num = -1

    with open('data/L05_2_{}.pkl'.format(pars), 'rb') as L05_2_inp:
        L = pickle.load(L05_2_inp)  # Lorenz_model_3_evaluated
        del L05_2_inp

        L.verbose =True

        # eval_paths = [
        #     'data/0/eval_[20, 3, 15, 1e-06, 10]_5cf6c561df4c432f8d68d661bfbafcdb0116194a.pkl',
        #     'data/0/eval_[20, 3, 15, 1e-06, 10]_e45e985d9300a5fe37bcb4421a50d006f27201bd.pkl',
        #     'data/0/eval_[20, 3, 15, 1e-06, 10]_2c25752d2c0cee65c9fa02629658230280b4d184.pkl',
        #     'data/0/eval_[20, 3, 15, 1e-06, 10]_013cde0b0c4e8de59d1401763a24261a8182ac1d.pkl',
        #     'data/0/eval_[20, 3, 15, 1e-06, 10]_fe5fd1c3ae3be10bef2c0874f0c6f08be664432e.pkl',
        #     'data/0/eval_[20, 3, 15, 1e-06, 10]_8e99383340e3c0c875269114a3131a750841328c.pkl',
        #     'data/0/eval_[20, 3, 15, 1e-06, 10]_edabcd4795b5b0cab9ee798d870cc3a7133d3088.pkl',
        #     'data/0/eval_[20, 3, 15, 1e-06, 10]_1e5f7ec89cdd5d2c48ed28e54ea9ff18c7bdb590.pkl',
        #     'data/0/eval_[20, 3, 15, 1e-06, 10]_5f442533fa049717aa8d0d205c5505ea36e44589.pkl',
        #     'data/0/eval_[20, 3, 15, 1e-06, 10]_d908b0343e6b95879feb877610f3d0d5b1f8d06b.pkl',
        # ]

        eval_path = 'data/eval_[20, 3, 15, 0.01, 100000]_d908b0343e6b95879feb877610f3d0d5b1f8d06b.pkl'

        L.calculate_BCD(eval_paths, eps_arr=[0.0008,0.0006,0.0004,0.0002])#, td_dim_span=(30, 32))

        with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:

            pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)
            del L05_2_outp
            del L

# Load object, calculate inf. dim., save object
if False:

    eval_path_num = -1

    with open('data/L05_2_{}.pkl'.format(pars), 'rb') as L05_2_inp:
        L = pickle.load(L05_2_inp)  # Lorenz_model_3_evaluated
        del L05_2_inp

        L.verbose =True

        eval_path = 'data/eval_[20, 2, 50, 0.001, 10000]_fa81fa48cd3db4f4e90c136ca835c15672d3a7e3.pkl'

        L.calculate_information_dim(eval_path)

        with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:

            pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)
            del L05_2_outp
            del L

# Load object, print some results
if False:


    eval_path_num = -1
    lyap_path_num = -1

    with open("data/L05_2_{'N': 20, 'K': 2, 'F': 30}/L05_2_{'N': 20, 'K': 2, 'F': 30}.pkl", 'rb') as L05_2_inp:
        L = pickle.load(L05_2_inp)  # Lorenz_model_3_evaluated
        del L05_2_inp

        L.get_terms()

        L.print_parameters()



        eval_path = L.evaluation_paths[eval_path_num]
        print(eval_path)

        eval_path = L.evaluation_paths[eval_path_num]  # Get (eval_path_num)'th eval_path
        #lyap_path = L.lyapunov_max_paths[lyap_path_num]  # Get (lyap_path_num)'th lyap_path

        with open(eval_path, 'rb') as L05_2_inp_eval:
            ev = pickle.load(L05_2_inp_eval)
            print(ev)


            print(ev)

            del L05_2_inp_eval
        # #
        # with open(lyap_path, 'rb') as L05_2_inp_lyap:
        #     ly = pickle.load(L05_2_inp_lyap)
        #     print(ly)
        #
        #     del L05_2_inp_lyap

# loop N or F
if False:

    Ns = range(5, 31, 1)
    lyaps = []
    lyaps_std = []
    sample_sizes =[]
    stopping_time = []
    stopping_time_std = []
    # av = []
    # ma = []
    # mi = []
    # sd = []
    # true_av = []
    # true_sd = []


    #for F in Fs:
    for N in Ns:

        ###### Generate data ######
        pars = {'N':N,  # N
                'K':3   ,  # K
                'F':30,  # F
        }

        print('##############################################')
        print('\nN = {}, F = {}, K = {}\n'.format(pars['N'],pars['K'],pars['F']))

        dt = 1e-4
        t_stop = 200

        gc.collect()

        ##### Generate data ######
        init_cond = np.asarray([random.random() * pars['F'] for i in range(pars['N'])])

        L = models.L05_2(N=pars['N'],
                         K=pars['K'],
                         F=pars['F'])
        #L.get_terms()
        L.verbose = True
        init_cond = L.evaluate(init_cond=init_cond, dt=1, t_stop=100, save=False)[1] # evolve for 100 time units first to reach the attractor
        eval_path = L.evaluate(init_cond=init_cond, dt=dt, t_stop=t_stop, save=True)[0]

        L.verbose = False

        # r = L.calculate_eval_props(eval_path)
        #
        # av.append(r['av'])
        # ma.append(r['ma'])
        # mi.append(r['mi'])
        # sd.append(r['sd'])
        # #true_av.append(r['true_av'])
        # true_sd.append(r['true_sd'])

        ly = L.calculate_lyapunov_max(eval_path, save=False)

        # print(ly['average lyapunov'])
        # print(ly['std lyapunov'])
        # print(ly['sample size'])
        # print(ly['average stopping time'])
        # print(ly['std stopping time'])

        lyaps.append(ly['average lyapunov'])
        lyaps_std.append(ly['std lyapunov'])
        sample_sizes.append(ly['sample size'])
        stopping_time.append(ly['average stopping time'])
        stopping_time_std.append(ly['std stopping time'])

        # print(ly['sample size'])


        #L.print_parameters()

        # Save data
        # with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:
        #
        #     pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)
        #     del L05_2_outp
        #     del L
        #
        # ##### Load data #######
        # with open('data/L05_2_{}.pkl'.format(pars), 'rb') as L05_2_inp:
        #     L = pickle.load(L05_2_inp)  # Lorenz_model_3_evaluated
        #     del L05_2_inp
        #
        #     #L.print_parameters()
        #
        #     eval_path = L.evaluation_paths[-1]
        #     lyap_max_path = L.lyapunov_max_paths[-1]
        #
        #     with open(lyap_max_path, 'rb') as L05_2_inp_lyap:
        #         ly = pickle.load(L05_2_inp_lyap)
        #
        #         lyaps.append(ly['average lyapunov'])
        #         lyaps_std.append(ly['std lyapunov'])
        #         sample_sizes.append(ly['sample size'])
        #         stopping_time.append(ly['average stopping time'])
        #         stopping_time_std.append(ly['std stopping time'])
        #
        #         del L05_2_inp_lyap, ly

    print('Ns=',[i for i in Ns])
    print('lyaps=',lyaps)
    print('lyaps_std=',lyaps_std)
    print('sample_sizes=',sample_sizes)
    print('stopping_time=',stopping_time)
    print('stopping_time_std=',stopping_time_std)

    # print('Ns = ', [i for i in Ns])
    # print('av = ', av)
    # print('ma = ', ma)
    # print('mi = ', mi)
    # print('sd = ', sd)
    # #print(true_av)
    # print('true_sd = ', true_sd)


# loop N/K
if True:
    Ns = np.asarray([20,40,60,80,120,160,200,240])
    Ks = np.asarray(Ns / 5).astype(int)
    lyaps = [[] for i in range(len(Ns))]
    sample_sizes = np.zeros(len(Ns))
    stopping_times = [[] for i in range(len(Ns))]

    av = []
    ma = []
    mi = []
    sd = []
    true_av = []
    true_sd = []


    for i in range(len(Ns)):
        print('N=',Ns[i])
        print('K=',Ks[i])


        ###### Generate data ######
        pars = {'N':Ns[i],  # N
                'K':Ks[i],  # K
                'F':15,  # F
        }
        dt = 1e-4
        t_stop = 14

        gc.collect()

        ##### Generate data ######
        init_cond = np.asarray([random.random() * pars['F'] for i in range(pars['N'])])

        L = models.L05_2(N=pars['N'],
                        K=pars['K'],
                        F=pars['F'])

        L.verbose = True
        #L.get_terms()
        init_cond = L.evaluate(init_cond=init_cond, dt=1, t_stop=100, save=False)[1] # evolve for 10 time units first to reach the attractor
        eval_path = L.evaluate(init_cond=init_cond, dt=0.1, t_stop=100, save=True)[0]

        r = L.calculate_eval_props(eval_path)

        av.append(r['av'])
        ma.append(r['ma'])
        mi.append(r['mi'])
        sd.append(r['sd'])
        #true_av.append(r['true_av'])
        true_sd.append(r['true_sd'])

        # for j in range(10):
        #     eval_path, init_cond = L.evaluate(init_cond=init_cond, dt=dt, t_stop=t_stop, save=True)
        #
        #     ly = L.calculate_lyapunov_max_v1(eval_path, save=False)
        #
        #     os.remove(eval_path)
        #
        #     print(ly['all generated lyapunov'])
        #
        #     lyaps[i] = np.concatenate((lyaps[i],ly['all generated lyapunov']))
        #     sample_sizes[i] += ly['sample size']
        #     stopping_times[i] = np.concatenate((stopping_times[i],ly['all generated stopping time']))


        #L.print_parameters()

        # Save data
        # with open('data/L05_2_{}.pkl'.format(pars), 'wb') as L05_2_outp:
        #
        #     pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)
        #     del L05_2_outp
        #     del L
        #
        # ##### Load data #######
        # with open('data/L05_2_{}.pkl'.format(pars), 'rb') as L05_2_inp:
        #     L = pickle.load(L05_2_inp)  # Lorenz_model_3_evaluated
        #     del L05_2_inp
        #
        #     #L.print_parameters()
        #
        #     eval_path = L.evaluation_paths[-1]
        #     lyap_max_path = L.lyapunov_max_paths[-1]
        #
        #     with open(lyap_max_path, 'rb') as L05_2_inp_lyap:
        #         ly = pickle.load(L05_2_inp_lyap)
        #
        #         lyaps.append(ly['average lyapunov'])
        #         lyaps_std.append(ly['std lyapunov'])
        #         sample_sizes.append(ly['sample size'])
        #         stopping_time.append(ly['average stopping time'])
        #         stopping_time_std.append(ly['std stopping time'])
        #
        #         del L05_2_inp_lyap, ly

    print('Ns=',[i for i in Ns])
    print('Ks=', [i for i in Ks])

    # print('lyaps=',[np.average(i) for i in lyaps])
    # print('lyaps_std=',[np.std(i) for i in lyaps])
    # print('sample_sizes=',sample_sizes)
    # print('stopping_times=', [np.average(i) for i in stopping_times])
    # print('stopping_times_std=', [np.std(i) for i in stopping_times])

    # print('av = ', av)
    # print('ma = ', ma)
    # print('mi = ', mi)
    # print('sd = ', sd)
    #print(true_av)
    print('true_sd = ', true_sd)

##############################################################

if False:

    # offdiag=0
    filenames = ["L05_2_SINDy_STLSQ_search_{'N': 20, 'K': 2, 'F': 30}_predefined_split_{'mu': 0, 'dt': 0.01, 'test_size': 300, 'all_point': 4900, 'max_it': 20, 'df': 'SFD'}_5120447993576d598cbd96ecf00f8b3e226a1b3f",
                 "L05_2_SINDy_ConsSR3_search_{'N': 20, 'K': 2, 'F': 30}_predefined_split_{'mu': 0, 'dt': 0.01, 'test_size': 300, 'all_point': 4900, 'max_it': 20, 'df': 'SFD'}_acd3e95f81a0c0e6f2ecc20d34d689d6c794581b",
                 "L05_2_SINDy_TrapSR3_search_{'N': 20, 'K': 2, 'F': 30}_predefined_split_eta=1_{'mu': 0, 'dt': 0.01, 'test_size': 300, 'all_point': 4900, 'max_it': 5, 'df': 'SFD'}_acd3e95f81a0c0e6f2ecc20d34d689d6c794581b",
#
                 "L05_2_SINDy_STLSQ_search_{'N': 20, 'K': 2, 'F': 30}_predefined_split_{'mu': 0.5, 'dt': 0.01, 'test_size': 300, 'all_point': 4900, 'max_it': 20, 'df': 'SFD'}_4144a6adf539e455c9bc20076ad3e79e276b4da2",
                 "L05_2_SINDy_ConsSR3_search_{'N': 20, 'K': 2, 'F': 30}_predefined_split_{'mu': 0.5, 'dt': 0.01, 'test_size': 300, 'all_point': 4900, 'max_it': 20, 'df': 'SFD'}_4144a6adf539e455c9bc20076ad3e79e276b4da2",
                 "L05_2_SINDy_TrapSR3_search_{'N': 20, 'K': 2, 'F': 30}_predefined_split_eta=1_{'mu': 0.5, 'dt': 0.01, 'test_size': 300, 'all_point': 4900, 'max_it': 20, 'df': 'SFD'}_4144a6adf539e455c9bc20076ad3e79e276b4da2",

                 "L05_2_SINDy_STLSQ_search_{'N': 20, 'K': 2, 'F': 30}_predefined_split_{'mu': 1, 'dt': 0.01, 'test_size': 300, 'all_point': 4900, 'max_it': 20, 'df': 'SFD'}_96c15d7ae8248de632be7ff2fdbc17721667edaf",
                 "L05_2_SINDy_ConsSR3_search_{'N': 20, 'K': 2, 'F': 30}_predefined_split_{'mu': 1, 'dt': 0.01, 'test_size': 300, 'all_point': 4900, 'max_it': 20, 'df': 'SFD'}_8dac0ae2a2f7aff9c0def393328b6408cff92151",
                 # TrapSR3 mu=1 diverges
                 ]

    # offdiag=0.5
    filenames = [
                #"L05_2_SINDy_STLSQ_search_{'N': 20, 'K': 2, 'F': 30}_predefined_split_{'mu': 0.5, 'dt': 0.01, 'test_size': 300, 'all_point': 4500, 'max_it': 20, 'df': 'SFD'}_cc28334e92fbc2020d6adc73a4d174cd5c0df927",
                # STLSQ mu=1 diverges

                #"L05_2_SINDy_ConsSR3_search_{'N': 20, 'K': 2, 'F': 30}_predefined_split_{'mu': 0.5, 'dt': 0.01, 'test_size': 300, 'all_point': 4500, 'max_it': 20, 'df': 'SFD'}_cc28334e92fbc2020d6adc73a4d174cd5c0df927",
                #"L05_2_SINDy_ConsSR3_search_{'N': 20, 'K': 2, 'F': 30}_predefined_split_{'mu': 1, 'dt': 0.01, 'test_size': 300, 'all_point': 4500, 'max_it': 20, 'df': 'SFD'}_1cb25185082cafeef884ef87bb8424c0ab133e15",

                # TrapSR3 mu=0.5 diverges
                # TrapSR3 mu=0.5 diverges
                ]

    c=0
    for filename in filenames:
        c=c+1

        print("\n--------------------------------------------------------------------------------------------------------")
        print(filename)
        print("--------------------------------------------------------------------------------------------------------\n")

        model_path = "data/L05_2_{'N': 20, 'K': 2, 'F': 30}/Trained_models_cov_off_diag_0/" + filename + ".pkl"
        os.mkdir("data/L05_2_{'N': 20, 'K': 2, 'F': 30}/Trained_models_cov_off_diag_0/sur/sur_" + filename)
        model_name = "data/L05_2_{'N': 20, 'K': 2, 'F': 30}/Trained_models_cov_off_diag_0/sur/sur_" + filename + '/sur_' + filename + ".pkl"

        # model_path = "data/L05_2_{'N': 20, 'K': 2, 'F': 30}/Trained_models_cov_off_diag_0.5/" + filename + ".pkl"
        # os.mkdir("data/L05_2_{'N': 20, 'K': 2, 'F': 30}/Trained_models_cov_off_diag_0.5/sur/sur_" + filename)
        # model_name = "data/L05_2_{'N': 20, 'K': 2, 'F': 30}/Trained_models_cov_off_diag_0.5/sur/sur_" + filename + '/sur_' + filename + ".pkl"




        L = models.sur_L05_2(N=pars['N'],
                        K=pars['K'],
                        F=pars['F'],
                        model_path=model_path
                            )

        L.verbose = True

        print("Evolve before start:")
        init_cond = L.evaluate(init_cond=init_cond, dt=0.1, t_stop=10, save=False)[1] # Evolve to get init_cond on attractor
        print('\nEvolve:')
        L.evaluate(init_cond=init_cond, dt=dt, t_stop=t_stop)
        print('\n')

        L.calculate_PSD(L.evaluation_paths[-1])
        L.calculate_periodic_moments(L.evaluation_paths[-1], k_arr=10 ** np.linspace(-2, 0, 100))

        spec = L.calculate_lyapunov_spectrum(init_cond=init_cond, T=4, max_iter_num=10, N=20)
        print(spec)

        with open(model_name, 'wb') as L05_2_outp:

            pickle.dump(L, L05_2_outp, pickle.HIGHEST_PROTOCOL)
            del L05_2_outp
            del L


if False:

    model_path = "data/L05_2_{'N': 20, 'K': 2, 'F': 30}/Trained_models_cov_off_diag_0/L05_2_SINDy_ConsSR3_search_{'N': 20, 'K': 2, 'F': 30}_predefined_split_{'mu': [0.5, 0, 1], 'dt': 0.01, 'test_size': 1000, 'all_point': 10000, 'max_it': 30, 'df': 'SFD'}_513a7d407de8989cdd7ffdf8c69cff497791af6e_1657970191.2881467.pkl"

    L = models.sur_L05_2(N=pars['N'],
                         K=pars['K'],
                         F=pars['F'],
                         model_path=model_path
                         )

    L.verbose = True


    L.calculate_covariance_and_accuracy(10)

if False:
    cut = 0.1

    # normalize covariance matrix with its max element
    Cov = Cov / np.max(Cov)

    # for plottin purpuses delete rows and cols that have max element < tol (<<1)
    Cov_reduced_1 = []
    Cov_reduced_2 = []
    C_reduced_rows = []
    for i, row in enumerate(Cov):
        if np.abs(np.max(row)) > cut:
            Cov_reduced_1.append(np.asarray(row))
            C_reduced_rows.append(i)
    for i, col in enumerate(np.asarray(Cov_reduced_1).T):
        if np.abs(np.max(col)) > cut:
            Cov_reduced_2.append(col)

    feat_names = []
    for i in range(self.N):
        feat_names += ['X' + str(i) + ' ' + j for j in self.model.feature_library.get_feature_names()]

    Cov_reduced = np.asarray(Cov_reduced_2)

    fig = plt.figsize = (10, 10)
    xlabs = [feat_names[i] for i in C_reduced_rows]
    ylabs = xlabs
    sn.heatmap(Cov_reduced, cmap='jet', square=True, xticklabels=xlabs, yticklabels=ylabs, center=0, vmin=-1, vmax=1)
    plt.show()

if False:

    pars = {'N': 20,
            'K': 2,
            'F': 30
            }

    dt = 1e-3
    t_stop = 1e3

    rootdirs = [
                "/home/kreljo/Documents/FAKS/Magistrska st./0Magistrska/program/data/L05_2_{'N': 20, 'K': 2, 'F': 30}/Trained_models_cov_off_diag_0",
                "/home/kreljo/Documents/FAKS/Magistrska st./0Magistrska/program/data/L05_2_{'N': 20, 'K': 2, 'F': 30}/Trained_models_cov_off_diag_0.5",
                "/home/kreljo/Documents/FAKS/Magistrska st./0Magistrska/program/data/L05_2_{'N': 20, 'K': 2, 'F': 30}/Trained_models_cov_off_diag_1",
                "/home/kreljo/Documents/FAKS/Magistrska st./0Magistrska/program/data/L05_2_{'N': 20, 'K': 2, 'F': 30}/Trained_models_cov_only_two_0.5",
                "/home/kreljo/Documents/FAKS/Magistrska st./0Magistrska/program/data/L05_2_{'N': 20, 'K': 2, 'F': 30}/Trained_models_cov_only_two_1",
                ]

    for rootdir in rootdirs:
        for filename in os.listdir(rootdir):
            model_path = os.path.join(rootdir, filename)
            try:
                if os.path.isfile(model_path):
                    print(model_path)

                    # with open(model_path, 'rb') as inp:
                    #     res = pickle.load(inp)
                    # with open(model_path, 'wb') as outp:
                    #     pickle.dump(res, outp, pickle.HIGHEST_PROTOCOL)

                    L = models.sur_L05_2(N=pars['N'],
                                         K=pars['K'],
                                         F=pars['F'],
                                         model_path=model_path
                                         )
                    L.verbose = True
                    init_cond = L.props['x'][-L.props['test_size']]  # start eval at beginning of the test set


                    L.calculate_coef_error()
                    L.evaluate(init_cond=init_cond, dt=dt, t_stop=t_stop)
                    L.calculate_PSD(L.evaluation_paths[-1])
                    L.calculate_periodic_moments(L.evaluation_paths[-1], k_arr=10 ** np.linspace(-2, 0, 100))
                    L.calculate_covariance_and_accuracy(m=100)
                    L.calculate_eval_props(L.evaluation_paths[-1])

                    # # Move results to a new directory
                    dir_name = model_path[:-4] + '_'
                    os.mkdir(dir_name)
                    shutil.move(model_path, os.path.join(dir_name, filename))

                    with open(os.path.join(dir_name, 'sur_mod_' + filename), 'wb') as outp:

                        pickle.dump(L, outp, pickle.HIGHEST_PROTOCOL)
                        del outp
                        del L
            except BaseException as e:
                print('##############################################################')
                print('##############################################################')
                print('##############################################################')
                print('##############################################################')
                print('##############################################################')


            # ###### PLOT COV and CORR mat ##########################################
            #
            # # plot parameters
            # cut = 0.1
            #
            # if filename[12:12+5] == 'STLSQ':
            #     name = 'STLSQ'
            # elif filename[12:12+7] == 'ConsSR3':
            #     name = 'ConsSR3'
            # elif filename[12:12+7] == 'TrapSR3':
            #     name = 'TrapSR3'
            #
            # # load results
            # cov_mat = L.stats[-1]['cov']
            # corr_mat = L.stats[-1]['corr']
            # C_max = np.max(cov_mat)
            # m = L.stats[-1]['m']  # num. perturbations
            # a = L.stats[-1]['a']  # pert. size
            # w_a = L.stats[-1]['w_a']  # accuracy
            # noise = L.props['noise'] # noise [ampl  diag, ampl diag/off_diag, type=1 or 2]
            #                          # where 1: all off_diag, 2: only two off_diag
            #
            # # normalize covariance matrix with its max element
            # Cov = cov_mat / C_max
            # Corr = corr_mat
            #
            # # for plottin purpuses delete rows and cols that have max element < tol (<<1)
            # Cov_reduced_1 = []
            # Cov_reduced_2 = []
            # Corr_reduced_1 = []
            # Corr_reduced_2 = []
            # C_reduced_rows = []
            # for i, row in enumerate(Cov):
            #     if np.abs(np.max(row)) > cut:
            #         Cov_reduced_1.append(np.asarray(row))
            #         Corr_reduced_1.append(np.asarray(Corr[i,:]))
            #         C_reduced_rows.append(i)
            # Corr_reduced_1 = np.asarray(Corr_reduced_1).T
            # for i, col in enumerate(np.asarray(Cov_reduced_1).T):
            #     if np.abs(np.max(col)) > cut:
            #         Cov_reduced_2.append(col)
            #         Corr_reduced_2.append(Corr_reduced_1[i])
            #
            # feat_names = []
            # for i in range(pars['N']):
            #     feat_names += ['X' + str(i) + ' ' + j for j in L.model.feature_library.get_feature_names()]
            #
            # Cov_reduced = np.asarray(Cov_reduced_2)
            # Corr_reduced = np.asarray(Corr_reduced_2)
            #
            # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08], 'height_ratios': [1]})
            # fig.suptitle('COV: ' + name + ', noise=' + str(noise) + ', num. of perturbations=' + str(m) + ',\n pert. size=' + str(a) + ', max=' +  str(round(C_max,10)) + ', cut=' + str(cut))
            # xlabs = [feat_names[i] for i in C_reduced_rows]
            # ylabs = xlabs
            # g1 = sn.heatmap(Cov_reduced, cmap='jet', square=True, xticklabels=xlabs, yticklabels=ylabs, center=0, vmin=-1,
            #                   vmax=1, ax=ax1, cbar=False, cbar_kws={"shrink": 0.5})
            # g1.set_title('Cov/max(Cov)')
            # g2 = sn.heatmap(Corr_reduced, cmap='jet', square=True, xticklabels=xlabs, yticklabels=ylabs, center=0, vmin=-1,
            #                   vmax=1, ax=ax2, cbar_ax=ax3, cbar_kws={"shrink": 0.5})
            # g2.set_title('Corr')
            #
            #
            # plt.show()








