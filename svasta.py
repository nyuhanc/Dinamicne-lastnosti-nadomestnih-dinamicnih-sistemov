from pysindy.feature_library import CustomLibrary
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.cm import rainbow
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from pysindy.utils import linear_damped_SHO
from pysindy.utils import cubic_damped_SHO
from pysindy.utils import linear_3D
from pysindy.utils import hopf
from pysindy.utils import lorenz
import pysindy as ps
import pickle

a = np.ones((3,5))
print(a)

print(np.average(a, axis=0))

from pysindy.feature_library import CustomLibrary
#


import sys
import time

import os
rootdir = "/home/kreljo/Documents/FAKS/Magistrska st./0Magistrska/program/data/L05_2_{'N': 20, 'K': 2, 'F': 30}"


print([i for i in range(239,0,-12)][::-1])


import numpy as np
import scipy.integrate as integrate

def calculate_FNN_v1(dt=0.5, t_stop=1000, T=10, Dt=0.5, R_tol=10, A_tol=1, norm='l2', indices_apart=None, td_dim_span=(1, 5)):
    # Calculates False Nearest Neighbour function. Par. T is the time lag used to construct time-delay
    # coordinates, Dt is the spacing in time when delay coordinates are constructed and td_dim is a tuple (min, max)
    # dimension of time delay coordinates vector space (where we estimate FNN). Both T as delta_t should be
    # multiples of provided dt, and aldo T should be a multiple of delta_t. A scalar function that is
    # used in the process is the first component of the evaluated trajectory. td_dim_span[0]=td_dim_min must be
    # at least 1. R_tol and A_tol are method intrinsic parameters (Determining embedding dimension for phase-space
    # reconstruction using a geometrical construction, M.b. Kennel, R. Brown, H.D.I. Abarbanel, 1992)
    # Additional parameter to the ones proposed in the article is the 'norm' of the distance between two points. In
    # the original article l2 norm is used, but one can choose here also dimension-normalized l2 ('l2/d') and the
    # supremum ('max') norm. Parameter indices_apart defines the minimal distance in array between two points where
    # if they can be checked for NN. The purpose of this parameter is that two points consecutive in time must not
    # identified as NN for algorithm to work properly.

    # Set default value of td_dim
    td_dim_min = td_dim_span[0]
    td_dim_max = td_dim_span[1]

    ################ GENERATE DATA ##############################################################

    integrator_keywords = {}
    integrator_keywords['rtol'] = 1e-12
    integrator_keywords['method'] = 'LSODA'
    integrator_keywords['atol'] = 1e-10
    integrator_keywords['min_step'] = 1e-7

    def lorenz(t, x, sigma=10, beta=2.66667, rho=28):
        return [
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ]

    t = np.arange(0, t_stop, dt)
    t_span = (t[0], t[-1])
    x0_train = [5.94857223, -0.12379429, 31.24273752]
    x_train = integrate.solve_ivp(lorenz, t_span, x0_train, t_eval=t, **integrator_keywords).y.T
    x0 = x_train[:, 0]

    ##############################################################################################


    print('\nCalculating FNNv1. Parameters:')
    print('Time delay (T): ', T)
    print('dt from evaluation: ', dt)
    print('max time from evaluation: ', t[-1])
    print('new dt:=Dt: ', Dt)
    print('R_tol: ', R_tol)
    print('A_tol: ', A_tol)
    print('norm: ', norm)

    # intigers pointing to the element equivalent to T and Dt
    Ti = int(T / dt)
    Dti = int(Dt / dt)

    # Check if provided T/dt, delta_t/dt and T/delta_t are all integers
    if (round(Ti, 10)) % 1 != 0:  # round because of numerical error
        raise ValueError('Time_delay should be'
                         ' some multiple of dt from provided evaluation')
    if (round(Dti, 10)) % 1 != 0:  # round because of numerical error
        raise ValueError('Par. delta_t should be'
                         ' some multiple of dt from provided evaluation')
    if (round(T / Dt, 10)) % 1 != 0:  # round because of numerical error
        raise ValueError('Par. T should be'
                         ' some multiple of delta_t from provided evaluation')
    if (round(len(x0) / Dti, 10)) % 1 != 0:  # round because of numerical error
        raise ValueError('Number of points from provided evaluation should'
                         ' be dividable by Dt/dt')
    print('Ti: ', Ti)
    print('Dti: ', Dti)

    # Estimate the variance of the signal x
    var = np.var(x0)

    # generate time delay vector function coordinates of dimension td_dim
    l = int((len(x0) - Ti * (td_dim_max - 1)) / Dti)
    print('num. of datapoints: ', l)

    # The points between where distance is calculated must be at least 1/100 of the sample apart in time so
    # that the NN are not NN in time
    if indices_apart == None:
        indices_apart = int(l / 100)
    print('indices apart: ', indices_apart)

    # Generate time-delay coorinates
    TD = np.zeros((l, td_dim_max))
    for i in range(l):
        for j in range(td_dim_max):
            TD[i, j] = x0[i * Dti + j * Ti]

    c = 0
    for i in range(indices_apart, l):
        c += np.abs(TD[0, 0] - TD[i, 0])
    print('Avarage dist. from 1st point to other points : ', c / (l - indices_apart))
    print('\n')

    def get_NN(d):
        # returns the indices of NN for dim. of delayed coord. space d, the distance squared
        # between these two points, and the distance squared between points in d+1 dim.
        # (returns [i, NN_i,  dist,  dist_of_dplus1] for each index i)

        # Initialize arrays
        NN_ind_arr = np.zeros((l, 4))
        TDd = TD[:, :d]  # shape = (l,td_dim_max)
        TDdplus1 = TD[:, d]  # shape = l

        # Find NN in dim. of time delay d and save the  dist. of every pair
        for i, pi in enumerate(TDd):
            NNj = None
            c = 10 ** 10  # initial  distance to NN

            # Find NN
            for j, pj in enumerate(TDd):
                if np.abs(i - j) > indices_apart:

                    if norm == 'max':
                        c_new = np.linalg.norm(pi - pj, ord=np.inf)
                    elif norm == 'l2':
                        c_new = np.linalg.norm(pi - pj)
                    elif norm == 'l2/d':
                        c_new = np.linalg.norm(pi - pj) / d ** 0.5
                    else:
                        raise ValueError("Parameter 'norm' should be one of ('max', 'l2', 'l2/d')")

                    if c_new < c:
                        c = c_new
                        NNj = j

            # Calculate the increase in dist. of every pair for dim. of time delay vector space d+1
            cc = np.abs(TDdplus1[i] - TDdplus1[NNj])

            # Save result
            NN_ind_arr[i] = [i, NNj, c, cc]
            # print([i,NNj, c, cc])

        return NN_ind_arr

    # Initialize return arrays
    fnn = []
    d_arr = []
    threshold_choice = []  # for each d, if R_tol * NNd[i,2] was always chosen for threshold the value is 1
    # and 0 if times A_tol * var was always chosen to be the threshold (and linear inbetween)

    # Attractor size bound
    A = A_tol * var

    # Calculate fnn for every d
    for d in np.arange(td_dim_min, td_dim_max):

        print('\nCalculating for dim. {}/{}'.format(d, td_dim_max - 1))

        NNd = get_NN(d)

        score = 0
        thr = 0

        NNdplus1_avr = 0
        NNd_avr = 0
        ind_apart_avr = 0

        for i in range(l):

            R = R_tol * NNd[i, 2]

            NNdplus1_avr += NNd[i, 3]
            NNd_avr += NNd[i, 2]
            ind_apart_avr += np.abs(NNd[i, 0] - NNd[i, 1])

            if d != td_dim_min and fnn[-1] < 0.1:
                threshold = np.max([R, A])
            else:
                threshold = R

            if threshold == R:
                thr += 1

            if NNd[i, 3] > threshold:
                score += 1

        d_arr.append(d)
        fnn.append(score / l)
        threshold_choice.append(thr / l)

        NNdplus1_avr /= l
        NNd_avr /= l
        ind_apart_avr /= l
        print('Average dist. in first {} dimensions:'.format(d), NNd_avr)
        print('Average dist. in {}+1 dimension:'.format(d), NNdplus1_avr)
        print('Average num. of indices apart in {} dimension:'.format(d), ind_apart_avr)
        print('fnn(d): ', fnn[-1])
        print('threshold choice: ', threshold_choice[-1])

        # break loop if fnn(d) = 0
        if fnn[-1] == 0:
            break

    results = {'d_arr': np.asarray(d_arr),
                        'fnn': np.asarray(fnn),
                        'thr_choice': threshold_choice,
                        'dt': dt,
                        't_stop':t_stop,
                        'T': T,
                        'Dt': Dt,
                        'inidces_apart': indices_apart,
                        'td_dim_span': td_dim_span}

    print(results)


#calculate_FNN_v1()


# for filename in os.listdir(rootdir):
#     f = os.path.join(rootdir,filename)
#     if os.path.isfile(f):
#         print(f)

#
# a = 0
# for x in range (0,3):
#     a = a + 1
#     b = ("Loading" + str(a))
#     # \r prints a carriage return first, so `b` is printed on top of the previous line.
#     sys.stdout.write('\r'+b)
#     time.sleep(0.5)
#
# init_cond=[0,0,0,0,0]
# delta_start=1
# N=5
#
# Z0 = np.asarray([init_cond for i in range(N + 1)])
# for i in range(1, N + 1):
#     Z0[i,i-1] += 1
#
# print(Z0)
# print([Z0_i for Z0_i in Z0])
#
# x=np.asarray([1,1,3])/2
# y=np.asarray([[2,37,1],[0,0,3]])
# print(np.sum(y,-1))
# print([np.random.exponential(0.1) for i in range(10)])
# # import numpy as np
# # from sklearn.model_selection import TimeSeriesSplit
# # X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
# # y = np.array([1, 2, 3, 4, 5, 6])
# # tscv = TimeSeriesSplit()
# # print(tscv)
# # TimeSeriesSplit(gap=0, max_train_size=1, n_splits=5, test_size=None)
# # for train_index, test_index in tscv.split(X):
# #     print("TRAIN:", train_index, "TEST:", test_index)
#
# print('\n\n\n\n')
# #
# x = np.array([[0.,-1,2,3,5],[1.,0.,2,3,5],[2.,-1.,2,3,5]])
#
# library_functions = [lambda : 1, lambda x: x, lambda x, y: x * y, lambda x: x ** 2]
# library_function_names = [lambda : 1, lambda x: x, lambda x, y: x + y, lambda x: x + x]
# sindy_library = ps.CustomLibrary(
#     library_functions=library_functions,
#     function_names=library_function_names
# )
# sindy_library.fit(x)
# print(sindy_library.transform(x))
# print(sindy_library.get_feature_names())
# print(sindy_library.powers_)
#
#
# print('##')
#
# sindy_library = ps.PolynomialLibrary(degree=2)
# sindy_library.fit(x)
# print(sindy_library.transform(x))
# print(sindy_library.get_feature_names())
# print(sindy_library.powers_)
#
# print('\n\n\n\n')
#
# # C = [1, 'x0', 'x1', 'x2', 'x3', 'x0x1', 'x0x2', 'x0x3', 'x1x2', 'x1x3', 'x2x3', 'x0x0', 'x1x1', 'x2x2', 'x3x3']
# #
# # r=4
# # D = ['__' for i in range(int((r ** 2 + 3 * r) / 2.0)+1)]
# #
# # for i in range(1, r):
# #     for j in range(0,r-i):
# #         D[ (i*r - (i-1)) + j + i + 1] = C[ (i*r - (i-1)) + j + 1]
# #
# # for i in range(1,r+1):
# #     D[-sum(range(i+1))] = C[-i]
# #
# #
# #
# #
# # print(D)
# #
# #
# # print('\n\n\n\n')
# #
# #
# # C = ['1', 'x0', 'x1', 'x2', 'x3', 'x0^2', 'x0 x1', 'x0 x2', 'x0 x3', 'x1^2', 'x1 x2', 'x1 x3', 'x2^2', 'x2 x3', 'x3^2']
# #
# # r=4
# # D = ['__' for i in range(int((r ** 2 + 3 * r) / 2.0)+1)]
# #
# # for i in range(1, r):
# #     for j in range(0,r-i):
# #         D[ (i*r - (i-1)) + j + 1] = C[ (i*r - (i-1)) + j + i + 1]
# #
# # for i in range(1,r+1):
# #     D[-i] = C[-sum(range(i+1))]
# #
# # print(D)
# #
# # pow = sindy_library.powers_
# # print(pow)
#
#
# C = ['1', 'x0', 'x1', 'x2', 'x3', 'x0x1', 'x0x2', 'x0x3', 'x1x2', 'x1x3', 'x2x3', 'x0x0', 'x1x1', 'x2x2', 'x3x3']
#
# r=4
# D = ['__' for i in range(int((r ** 2 + 3 * r) / 2.0) + 1)]
#
#
#
# c = r
# for i in range(1,r):
#     for j in range(1,r-i+1):
#         ii = c + j
#         D[ii+i] = C[ii]
#     c = ii
#
# for i in range(1,r+1):
#     D[-sum(range(i+1))] = C[-i]
#
# for i in range(0,r+1):
#     D[i] = C[i]
#
#
#
# print(D)
#
#
# print('\n\n\n\n')
#
#
# C = ['1', 'x0', 'x1', 'x2', 'x3', 'x0^2', 'x0 x1', 'x0 x2', 'x0 x3', 'x1^2', 'x1 x2', 'x1 x3', 'x2^2', 'x2 x3', 'x3^2']
#
# r=4
# D = ['__' for i in range(int((r ** 2 + 3 * r) / 2.0)+1)]
#
# c = r
# for i in range(1,r):
#     for j in range(1,r-i+1):
#         ii = c + j
#         D[ii] = C[ii+i]
#     c = ii
#
# for i in range(1,r+1):
#     D[-i] = C[-sum(range(i+1))]
#
# for i in range(0,r+1):
#     D[i] = C[i]
#
# print(D)
#
# pow = sindy_library.powers_
# print(pow)
#
#
# print('\n\n\n\n')
#
#
#
# print(''.join(str(i) for i in np.array([1,2,3]).tolist() + ['a',1,'lala'] ))
# # ZA KATERE ZACETNE PARAMETRE SO MODELI STABILNI???
#
# # I =12
# # alpha = (3 * (I ** 2) + 3) / (2 * (I ** 3) + 4 * I)
# # beta = (2 * (I ** 2) + 1) / ((I ** 4) + 2 * (I ** 2))
# #
# # print(alpha, beta)
#
# def a(b=1,c=1,*args):
#
#
#     print(args == ())
#
# a()
#
#
# # with open('data/data_backup/Lorenz_model_3_[960, 32, 12, 10, 2.5, 15, 0.0001, 5].pkl', 'rb') as inp:
# #     L = pickle.load(inp)  # Lorenz_model_3_evaluated
# #
# # # L.t1 = 20
# # # L.evaluate()
# #
# #
# # print(L.lyapunov_max)
#
# """
# # ignore user warnings
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
#
# np.random.seed(1000)  # Seed for reproducibility
#
# # Integrator keywords for solve_ivp
# integrator_keywords = {}
# integrator_keywords['rtol'] = 1e-12
# integrator_keywords['method'] = 'LSODA'
# integrator_keywords['atol'] = 1e-12
#
# # Generate training data
#
# dt = 0.01
# t_train = np.arange(0, 25, dt)
# t_train_span = (t_train[0], t_train[-1])
# x0_train = [2, 0]
# x_train = solve_ivp(linear_damped_SHO, t_train_span,
#                     x0_train, t_eval=t_train, **integrator_keywords).y.T
#
# # print(x_train)
#
#
# # Fit the model
#
# poly_order = 5
# threshold = 0.05
#
# #library=ps.PolynomialLibrary(degree=poly_order)
#
# def func(Z):
#     return Z[0]
#
#
# # functions = [func]
# functions = [lambda x : x]
# lib = CustomLibrary(library_functions=functions)
#
# model = ps.SINDy(
#     optimizer=ps.STLSQ(threshold=threshold),
#     feature_library=lib,
# )
# model.fit(x_train, t=dt)
# model.print()
#
#
# # Simulate and plot the results
#
# x_sim = model.simulate(x0_train, t_train)
# plot_kws = dict(linewidth=2)
#
# print(model.coefficients())
#
#
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# axs[0].plot(t_train, x_train[:, 0], "r", label="$x_0$", **plot_kws)
# axs[0].plot(t_train, x_train[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
# axs[0].plot(t_train, x_sim[:, 0], "k--", label="model", **plot_kws)
# axs[0].plot(t_train, x_sim[:, 1], "k--")
# axs[0].legend()
# axs[0].set(xlabel="t", ylabel="$x_k$")
#
# axs[1].plot(x_train[:, 0], x_train[:, 1], "r", label="$x_k$", **plot_kws)
# axs[1].plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", **plot_kws)
# axs[1].legend()
# axs[1].set(xlabel="$x_1$", ylabel="$x_2$")
#
# plt.show()
#
# """
#
# """
#     def calculate_FNN_met1(self, eval_path, T, Dt, Dt_n_apart = None, td_dim_span=(None,None)):
#         # Calculates False Nearest Neighbour function. Par. T is the time lag used to construct time-delay
#         # coordinates, Dt is the spacing in time when delay coordinates are constructed and td_dim is a tuple (min, max)
#         # dimension of time delay coordinates vector space (where we estimate FNN). Both T as delta_t should be
#         # multiples of provided dt (from eval), and aldo T should be a multiple of delta_t. Param Dt_n_apart defines
#         # the number of Dt units two compared points should be apart when finding a NN. It is set by default such that
#         # two points can be at min 1 time unit close when searching for NN. A scalar function that is used in the
#         # process is the average of all components at one time instance.
#
#         # Set default value of td_dim
#         if td_dim_span == (None, None):
#             td_dim_min = 0
#             td_dim_max = self.N + 10
#         else:
#             td_dim_min = td_dim_span[0]
#             td_dim_max = td_dim_span[1]
#
#         if Dt_n_apart == None:
#             Dt_n_apart = int(1/Dt)
#
#         with open(eval_path, 'rb') as LM05_inp_eval:
#             ev = pickle.load(LM05_inp_eval)
#
#             dt = ev.t[1] - ev.t[0]
#             t = ev.t
#             x0 = np.sum(ev.y.T, axis=-1) / self.N # average of all components at a time instant
#
#             del LM05_inp_eval, ev
#
#         print('\nCalculating FNN. Parameters:')
#         print('Time delay: ', T)
#         print('dt from evaluation: ', dt)
#         print('new dt:=Dt: ', Dt)
#
#         # intigers pointing to the element equivalent to T and Dt
#         Ti = int(T/dt)
#         Dti = int(Dt/dt)
#
#         # Check if provided T/dt, delta_t/dt and T/delta_t are all intigers
#         if (round(Ti, 10))%1 != 0:  # round because of numerical error
#             raise ValueError('Time_delay should be'
#                              ' some multiple of dt from provided evaluation')
#         if (round(Dti, 10))%1 != 0:  # round because of numerical error
#             raise ValueError('Par. delta_t should be'
#                              ' some multiple of dt from provided evaluation')
#         if (round(T / Dt, 10))%1 != 0:  # round because of numerical error
#             raise ValueError('Par. T should be'
#                              ' some multiple of delta_t from provided evaluation')
#         if (round(len(x0) / Dti, 10)) % 1 != 0:  # round because of numerical error
#             raise ValueError('Number of points from provided evaluation should'
#                              ' be dividable by Dt/dt')
#
#
#         # generate time delay vector function coordinates of dimension td_dim
#         l = int((len(x0) - Ti * (td_dim_max - 1)) / Dti)
#         TD = np.zeros((l, td_dim_max))
#         for i in range(l):
#             for j in range(td_dim_max):
#                 TD[i,j] = x0[i*Dti + j*Ti]
#
#
#         def get_NN(d):
#             # returns the indices of NN for dim. of delayed coord. space d
#
#             NN_ind_arr = np.zeros((l, 2))
#             TDd = TD[:,:d]
#
#             for i, pi in enumerate(TDd):
#                 NNj = None
#                 c = 10**10  # initial distance to NN
#
#                 for j, pj in enumerate(TDd):
#                     if np.abs(i - j) > Dt_n_apart:
#                         c_new = np.linalg.norm(pi-pj)
#                         if c_new < c:
#                             c = c_new
#                             NNj = j
#
#                 NN_ind_arr[i] = [i,NNj]
#
#             return(NN_ind_arr)
#
#
#         fnn = np.zeros(td_dim_max - td_dim_min - 1)
#         d_arr = np.arange(td_dim_min, td_dim_max - 1)
#         NNd = get_NN(td_dim_min)
#
#         for d in d_arr:
#
#             if self.verbose:
#                 print('Calculating for dim. {}/{}'.format(d, td_dim_max-1))
#
#             NNdplus1 = get_NN(d+1)
#
#             score = 0
#             for i in range(l):
#                 if NNd[i,1] == NNdplus1[i,1]:
#                     score += 1
#
#             fnn[d - td_dim_min] = 1 - score / l
#             if self.verbose:
#                 print('fnn(d): ', fnn[d - td_dim_min])
#
#             NNd = NNdplus1
#
#         self.FNN.append({'d_arr' : d_arr,
#                         'fnn' : fnn,
#                         'eval_path' : eval_path,
#                         'T' : T,
#                         'Dt' : Dt,
#                         'Dt_n_apart' : Dt_n_apart,
#                         'td_dim' : td_dim_max})
#
# """
#
# """
#    # def calculate_BCD(self, eval_paths, eps_arr=[0.1]):
#     #     # Box Counting Dimension. Takes in
#     #
#     #     # Load data
#     #     with open(eval_paths[0], 'rb') as L05_2_inp_eval:
#     #         ev = pickle.load(L05_2_inp_eval)
#     #
#     #         x = ev.y.T
#     #         l = len(ev.y.T)
#     #
#     #         del L05_2_inp_eval, ev
#     #
#     #     #################################
#     #     # check for the largest distance between two points - this dist. must not be smaller
#     #     # than the smallest epsilon
#     #
#     #     min_dist = 1e10
#     #     ind = 0
#     #     for i in range(1000,l):
#     #         dist = np.linalg.norm(x[i]-x[0])
#     #         if dist < min_dist:
#     #             min_dist = dist
#     #             ind = i
#     #
#     #     #max_dist = np.linalg.norm(x[1]-x[0])
#     #
#     #     print('max_dist: ', min_dist, ind)
#     #
#     #     del x
#     #     #################################
#     #
#     #     N_arr = []
#     #     d_arr = []
#     #
#     #     def get_unique_num(arr):
#     #         primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71]
#     #         sum = 0
#     #         for i, num in enumerate(arr):
#     #             sum += primes[i] ** (num/10 + 1)
#     #
#     #         return sum
#     #
#     #     # For each epsilon, calculate into which box a point falls and save the box position. Afterwards,
#     #     # turn the array of all boxes into a set (only different elements
#     #     for eps in eps_arr:
#     #
#     #         if False:#eps < min_dist:
#     #             print('eps={} < min_dist between time consecutive points ={}'.format(eps,min_dist))
#     #         else:
#     #
#     #             red_set = np.asarray([])
#     #
#     #             for eval_path in eval_paths:
#     #                 # Load data
#     #                 with open(eval_path, 'rb') as L05_2_inp_eval:
#     #                     ev = pickle.load(L05_2_inp_eval)
#     #
#     #                     x = ev.y.T
#     #                     l = len(ev.y.T)
#     #
#     #                     del L05_2_inp_eval, ev
#     #
#     #                 og_set = np.unique(x // eps, axis=0)
#     #                 print('1:',og_set.size * og_set.itemsize)
#     #
#     #                 red_set = np.concatenate((np.asarray([get_unique_num(i) for i in og_set]),red_set))
#     #                 print('2:',red_set.size * red_set.itemsize)
#     #
#     #                 red_set = np.unique(red_set)
#     #                 print('3:',red_set.size * red_set.itemsize)
#     #
#     #                 del og_set
#     #
#     #
#     #
#     #             N = len(red_set)
#     #             d = np.log(N) / np.log(1/eps)
#     #
#     #             print('eps={}, N={}, d={}'.format(eps, N, d))
#     #
#     #             N_arr.append(N)
#     #             d_arr.append(d)
#     #
#     #     eps_arr = eps_arr[:len(N_arr)]
#     #
#     #     print(eps_arr)
#     #     print(N_arr)
#     #     print(d_arr)
#     #
#     #     self.BCD.append({'d' : d_arr,
#     #                      'eps_arr' : eps_arr,
#     #                      'N_arr' : N_arr})
#  """
#
#
#
#
#
#
#
#
#
#
#
# def calculate_lyapunov_dim(lyap_arr):
#     N = len(lyap_arr)
#
#     sum = 0
#     for i in range(N):
#         s = sum + lyap_arr[i]
#         if s >= 0:
#             sum = s
#         else:
#             break
#
#     print(sum)
#     print(lyap_arr[i])
#
#     d = i + 1 / abs(lyap_arr[i]) * sum
#
#     return d
#
#
# lyaps_20_2_30 = [2.596409265343265, 1.259614555695911, 0.6295821497260383, 0.2502340989852911, -0.021447083086951133, -0.13615920176474225, -0.3726483370914156, -0.5510901276392345, -0.6305570076351025, -0.7076274186190259, -0.8227039676158484, -0.9017573215423863, -0.9440226982433677, -1.00635315235581, -1.0567391661196146, -1.110766846290399, -1.1440472865375177, -1.2170308357453552, -1.3070710393234888, -1.4028458556100496]
# print(calculate_lyapunov_dim(lyaps_20_2_30))
#
#
# print(np.random.normal(loc=0,scale=1,size=(10)))
