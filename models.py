import numpy as np
import scipy.integrate as integrate
import random as rd
import hashlib
import pickle
import pysindy as ps
from scipy import signal
from matplotlib import pyplot as plt
import sys
import copy
import pandas as pd
import time
import seaborn as sn
import matplotlib.pyplot as plt
import re

# a class that is used to mask results obtained from surrogate model prediction as an evaluation object of solve_ivp
# (so that we dont need to change most of the code for surrogate model)
class ev:

    def __init__(self, y, t_eval):
        self.y = y
        self.t = t_eval


def get_unique_name(params):
    # returns a unique string, generated with respect to given parameters
    # input must be a list whose elements are single values
    unique_id = ''.join([str(i) for i in params]).encode('utf-8')

    h = hashlib.new('sha1')
    h.update(unique_id)

    return h.hexdigest()

#################################################################################################
#########################    'ANALYTIC' TERMS    ################################################
#################################################################################################

class cons_term:
    # a class needed for 'analytic' representation of ODE

    def __init__(self, amplitude):
        self.amp = amplitude

    def __repr__(self):
        return '{} 1'.format(self.amp)


class lin_term:
    # a class needed for 'analytic' representation of ODE

    def __init__(self, amplitude, n, N):
        self.amp = amplitude    # float
        self.N = N
        self.n = n % N        # int

    def __repr__(self):

        if self.amp == 0:
            return ''
        else:
            return '+ {} x{}'.format(self.amp, self.n)

    def __add__(self, otherTerm):
        if self.n == otherTerm.n:
            return lin_term(self.amp + otherTerm.amp, self.n, self.N)
        else:
            raise ValueError('Not the same term (got x{} and x{}'.format(self.n, otherTerm.n))

    def __mul__(self, scalar):
        if type(scalar) == type(1) or type(scalar) == type(1.0):
            return quad_term(self.amp * scalar, self.n, self.N)
        else:
            raise ValueError('Only multiplication with scalar possible')


class quad_term:
    # a class needed for 'analytic' representation of ODE

    def __init__(self, amplitude, n1, n2, N):
        self.amp = amplitude    # float
        self.N = N
        if n1 % N < n2 % N:
            self.n1 = n1 % N        # int
            self.n2 = n2 % N        # int
        else:
            self.n1 = n2 % N        # int
            self.n2 = n1 % N        # int

    def __repr__(self):

        if self.amp == 0:
            return ''
        else:
            if self.n1 == self.n2:
                return '+ {} x{}^2'.format(self.amp, self.n1)
            else:
                return '+ {} x{} x{}'.format(self.amp, self.n1, self.n2)

    def __add__(self, otherTerm):
        if self.n1 == otherTerm.n1 and self.n2 == otherTerm.n2:
            return quad_term(self.amp + otherTerm.amp, self.n1, self.n2, self.N)
        else:
            raise ValueError('Not the same term (got (x{} x{}) and (x{} x{})'.format(self.n1, self.n2, otherTerm.n1, otherTerm.n2))

    def __mul__(self, scalar):
        if type(scalar) == type(1) or type(scalar) == type(1.0):
            return quad_term(self.amp * scalar, self.n1, self.n2, self.N)
        else:
            raise ValueError('Only multiplication with scalar possible')

#################################################################################################
#########################      LORENZ MODELS       ##############################################
#################################################################################################


# MODEL 2 FROM LORENZ 2005
class L05_2:
    # Class based on the Model 2 presented in a paper by Lorenz: 'Designing Chaotic Models' (2005).
    # For K = 1 this simplifies to Model 1


    def __init__(self, N, K, F, integrator_type = 'LSODA'):

        # ----- Intrinsic Model Parameters -----
        self.N = N
        self.K = K
        self.F = F

        # Integrator, 'LSODA' by default (other possible choices: 'RK45' 'RK23', 'DOP853', 'BDF')
        self.integrator = integrator_type

        # Terms in equation resented in practical form. Can be evaluated by calling .get_terms()
        self.terms = None

        # To be evaluated ( by calling .evaluate(...) ). Each item in evaluation_path is a unique string
        # path pointing to a solve_ivp object (in a sense of a relative file path)
        self.evaluation_paths = []


        # Properties of an evaluated trajectory (min,max,avr,std). Evaluated with self.calculate_eval_props
        self.eval_props = []

        # To be evaluated ( by calling .get_lyapunov_max(...) ). When evaluated, a dictionary is created
        # with keys: 'average', 'all generated', 'delta start', 'delta stop'. Each item in
        # lyapunov_paths is a unique string that represents a relative path to an object where the result is stored
        self.lyapunov_max = []

        # To be evaulated with calculate_lyapunov_spectrum. Each element is a dict with keys 'init_cond',
        # 'renormalization time', 'max iteration number', 'delta_start', 'average of log10(delta_stop)',
        # 'lyapunov exponents', 'lyapunov std'.
        self.lyapunov_spectrum = []
        # Power Spectrum Densities. Evaluated with self.calculate_PSD. Each element in a list is a dict. with keys
        # 'freqs', 'psd', 'window_length', 'eval_path'
        self.PSD = []

        # Periodic moments. Evaluated with self.calculate_periodic_moments. Each element is a dict. with keys k_arr',
        # 'mom_arr', eval_path'
        self.moments = []

        # Mutual information function. Evaluated with self.calcuate_mutual_information. Each element is a dict with
        # keys 'time_delays', 'mutual_info', 'partition_num' and 'eval_path'
        self.mutual_information = []

        # False Nearest Neighbour function. Evaluated with self.calculate_FNN. Each element is a dict with keys
        # 'd_arr', 'fnn', 'thr_choice', 'eval_path', 'T', 'Dt', 'time_units_apart', 'td_dim_span'
        self.FNN_v1 = []

        # False Nearest Neighbour function. Evaluated with self.calculate_FNN. Each element is a dict with keys
        # 'd_arr', 'E1_arr', 'E2_arr', 'eval_path', 'T', 'Dt', 'time_units_apart', 'td_dim_span'
        self.FNN_v2 = []

        # Print object parameters
        self.print_parameters()

        # If true, print extra info such as elapsed time when integrating etc.
        self.verbose = False


    def print_parameters(self):
        print('\n#########################################\n')
        print('N: ', self.N)
        print('K: ', self.K)
        print('F: ', self.F)
        print('Integrator:', self.integrator)
        print('Evaluation paths:')
        for path in self.evaluation_paths:
            print('                  {}'.format(path))
        print('Lyapunov max paths:')
        for path in self.lyapunov_max:
            print('                {}'.format(path))
        print('Lyapunov spectrum:')
        for i in self.lyapunov_spectrum:
            print(i)
        print('PSD:')
        for i in self.PSD:
            print(i)
        print('Periodic moments:')
        for i in self.moments:
            print(i)
        print('FNNv1:')
        for i in self.FNN_v1:
            print(i)
        print('FNNv2:')
        for i in self.FNN_v1:
            print(i)
        if self.terms != None:
            self.stringify_ODE()
        print('\n#########################################\n')


    def mod(self, n):
        # just a cleaner notation for modulo operation

        return n % self.N


    def br(self, X1, X2, K, n):
        # the bracket formula; eq. (10) and (9) in Lorenz's 2005 paper

        sum = 0

        if K % 2 == 1:
            J = int((K - 1) / 2)

            for i in range(-J,J+1):
                for j in range(-J,J+1):
                    sum1 = X1[self.mod(n - 2*K - i)] * X2[self.mod(n - K - j)]
                    sum2 = X1[self.mod(n - K + j - i)] * X2[self.mod(n + K + j)]
                    sum += sum2 - sum1

        elif K % 2 == 0:
            J = int(K / 2)

            for i in range(-J, J + 1):
                for j in range(-J, J + 1):
                    sum1 = X1[self.mod(n - 2 * K - i)] * X2[self.mod(n - K - j)]
                    sum2 = X1[self.mod(n - K + j - i)] * X2[self.mod(n + K + j)]
                    sum3 = sum2 - sum1
                    if abs(i) == J:
                        sum3 /= 2
                    if abs(j) == J:
                        sum3 /= 2
                    sum += sum3

        return sum / (K ** 2)


    def model(self, t, Z):
        # Representing f(Z(t)) in dZ(t)/dt = f(Z(t))

        # Logging integration time
        if self.verbose:
            tt = 't = ' + str(round(t, 5))
            sys.stdout.write('\r' + tt)

        def Z_dot_n(n):
            # Lorenz 2005, eq. (8)

            return self.br(Z, Z, self.K, n) - Z[n] +  self.F

        return Z_dot_n(np.arange(0, self.N))


    def evaluate(self, init_cond, dt, t_stop, save=True):
        # The core function of this class, called to evaluate the integral of model() function.
        # The result is stored in self.evaluation

        # ----- Integration Parameters -----
        init_cond = np.asarray(init_cond)
        t_eval = np.arange(0, t_stop, dt)

        # Seed for reproducibility
        np.random.seed(1000)

        # Integrator keywords for solve_ivp
        integrator_keywords = {}
        integrator_keywords['method'] = self.integrator
        integrator_keywords['rtol'] = 1e-12
        integrator_keywords['atol'] = 1e-12
        #integrator_keywords['min_step'] = 1e-7

        # Evaluation
        if self.verbose:
            print('\nEvolve:')

        evaluation = integrate.solve_ivp(fun = self.model, t_span = (t_eval[0], t_eval[-1]),
                                         y0 = init_cond, t_eval = t_eval, **integrator_keywords)
        if self.verbose:
            print('\n')

        # Save the evaluation as a separate object with a unique name
        parameters = {'N':self.N,
                      'K':self.K,
                      'F':self.F,
                      'dt':dt,
                      't1':t_stop}
        name = 'data/eval_{}_{}_{}.pkl'.format(parameters,get_unique_name(init_cond.tolist()),str(time.time()))
        if save:
            self.evaluation_paths.append(name)
            with open(self.evaluation_paths[-1], 'wb') as outp:
                pickle.dump(evaluation, outp, pickle.HIGHEST_PROTOCOL)
                del outp

        final_cond=evaluation.y.T[-1]
        del evaluation

        return name, final_cond


    def calculate_eval_props(self, eval_path):

        with open(eval_path, 'rb') as L05_2_inp_eval:
            ev = pickle.load(L05_2_inp_eval)

        x = ev.y.T

        r = {'eval_path' : eval_path,
             'mi' : np.min(x),
             'ma' : np.max(x),
             'av' : np.average(x),
             'sd' : np.std(x),
             'true_av' : np.average(x, axis=0)
            }

        r['true_sd'] = ( 1/len(x) * np.sum([np.linalg.norm(r['true_av']-i)**2 for i in x]) ) ** 0.5

        if self.verbose == True:
            for i,j in r.items():
                print(i,': ',j)

        self.eval_props.append(r)

        return r


    def get_terms(self):
        # Returns 'a matrix' f(X) (RHS of ODE) in a form that might be convenient for
        # comparison and error estimation of ODE discovered by SINDy. The output is based
        # on classes' quad_term, lin_term,...

        def group_lin_terms(arr):
            # returns an ordered, grouped list of terms of order 1

            new_arr = []

            for i in range(self.N):

                term = lin_term(0, i, self.N)

                for term_in_arr in arr:
                    if term_in_arr.n == term.n:
                        term += term_in_arr

                if term.amp != 0:
                    new_arr.append(term)

            return new_arr

        def group_quad_terms(arr):
            # returns an ordered, grouped list of terms of order 2

            new_arr = []

            for i in range(self.N):
                for j in range(i, self.N):

                    term = quad_term(0,i,j,self.N)

                    for term_in_arr in arr:
                        if term_in_arr.n1 == term.n1 and term_in_arr.n2 == term.n2:
                            term += term_in_arr

                    if term.amp != 0:
                        new_arr.append(term)

            return new_arr


        def br_str(X1_amps, X2_amps, K, n):
            # The bracket formula; eq. (10) and (9) in Lorenz's 2005 paper. Return of
            # this function is of analytic nature

            amplitude = X1_amps[n] * X2_amps[n]
            terms = []

            if K % 2 == 1:
                J = int((K - 1) / 2)

                for i in range(-J, J + 1):
                    for j in range(-J, J + 1):
                        term1 = quad_term((-1) * amplitude / K ** 2, n - 2 * K - i, n - K - j, self.N)
                        term2 = quad_term(  1  * amplitude / K ** 2, n - K + j - i, n + K + j, self.N)

                        terms.append(term1)
                        terms.append(term2)


            elif K % 2 == 0:
                J = int(K / 2)

                for i in range(-J, J + 1):
                    for j in range(-J, J + 1):
                        term1 = quad_term((-1) * amplitude / K ** 2, n - 2 * K - i, n - K - j, self.N)
                        term2 = quad_term(  1  * amplitude / K ** 2, n - K + j - i, n + K + j, self.N)
                        if abs(i) == J:
                            term1.amp /= 2
                            term2.amp /= 2
                        if abs(j) == J:
                            term1.amp /= 2
                            term2.amp /= 2

                        terms.append(term1)
                        terms.append(term2)


            return terms

        # Compute the coefficients to start with
        Z = np.ones(self.N)

        def Z_dot_n_str(n):
            # Lorenz 2005, eq. (15)
            print('Calculating analytical terms for component {}/{}'.format(n+1, self.N))

            quadratic_terms = group_quad_terms(br_str(Z, Z, self.K, n))
            linear_terms = group_lin_terms([lin_term(- Z[n], n, self.N)])
            constant_terms = [cons_term(self.F)]

            return constant_terms + linear_terms + quadratic_terms


        self.terms = [Z_dot_n_str(i) for i in range(self.N)]
        print('\n')


    def stringify_ODE(self):
        # A wrapper for get_terms() method when one wishes to print out
        # the form of the ODE

        if self.terms == None:
            self.get_terms()

        print('ODE:')
        for i in range(self.N):
            print("(x{})' =".format(i), *self.terms[i])


    def calculate_lyapunov_max(self, eval_path, delta_start=10 ** (-8), delta_stop=0.5, save=True):
        # Calculate maximal Lyapunov exponent. This method should be used to estimate error growth time that
        # can later be used in calculation of the whole lyapunov spectrum (because the later method requires a fixed
        # stopping time). Method calculate_lyapunov_max is also useful if one is calculating lyap. exp. of a higher
        # dimensional system that is computationally slow to integrate (so that we compute only one lyap. exp., not
        # the whole spectrum). But, because it requires (cuz of the solve_ivp stopping condition) a relatively big
        # delta_stop it is not very precise

        # Load an already calculated trajectory
        with open(eval_path, 'rb') as L05_2_inp_eval:
            evaluation = pickle.load(L05_2_inp_eval)
            del L05_2_inp_eval

        # Time parameters
        t_train_span = (evaluation.t[0], evaluation.t[-1])
        dt = evaluation.t[1] - evaluation.t[0]

        # Parameters for calculating Lyapunov exp.
        total_t = 0
        times = []
        normalization_factors = []
        #error_growth = []
        lyap_enum = 0

        def time_to_normalize(t, Z):
            # A function to be passed to solve_ivp(events= )
            # When the error (initially of norm delta_min grows to the size of delta_max) this function hits zero
            # and the integration is to be stopped

            # perturbed vector = original trajectory - perturbed trajectory
            pert_vec = np.asarray(evaluation.y.T[int(round((total_t + t) / dt))]) - np.asarray(Z)
            error = np.linalg.norm(pert_vec)
            #error_growth[lyap_enum].append([t, error])

            if self.verbose:
                tt = 't = ' + str(round(t, 5)) + 'error: ' + str(np.linalg.norm(pert_vec)) + 'delta_stop = ' + str(delta_stop)
                sys.stdout.write('\r' + tt)

            return error - delta_stop

        # Seed for reproducibility
        np.random.seed(1000)

        # Integrator keywords for solve_ivp
        integrator_keywords = {}
        integrator_keywords['method'] = self.integrator
        integrator_keywords['rtol'] = 1e-12
        integrator_keywords['atol'] = 1e-12
        #integrator_keywords['min_step'] = 1e-7
        time_to_normalize.terminal = True  # Stop integration when the event happen

        # We start with an init cond. + perturbation vector of norm
        # delta_min, equally big on all components
        Z0 = evaluation.y.T[0] + np.ones(self.N) * np.sqrt(delta_start ** 2 / self.N)

        while True:
            try:
                #error_growth.append([])

                integrate_before_normalize = integrate.solve_ivp(self.model,
                                                                 t_train_span, Z0,
                                                                 t_eval=evaluation.t,
                                                                 events=time_to_normalize,
                                                                 **integrator_keywords)

                # Time at which the event occurred (size of an error reached delta_stop)
                stopping_time = integrate_before_normalize.t_events[0][0]
                total_t += stopping_time
                times.append(stopping_time)

                # Caluculate the displacement (error) vector
                displacement = (integrate_before_normalize.y_events[0][0] - evaluation.y.T[int(round(total_t / dt))])
                beta = np.linalg.norm(displacement) / delta_start  # Normalization factor
                normalization_factors.append(beta)

                # Define a new perturbed vector as the original solution at
                # current time plus normalized perturbation vector
                Z0 = evaluation.y.T[int(round(total_t / dt))] + displacement * 1 / beta

                lyap_enum += 1

            except IndexError:
                print('The end of evaluated trajectory reached')
                break

        # Calculate all generated Lyapunov coefficients
        lambdas = 1 / np.asarray(times) * np.log(np.asarray(normalization_factors))

        # Dict. with output parameters
        res = {'eval_path':eval_path,
               'average lyapunov': np.average(lambdas),
               'std lyapunov':np.std(lambdas),
               'all generated lyapunov': lambdas,
               'average stopping time': np.average(times),
               'std stopping time':np.std(times),
               'all generated stopping time': times,
               'sample size': len(lambdas),
               #'error growth': [np.asarray(i) for i in error_growth[:-1]],  # The last one didn't reach the end
               'delta start': delta_start,
               'delta stop': delta_stop,
               'corresponding evaluation path': eval_path,
               'precision': dt}


        # Save result
        self.lyapunov_max.append(res)

        del evaluation

        return res


    def calculate_lyapunov_spectrum(self, init_cond, T, max_iter_num = 50, delta_start =10 ** (-8), N=None):
        # Calculate lyapunov spectrum. Parameter T should be chosen similar to ALREADY CALCULATED
        # average 'stopping time' from calculate_lyapunov_max i.e. it is the time in which the trajectories will
        # be normalized (Gram-Schmidt). Parameter max_iter_num is the number of times we estimate the renormalization factors
        # needed to calculate Lyapunov exponents. N is the number of lyapunov exponents calculated (mast be smaller then selfN

        if N > self.N:
            raise ValueError('N must be smaller or equal than the dimension of the sysem (self.N)')
        if N == None:
            N = self.N

        # Initialize some parameters
        init_cond = np.asarray(init_cond)
        t_span = (0,  T)
        all_betas = np.zeros((max_iter_num, N))
        deltas_stop = np.zeros(max_iter_num)  # log deviations at which renormalization happens - ideally this
                                              # number should be around 10^(-2) (numerical recommendations)

        print('\n####################################################\n')
        print('Calculating Lyapunov spectrum:')
        print('Stopping time (T): ', T, '(=time in which the the trajectories will be normalized (Gram-Schmidt))')
        print('Number of iterations: ', max_iter_num)
        print('Number of lyap. exps. generated: ', N)


        # Seed for reproducibility
        np.random.seed(1000)

        # Integrator keywords for solve_ivp
        integrator_keywords = {}
        integrator_keywords['method'] = self.integrator
        integrator_keywords['rtol'] = 1e-12
        integrator_keywords['atol'] = 1e-12
        integrator_keywords['min_step'] = 1e-7

        # We start with an N+1 different initial conditions; og. init. cond and N init.
        # conds. perturbed with delta_start
        Z0 = np.asarray([init_cond for i in range(N + 1)])
        for i in range(1, N + 1):
            Z0[i, i - 1] += delta_start

        iter_num = 0
        while iter_num < max_iter_num:

            iter_num += 1
            if self.verbose:
                print('\nIteration: ', iter_num)

            # Perform integrations to time T
            integ = [integrate.solve_ivp(self.model,
                                              t_span,
                                              Z0_i,
                                              **integrator_keywords).y.T[-1]
                    for Z0_i in Z0]

            # Calculate displacement sizes and store them as tuples with their correspondent vectors. Afterwards sort
            # the tuples from the one with biggest to the smallest displacement. Note that sorting is not neccessary
            # here, we only do it due to better numerical stability of the algorithm
            displ = [[np.linalg.norm(integ_i - integ[0]), integ_i - integ[0]] for integ_i in integ[1:]]
            displ = sorted(displ, key=lambda x: x[0], reverse=True)

            renormalized_vector_norms = np.zeros(N)  # normalization factors
            e = np.zeros((N, self.N))  # ortonormalized_basis

            # Gram - Schmidt orthonormalizing
            for i in range(N):
                e[i,:] = displ[i][1]
                for j in range(i):
                    e[i,:] -= np.dot(displ[i][1], e[j,:]) * e[j,:]

                renormalized_vector_norms[i] = np.linalg.norm(e[i])
                e[i,:] = e[i,:] / renormalized_vector_norms[i]

            # Save renormalization factors
            all_betas[iter_num-1,:] = sorted(renormalized_vector_norms / delta_start, reverse=True)
            deltas_stop = all_betas[iter_num-1,0] * delta_start  # save the biggest renorm. factor

            # Re-initiate new initial conditions for the next loop
            Z0[0] = integ[0]
            for i in range(N):
                Z0[i+1] = integ[0] + e[i,:] * delta_start

        lambdas = [1.0/(T*max_iter_num) * np.sum([np.log(all_betas[j,i]) for j in range(max_iter_num)]) for i in range(N)]
        lambdas_std = [np.std([1.0/(T)*np.log(all_betas[j,i]) for j in range(max_iter_num)]) for i in range(N)]

        results = {'init_cond' : init_cond,
                   'renormalization time' : T,
                   'max iteration number' : max_iter_num,
                   'delta_start' : delta_start,
                   'average of log10(delta_stop)' : np.average(np.log10(deltas_stop)),
                   'lyapunov exponents': lambdas,
                   'lyapunov std': lambdas_std}

        self.lyapunov_spectrum.append(results)
        return results


    def calculate_PSD(self, eval_path, nperseg=1e5, window='hanning'):
        # Calculate Power Spectral Density for specified evaluated trajectory. Only one (but long) trajectory is
        # needed if we suppose the attractor has an invariant natural measure. Parameters window is an array of
        # consecutive integers (< trajectory length) that determines the window for Welch's method. It is correlated
        # to the resolution of the result.
        # Results are saved under

        print('Calculating PSD ... eval_path={}'.format(eval_path))

        with open(eval_path, 'rb') as L05_2_inp_eval:
            ev = pickle.load(L05_2_inp_eval)
            del L05_2_inp_eval

        # first component centered around zero
        x0 = ev.y.T[:,0]- np.average(ev.y.T[:,0])

        freqs, psd = signal.welch(x=x0, fs=(ev.t[1]-ev.t[0])**(-1), nperseg=int(nperseg), window=window)

        self.PSD.append({'dt':ev.t[1]-ev.t[0],
                         't_stop':ev.t[-1],
                         'freqs':freqs,
                         'psd':psd,
                         'window':window,
                         'npesreg':nperseg,
                         'eval_path':eval_path})

        print('Done')


    def calculate_periodic_moments(self, eval_path, k_arr):
        # Return the time average of a function cos(kx) for an array of k's

        with open(eval_path, 'rb') as L05_2_inp_eval:
            ev = pickle.load(L05_2_inp_eval)
            del L05_2_inp_eval

        dt = ev.t[1] - ev.t[0]
        T = ev.t[-1]
        f_Z_integs = np.zeros(len(k_arr))

        f = lambda k, x: np.cos(k*x)
        #f = lambda k, x: np.exp(1j*k * x)

        # for each k avarage the value of f over time and over all trajectories
        for i in range(len(k_arr)):
            if self.verbose:
                print('Calculating periodic moment for k={} ({}/{})'.format(k_arr[i], i+1, len(k_arr)))
            for j in range(self.N):
                f_Z_integs[i] += 1/(T*self.N) * np.trapz(y=f(k_arr[i], ev.y.T[:, j]), dx=dt)

        self.moments.append({'k_arr' : k_arr,
                             'mom_arr' : f_Z_integs,
                             'eval_path' : eval_path})

        print('Done')


    def calculate_mutual_information(self, eval_path, time_delays, partition_num=50):
        # Mutual information as a function of time delay. Values in an array time_delays should be
        # some multiples of dt from provided evaluation. A scalar function that is used in the process
        # is the sum / N of all components at one time instance

        with open(eval_path, 'rb') as L05_2_inp_eval:
            ev = pickle.load(L05_2_inp_eval)

            dt = ev.t[1] - ev.t[0]
            t = ev.t
            x0 = np.sum(ev.y.T, axis=-1) / self.N # average of all components at a time instant

            del L05_2_inp_eval, ev

        print('\nCalculating Mutual Information:')
        print('dt from evaluation: ', dt)
        print('max time from evaluation: ', t[-1])
        print('partition num.: ', partition_num)
        print('\n')

        # Check if provided time delays are multiples of dt
        for T in time_delays:
            if (round(T/dt, 10))%1 != 0:  # round because of numerical error
                raise ValueError('Values in an array time_delays should be'
                                 ' some multiples of dt from provided evaluation')

        partitions_borders = np.linspace(np.min(x0), np.max(x0), partition_num + 1, endpoint=True)
        mut_inf = np.zeros(len(time_delays))

        def bin(x, y):
            # Return the mutual partition of two numbers
            xi, yi = -1, -1
            for i in range(partition_num):
                if partitions_borders[i + 1] >= x:
                    xi = i
                    break
            for i in range(partition_num):
                if partitions_borders[i + 1] >= y:
                    yi = i
                    break
            if not (xi==-1 and yi==-1):
                return xi,yi
            else:
                raise ValueError('x or y not in any partition')

        for i,T in enumerate(time_delays):
            if self.verbose:
                print('Calculating for time delay {} ({}/{})'.format(round(T,10), i+1, len(time_delays)))


            P_mn = np.zeros((partition_num, partition_num)) # Num. of events when x(t) is in bin m and x(t+T) is in bin n
            P_m = np.zeros(partition_num)  # Num. of events when x(t) is in bin n
            Ti = int(T/dt) # index that will point to time t + T

            for j in range(len(x0)-Ti):
                m, n = bin(x0[j], x0[j+Ti])
                P_m[m] += 1
                P_mn[m,n] += 1

            # Num. of events to probabilities
            P_m = P_m / np.sum(P_m)
            P_mn = P_mn / np.sum(P_mn)

            for m in range(partition_num):
                if P_m[m] != 0:
                    mut_inf[i] += - 2*P_m[m] * np.log(P_m[m])
                for n in range(partition_num):
                    if P_mn[m,n] != 0:  # we take out zeros otherwise log(0) = -inf and PlogP produces a nan
                        mut_inf[i] += P_mn[m,n] * np.log(P_mn[m,n])

        results = {'time_delays' : time_delays,
                   'mutual_info' : mut_inf,
                   'partition_num' : partition_num,
                   'eval_path' : eval_path
                   }

        self.mutual_information.append(results)

        return results


    def calculate_FNN_v1(self, eval_path, T, Dt, R_tol=10, A_tol=1, norm='l2', indices_apart = None, td_dim_span=(None, None)):
        # Calculates False Nearest Neighbour function. Par. T is the time lag used to construct time-delay
        # coordinates, Dt is the spacing in time when delay coordinates are constructed and td_dim is a tuple (min, max)
        # dimension of time delay coordinates vector space (where we estimate FNN). Both T as delta_t should be
        # multiples of provided dt (from eval), and aldo T should be a multiple of delta_t. A scalar function that is
        # used in the process is the sum / N of all components at one time instance. td_dim_span[0]=td_dim_min must be
        # at least 1. R_tol and A_tol are method intrinsic parameters (Determining embedding dimension for phase-space
        # reconstruction using a geometrical construction, M.b. Kennel, R. Brown, H.D.I. Abarbanel, 1992)
        # Additional parameter to the ones proposed in the article is the 'norm' of the distance between two points. In
        # the original article l2 norm is used, but one can choose here also dimension-normalized l2 ('l2/d') and the
        # supremum ('max') norm. Parameter indices_apart defines the minimal distance in array between two points where
        # if they can be checked for NN. The purpose of this parameter is that two points consecutive in time must not
        # identified as as NN.


        # Set default value of td_dim
        if td_dim_span == (None, None):
            td_dim_min = 1
            td_dim_max = self.N + 2
        else:
            if td_dim_span[0] < 1:
                raise ValueError('Min time delay dimension cannot be smaller then 1')
            td_dim_min = td_dim_span[0]
            td_dim_max = td_dim_span[1] + 2

        with open(eval_path, 'rb') as L05_2_inp_eval:
            ev = pickle.load(L05_2_inp_eval)

            dt = ev.t[1] - ev.t[0]
            t = ev.t
            x0 = np.sum(ev.y.T, axis=-1) / self.N # average of all components at a time instant
            #x0 = ev.y.T[:,0] # first component

            ##############################################################################################

            integrator_keywords = {}
            integrator_keywords['rtol'] = 1e-12
            integrator_keywords['method'] = 'LSODA'
            integrator_keywords['atol'] = 1e-12
            integrator_keywords['min_step'] = 1e-7

            def lorenz(t, x, sigma=10, beta=2.66667, rho=28):
                return [
                    sigma * (x[1] - x[0]),
                    x[0] * (rho - x[2]) - x[1],
                    x[0] * x[1] - beta * x[2],
                ]

            dt = 0.5

            t = np.arange(0, 1000, dt)
            t_span = (t[0], t[-1])
            x0_train = [5.94857223, -0.12379429, 31.24273752]
            x_train = integrate.solve_ivp(lorenz, t_span, x0_train, t_eval=t, **integrator_keywords).y.T
            x0 = x_train[:,0]


            ##############################################################################################

            del L05_2_inp_eval, ev

        print('\nCalculating FNNv1. Parameters:')
        print('Time delay (T): ', T)
        print('dt from evaluation: ', dt)
        print('max time from evaluation: ', t[-1])
        print('new dt:=Dt: ', Dt)
        print('R_tol: ', R_tol)
        print('A_tol: ', A_tol)
        print('norm: ', norm)

        # intigers pointing to the element equivalent to T and Dt
        Ti = int(T/dt)
        Dti = int(Dt/dt)

        # Check if provided T/dt, delta_t/dt and T/delta_t are all integers
        if (round(Ti, 10))%1 != 0:  # round because of numerical error
            raise ValueError('Time_delay should be'
                             ' some multiple of dt from provided evaluation')
        if (round(Dti, 10))%1 != 0:  # round because of numerical error
            raise ValueError('Par. delta_t should be'
                             ' some multiple of dt from provided evaluation')
        if (round(T / Dt, 10))%1 != 0:  # round because of numerical error
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
            indices_apart = int(l/100)
        print('indices apart: ', indices_apart)

        # Generate time-delay coorinates
        TD = np.zeros((l, td_dim_max))
        for i in range(l):
            for j in range(td_dim_max):
                TD[i,j] = x0[i*Dti + j*Ti]


        c = 0
        for i in range(indices_apart, l):
            c += np.abs(TD[0,0] - TD[i,0])
        print('Avarage dist. from 1st point to other points : ', c/(l-indices_apart))
        print('\n')



        def get_NN(d):
            # returns the indices of NN for dim. of delayed coord. space d, the distance squared
            # between these two points, and the distance squared between points in d+1 dim.
            # (returns [i, NN_i,  dist,  dist_of_dplus1] for each index i)

            # Initialize arrays
            NN_ind_arr = np.zeros((l, 4))
            TDd = TD[:,:d]    # shape = (l,td_dim_max)
            TDdplus1 = TD[:,d]   #shape = l

            # Find NN in dim. of time delay d and save the  dist. of every pair
            for i, pi in enumerate(TDd):
                NNj = None
                c = 10**10  # initial  distance to NN

                # Find NN
                for j, pj in enumerate(TDd):
                    if np.abs(i - j) > indices_apart:

                        if norm == 'max':
                            c_new = np.linalg.norm(pi-pj, ord=np.inf)
                        elif norm == 'l2':
                            c_new = np.linalg.norm(pi-pj)
                        elif norm == 'l2/d':
                            c_new = np.linalg.norm(pi-pj) / d**0.5
                        else:
                            raise ValueError("Parameter 'norm' should be one of ('max', 'l2', 'l2/d')")

                        if c_new < c:
                            c = c_new
                            NNj = j

                # Calculate the increase in dist. of every pair for dim. of time delay vector space d+1
                cc = np.abs(TDdplus1[i] - TDdplus1[NNj])

                # Save result
                NN_ind_arr[i] = [i,NNj, c, cc]
                #print([i,NNj, c, cc])


            return NN_ind_arr

        # Initialize return arrays
        fnn = []
        d_arr = []
        threshold_choice = [] # for each d, if R_tol * NNd[i,2] was always chosen for threshold the value is 1
                              # and 0 if times A_tol * var was always chosen to be the threshold (and linear inbetween)

        # Attractor size bound
        A = A_tol * var

        # Calculate fnn for every d
        for d in np.arange(td_dim_min, td_dim_max):

            if self.verbose:
                print('\nCalculating for dim. {}/{}'.format(d, td_dim_max-1))

            NNd = get_NN(d)

            score = 0
            thr = 0

            NNdplus1_avr = 0
            NNd_avr = 0
            ind_apart_avr = 0

            for i in range(l):

                R = R_tol * NNd[i,2]

                NNdplus1_avr += NNd[i,3]
                NNd_avr += NNd[i,2]
                ind_apart_avr += np.abs(NNd[i,0] - NNd[i,1])

                if d != td_dim_min and fnn[-1] < 0.1:
                    threshold = np.max([R, A])
                else:
                    threshold = R

                if threshold == R:
                    thr += 1

                if NNd[i,3] > threshold:
                    score += 1


            d_arr.append(d)
            fnn.append(score / l)
            threshold_choice.append(thr / l)

            if self.verbose:
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


        self.FNN_v1.append({'d_arr' : np.asarray(d_arr),
                            'fnn' : np.asarray(fnn),
                            'thr_choice' : threshold_choice,
                            'eval_path' : eval_path,
                            'T' : T,
                            'Dt' : Dt,
                            'inidces_apart' : indices_apart,
                            'td_dim_span' : td_dim_span})


    def calculate_FNN_v2(self, eval_path, T, Dt, norm = 'l2', time_units_apart = None, td_dim_span=(None, None)):
        # Calculates False Nearest Neighbour according to the approach proposed in 'Practical method for determining
        # the minimum embedding dimension of a scalar time series' by Liangyue Cao (Physica D, 1997). Par. T is the
        # time lag used to construct time-delay coordinates, Dt is the spacing in time when delay coordinates are
        # constructed and td_dim is a tuple (min, max) dimension of time delay coordinates vector space (where we
        # estimate FNN). Both T as delta_t should be multiples of provided dt (from eval), and aldo T should be a
        # multiple of delta_t. Param time_units_apart defines the number of time units two compared points should be
        # apart when finding a NN. It is set by default such that two points can be at min Dt time unit close when
        # searching for NN. A scalar function that is used in the process is the sum / N of all components at one time
        # instance. td_dim_span[0]=td_dim_min must be at least 1.
        # Additional parameter to the ones proposed in the article is the 'norm' of the distance between two points. In
        # the original article supremum ('sup') norm is used, but one can choose here also dimension-normalized l2
        # ('l2/d') and the regular l2 ('l2') norm


        # Set default value of td_dim
        if td_dim_span == (None, None):
            td_dim_min = 1
            td_dim_max = self.N + 3
        else:
            if td_dim_span[0] < 1:
                raise ValueError('Min time delay dimension cannot be smaller then 1')
            td_dim_min = td_dim_span[0]
            td_dim_max = td_dim_span[1] + 1

        if time_units_apart == None:
            time_units_apart = Dt

        with open(eval_path, 'rb') as L05_2_inp_eval:
            ev = pickle.load(L05_2_inp_eval)

            dt = ev.t[1] - ev.t[0]
            t = ev.t
            x0 = np.sum(ev.y.T, axis=-1) / self.N # average of all components at a time instant
            # x0 = ev.y.T[:,0] # first component


            del L05_2_inp_eval, ev

        print('\nCalculating FNNv2. Parameters:')
        print('Time delay (T): ', T)
        print('dt from evaluation: ', dt)
        print('max time from evaluation: ', t[-1])
        print('new dt:=Dt: ', Dt)
        print('time_units_apart: ', time_units_apart)
        print('norm: ', norm)
        print('\n')


        # intigers pointing to the element equivalent to T and Dt
        Ti = int(T/dt)
        Dti = int(Dt/dt)

        # Check if provided T/dt, delta_t/dt and T/delta_t are all intigers
        if (round(Ti, 10))%1 != 0:  # round because of numerical error
            raise ValueError('Time_delay should be'
                             ' some multiple of dt from provided evaluation')
        if (round(Dti, 10))%1 != 0:  # round because of numerical error
            raise ValueError('Par. delta_t should be'
                             ' some multiple of dt from provided evaluation')
        if (round(T / Dt, 10))%1 != 0:  # round because of numerical error
            raise ValueError('Par. T should be'
                             ' some multiple of delta_t from provided evaluation')
        if (round(len(x0) / Dti, 10)) % 1 != 0:  # round because of numerical error
            raise ValueError('Number of points from provided evaluation should'
                             ' be dividable by Dt/dt')
        if t[-1] < time_units_apart / 2:
            print('Warning: t_max < time_units_apart / 2')

        indices_apart = int(time_units_apart / Dt)


        # generate time delay vector function coordinates of dimension td_dim
        l = int((len(x0) - Ti * (td_dim_max - 1)) / Dti)
        print('num. of datapoints: ', l)
        print('\n')

        TD = np.zeros((l, td_dim_max))
        for i in range(l):
            for j in range(td_dim_max):
                TD[i,j] = x0[i*Dti + j*Ti]


        def get_NN(d):
            # returns the indices of NN for dim. of delayed coord. space d, the distance squared
            # between these two points, and the distance squared between points in d+1 dim.
            # (returns [i, NN_i,  dist,  dist_of_dplus1] for each index i)

            # Initialize arrays
            NN_ind_arr = np.zeros((l, 4))
            TDd = TD[:,:d]
            TDdplus1 = TD[:,d]

            # Find NN in dim. of time delay d and save the  dist. of every pair
            for i, pi in enumerate(TDd):
                NNj = None
                c = 10**10  # initial  distance to NN

                for j, pj in enumerate(TDd):
                    if np.abs(i - j) > indices_apart:

                        if norm == 'sup':
                            c_new = np.linalg.norm(pi-pj, ord=np.inf)
                        elif norm == 'l2':
                            c_new = np.linalg.norm(pi-pj) ** 2
                        elif norm == 'l2/d':
                            c_new = np.linalg.norm(pi-pj) ** 2 / d
                        else:
                            raise ValueError("Parameter 'norm' should be one of ('max', 'l2', 'l2/d')")

                        if c_new < c:
                            c = c_new
                            NNj = j

                NN_ind_arr[i] = [i,NNj, c, 0]

            # Calculate the increase in dist. of every pair for dim. of time delay vector space d+1
            for NN_pair in NN_ind_arr:
                i, j = int(NN_pair[0]), int(NN_pair[1])
                dplus1dist = np.abs(TDdplus1[i] - TDdplus1[j])
                if dplus1dist > NN_pair[2]:
                    NN_pair[3] = dplus1dist
                else:
                    NN_pair[3] = NN_pair[2]

            return NN_ind_arr

        # Initialize return arrays
        E_arr = []
        E_star_arr = []
        d_arr = []

        # Calculate fnn for every d
        for d in np.arange(td_dim_min, td_dim_max):

            if self.verbose:
                print('Calculating for dim. {}/{}'.format(d, td_dim_max-1))

            NNd = get_NN(d)

            E_d = 0
            E_star_d = 0

            for i in range(l):

                a_d = NNd[i,3] / NNd[i,2]
                E_d += a_d
                E_star_d += NNd[i,3]

            d_arr.append(d)
            E_arr.append(E_d / l)
            E_star_arr.append(E_star_d / l)

        d_arr = d_arr[:-1]
        E1_arr = [E_arr[i+1]/E_arr[i] for i in range(len(d_arr))]
        E2_arr = [E_star_arr[i+1]/E_star_arr[i] for i in range(len(d_arr))]

        print(d_arr)
        print(E1_arr)
        print(E2_arr)


        self.FNN_v2.append({'d_arr' : np.asarray(d_arr),
                        'E1_arr' : np.asarray(E1_arr),
                        'E2_arr' : np.asarray(E2_arr),
                        'eval_path' : eval_path,
                        'T' : T,
                        'Dt' : Dt,
                        'time_units_apart' : time_units_apart,
                        'td_dim_span' : td_dim_span})


    def calculate_lyapunov_dim(self, lyap_arr):

        N = len(lyap_arr)

        sum = 0
        for i in range(N):
            s = sum + lyap_arr[i]
            if s >= 0:
                sum = s
            else:
                break

        d_lyap = i + 1/lyap_arr[i] * sum

        return d_lyap


# MODEL 2 FROM LORENZ 2005 (= MODEL FROM LORENZ, EMANUEL 1998)
class sur_L05_2: # sur_L05_2
    # Class based on the surrogate SINDy Model 2 presented in a paper by Lorenz: 'Designing Chaotic Models' (2005).
    # For K = 1 this further simplifies to Model 1


    def __init__(self, N, K, F, model_path, integrator_type = 'LSODA'):

        # ----- Intrinsic Model Parameters -----
        self.N = N
        self.K = K
        self.F = F

        with open(model_path, 'rb') as inp:

            # Results of GridSearch (SINDy)
            res = pickle.load(inp)

            # Save best estimator as an intrinsic model of this object
            self.model = res['best_est']

            # Save other Search parameters as a dict
            self.props = {}
            for key, val in res.items():
                if key != 'best_est':
                    self.props[key] = val

            del inp


        # If the integration doesnt reach the end i.e. the trajectory is divergent, self.divergent is set to true.
        # In this case the properties cannot be evaluated.
        self.divergent = None

        # Integrator, 'LSODA' by default (other possible choices: 'RK45' 'RK23', 'DOP853', 'BDF')
        self.integrator = integrator_type

        # Terms in equation resented in practical form. Can be evaluated by calling .get_terms()
        self.terms = None
        self.get_terms()

        # To be evaluated ( by calling .evaluate(...) ). Each item in evaluation_path is a unique string
        # path pointing to a solve_ivp object (in a sense of a relative file path)
        self.evaluation_paths = []

        # Properties of an evaluated trajectory (min,max,avr,std). Evaluated with self.calculate_eval_props
        self.eval_props = []

        # To be evaulated with calculate_lyapunov_spectrum. Each element is a dict with keys 'init_cond',
        # 'renormalization time', 'max iteration number', 'delta_start', 'average of log10(delta_stop)',
        # 'lyapunov exponents', 'lyapunov std'.
        self.lyapunov_spectrum = []


        # Power Spectrum Densities. Evaluated with self.calculate_PSD. Each element in a list is a dict. with keys
        # 'freqs', 'psd', 'window_length', 'eval_path'
        self.PSD = []

        # Periodic moments. Evaluated with self.calculate_periodic_moments. Each element is a dict. with keys k_arr',
        # 'mom_arr', eval_path'
        self.moments = []

        # Mutual information function. Evaluated with self.calcuate_mutual_information. Each element is a dict with
        # keys 'time_delays', 'mutual_info', 'partition_num' and 'eval_path'
        self.mutual_information = []

        # Error on coefficients. To be evaluated with self.calculate_coef_error() -> it gets a dict with three keys:
        # 'org', 'sur' and 'dif', where each value is a list of self.N dicts., containing term (feature) names
        # and their respective values
        self.coef_error = None

        # If true, print extra info such as elapsed time when integrating etc.
        self.verbose = False

        # To be evaluated with calculate_covariance_and_accuracy. Each element is a dict. with keys 'm', 'a',
        # 'w_a', 'cov', 'corr
        self.stats = []

        self.print_parameters()


    def print_parameters(self):
        print('\n#########################################\n')
        print('N: ', self.N)
        print('K: ', self.K)
        print('F: ', self.F)
        print('Integrator:', self.integrator)
        print('Evaluation paths:')
        for path in self.evaluation_paths:
            print('                  {}'.format(path))
        print('Lyapunov spectrum:')
        for i in self.lyapunov_spectrum:
            print(i)
        print('PSD:')
        for i in self.PSD:
            print(i)
        print('Periodic moments:')
        for i in self.moments:
            print(i)
        self.stringify_ODE_OG()
        self.stringify_ODE_SUR()
        for key, val in self.props.items():
            if key != 'cv_results' and key != 'x':
                print(key, ' : ', val)
        print('\n#########################################\n')


    def evaluate(self, init_cond, dt, t_stop, save=True):
        # The core function of this class, called to evaluate the integral of model() function.
        # The result is stored in self.evaluation

        # ----- Integration Parameters -----
        init_cond = np.asarray(init_cond)
        t_eval = np.arange(0, t_stop, dt)

        # Seed for reproducibility
        np.random.seed(1000)

        # Integrator keywords for solve_ivp
        integrator_keywords = {}
        integrator_keywords['method'] = self.integrator
        integrator_keywords['rtol'] = 1e-12
        integrator_keywords['atol'] = 1e-12
        integrator_keywords['min_step'] = 1e-7

        # Evaluation
        if self.verbose:
            print('\nEvolve:')

        x = self.model.simulate(init_cond, t_eval, integrator_kws=integrator_keywords)
        evaluation = ev(x.T, t_eval)

        # Modify object's property if thetrajectory is divergent
        if len(x) != len(t_eval):
            self.divergent = True
            print('\n')
            print('Trajectory is divergent')
            return None
        else:
            self.divergent = False
            print('\n')
            print("Trajectory didn't diverge")

            if self.verbose:
                print('\n')

            # Save the evaluation as a separate object with a unique name
            parameters = {'N':self.N,
                          'K':self.K,
                          'F':self.F,
                          'dt':dt,
                          't1':t_stop}
            name = 'data/sur_eval_{}_{}_{}.pkl'.format(parameters,get_unique_name(init_cond.tolist()),str(time.time()))
            if save:
                self.evaluation_paths.append(name)
                with open(self.evaluation_paths[-1], 'wb') as outp:
                    pickle.dump(evaluation, outp, pickle.HIGHEST_PROTOCOL)
                    del outp

            final_cond=evaluation.y.T[-1]
            del evaluation

            return name, final_cond


    def calculate_eval_props(self, eval_path):

        if self.divergent == True:
            raise ValueError('Thise model is divergent')

        with open(eval_path, 'rb') as L05_2_inp_eval:
            ev = pickle.load(L05_2_inp_eval)

        r = {'eval_path' : eval_path,
            'mi' : np.min(ev.y.T),
            'ma' : np.max(ev.y.T),
            'av' : np.average(ev.y.T),
            'st' : np.std(ev.y.T),
            }

        if self.verbose == True:
            print(r)

        self.eval_props.append(r)

        return r


    def get_terms(self):
        # Returns 'a matrix' f(X) (RHS of ODE) in a form that might be convenient for
        # comparison and error estimation of ODE discovered by SINDy. The output is based
        # on classes' quad_term, lin_term,...

        def group_lin_terms(arr):
            # returns an ordered, grouped list of terms of order 1

            new_arr = []

            for i in range(self.N):

                term = lin_term(0, i, self.N)

                for term_in_arr in arr:
                    if term_in_arr.n == term.n:
                        term += term_in_arr

                if term.amp != 0:
                    new_arr.append(term)

            return new_arr

        def group_quad_terms(arr):
            # returns an ordered, grouped list of terms of order 2

            new_arr = []

            for i in range(self.N):
                for j in range(i, self.N):

                    term = quad_term(0,i,j,self.N)

                    for term_in_arr in arr:
                        if term_in_arr.n1 == term.n1 and term_in_arr.n2 == term.n2:
                            term += term_in_arr

                    if term.amp != 0:
                        new_arr.append(term)

            return new_arr


        def br_str(X1_amps, X2_amps, K, n):
            # The bracket formula; eq. (10) and (9) in Lorenz's 2005 paper. Return of
            # this function is of analytic nature

            amplitude = X1_amps[n] * X2_amps[n]
            terms = []

            if K % 2 == 1:
                J = int((K - 1) / 2)

                for i in range(-J, J + 1):
                    for j in range(-J, J + 1):
                        term1 = quad_term((-1) * amplitude / K ** 2, n - 2 * K - i, n - K - j, self.N)
                        term2 = quad_term(  1  * amplitude / K ** 2, n - K + j - i, n + K + j, self.N)

                        terms.append(term1)
                        terms.append(term2)


            elif K % 2 == 0:
                J = int(K / 2)

                for i in range(-J, J + 1):
                    for j in range(-J, J + 1):
                        term1 = quad_term((-1) * amplitude / K ** 2, n - 2 * K - i, n - K - j, self.N)
                        term2 = quad_term(  1  * amplitude / K ** 2, n - K + j - i, n + K + j, self.N)
                        if abs(i) == J:
                            term1.amp /= 2
                            term2.amp /= 2
                        if abs(j) == J:
                            term1.amp /= 2
                            term2.amp /= 2

                        terms.append(term1)
                        terms.append(term2)


            return terms

        # Compute the coefficients to start with
        Z = np.ones(self.N)

        def Z_dot_n_str(n):
            # Lorenz 2005, eq. (15)
            print('Calculating analytical terms for component {}/{}'.format(n+1, self.N))

            quadratic_terms = group_quad_terms(br_str(Z, Z, self.K, n))
            linear_terms = group_lin_terms([lin_term(- Z[n], n, self.N)])
            constant_terms = [cons_term(self.F)]

            return constant_terms + linear_terms + quadratic_terms


        self.terms = [Z_dot_n_str(i) for i in range(self.N)]
        print('\n')


    def stringify_ODE_OG(self):
        # A wrapper for get_terms() method when one wishes to print out
        # the form of the ODE

        if self.terms == None:
            self.get_terms()

        print('ODE (OG):')
        for i in range(self.N):
            print("(x{})' =".format(i), *self.terms[i])


    def stringify_ODE_SUR(self):
        # A wrapper for get_terms() method when one wishes to print out
        # the form of the ODE

        print('ODE (SUR):')
        self.model.print()


    def calculate_coef_error(self):
        # compare coefficients of original and surrogate model

        # list of dicts containing term names and the complementary errors
        # between og. and sur. model
        differences_terms = []
        originl_terms = []
        surrogate_terms = []


        # list of string lhs of surrogate model
        sur_model_rhs = self.model.equations(precision=5)

        # list of string lhs of original model
        org_model_rhs_0 = self.terms # self.terms should be != None on this point
        org_model_rhs = []
        for i in range(self.N):
            s = ''
            for j in org_model_rhs_0[i]:
                s += str(j) + ' '
            org_model_rhs.append(s)

        # for every component of ODE
        for eq_num in range(self.N):
            sur_terms = {}
            org_terms = {}

            sur_pattern = sur_model_rhs[eq_num] + ' +' # we ad ' +' at the end for easier re search
            org_pattern = org_model_rhs[eq_num] + '+'  # we ad '+' at the end for easier re search

            # constant terms
            sur_const = re.search(r"(-?[0-9]*\.?[0-9]*) 1", sur_pattern)
            org_const = re.search(r"(-?[0-9]*\.?[0-9]*) 1", org_pattern)
            if sur_const != None:
                sur_terms['1'] = float(sur_const.groups()[0])
            if org_const != None:
                org_terms['1'] = float(org_const.groups()[0])

            # linear terms
            for i in range(self.N):
                sur_lin = re.search(r"\+ (-?[0-9]*\.?[0-9]*) x{} \+".format(i), sur_pattern)
                org_lin = re.search(r"\+ (-?[0-9]*\.?[0-9]*) x{} \+".format(i), org_pattern)
                if sur_lin != None:
                    sur_terms['x{}'.format(i)] = float(sur_lin.groups()[0])
                if org_lin != None:
                    org_terms['x{}'.format(i)] = float(org_lin.groups()[0])

            # quadratic terms
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        sur_qua = re.search(r"\+ (-?[0-9]*\.?[0-9]*) x{}\^2 \+".format(i), sur_pattern)
                        org_qua = re.search(r"\+ (-?[0-9]*\.?[0-9]*) x{}\^2 \+".format(i), org_pattern)
                        if sur_qua != None:
                            sur_terms['x{}^2'.format(i)] = float(sur_qua.groups()[0])
                        if org_qua != None:
                            org_terms['x{}^2'.format(i)] = float(org_qua.groups()[0])
                    else:
                        sur_qua = re.search(r"\+ (-?[0-9]*\.?[0-9]*) x{} x{} \+".format(i,j), sur_pattern)
                        org_qua = re.search(r"\+ (-?[0-9]*\.?[0-9]*) x{} x{} \+".format(i,j), org_pattern)
                        if sur_qua != None:
                            sur_terms['x{} x{}'.format(i,j)] = float(sur_qua.groups()[0])
                        if org_qua != None:
                            org_terms['x{} x{}'.format(i,j)] = float(org_qua.groups()[0])


            dif_terms = copy.deepcopy(sur_terms)
            for org_key, org_val in org_terms.items():
                if org_key in dif_terms.keys():
                    dif_terms[org_key] -= org_val
                else:
                    dif_terms[org_key] = -org_val

            differences_terms.append(dif_terms)
            originl_terms.append(org_terms)
            surrogate_terms.append(sur_terms)


        res = {'dif':differences_terms,
               'org':originl_terms,
               'sur':surrogate_terms}

        self.coef_error = res

        return res


    def calculate_lyapunov_spectrum(self, init_cond, T, max_iter_num=50, delta_start=10 ** (-8), N=None):
        # Calculate lyapunov spectrum. Parameter T should be chosen similar to ALREADY CALCULATED
        # average 'stopping time' from calculate_lyapunov_max i.e. it is the time in which the trajectories will
        # be normalized (Gram-Schmidt). Parameter max_iter_num is the number of times we estimate the renormalization factors
        # needed to calculate Lyapunov exponents. N is the number of lyapunov exponents calculated (mast be smaller then selfN

        if self.divergent == True:
            raise ValueError('Thise model is divergent')

        if N > self.N:
            raise ValueError('N must be smaller or equal than the dimension of the sysem (self.N)')
        if N == None:
            N = self.N

        # Initialize some parameters
        init_cond = np.asarray(init_cond)
        t_span = (0,  T)
        all_betas = np.zeros((max_iter_num, N))
        deltas_stop = np.zeros(max_iter_num)  # log deviations at which renormalization happens - ideally this
                                              # number should be around 10^(-2) (numerical recommendations)

        print('\n####################################################\n')
        print('Calculating Lyapunov spectrum:')
        print('Stopping time (T): ', T, '(=time in which the the trajectories will be normalized (Gram-Schmidt))')
        print('Number of iterations: ', max_iter_num)
        print('Number of lyap. exps. generated: ', N)


        # Seed for reproducibility
        np.random.seed(1000)

        # Integrator keywords for solve_ivp
        integrator_keywords = {}
        integrator_keywords['method'] = self.integrator
        integrator_keywords['rtol'] = 1e-12
        integrator_keywords['atol'] = 1e-12
        integrator_keywords['min_step'] = 1e-7

        # We start with an N+1 different initial conditions; og. init. cond and N init.
        # conds. perturbed with delta_start
        Z0 = np.asarray([init_cond for i in range(N + 1)])
        for i in range(1, N + 1):
            Z0[i, i - 1] += delta_start

        iter_num = 0
        while iter_num < max_iter_num:

            iter_num += 1
            if self.verbose:
                print('Iteration: ', iter_num)

            # Perform integrations to time T
            integ = [self.model.simulate(Z0_i, np.linspace(0,T,100), integrator_kws=integrator_keywords)[-1]
                    for Z0_i in Z0]

            # Calculate displacement sizes and store them as tuples with their correspondent vectors. Afterwards sort
            # the tuples from the one with biggest to the smallest displacement. Note that sorting is not neccessary
            # here, we only do it due to better numerical stability of the algorithm
            displ = [[np.linalg.norm(integ_i - integ[0]), integ_i - integ[0]] for integ_i in integ[1:]]
            displ = sorted(displ, key=lambda x: x[0], reverse=True)

            renormalized_vector_norms = np.zeros(N)  # normalization factors
            e = np.zeros((N, self.N))  # ortonormalized_basis

            # Gram - Schmidt orthonormalizing
            for i in range(N):
                e[i,:] = displ[i][1]
                for j in range(i):
                    e[i,:] -= np.dot(displ[i][1], e[j,:]) * e[j,:]

                renormalized_vector_norms[i] = np.linalg.norm(e[i])
                e[i,:] = e[i,:] / renormalized_vector_norms[i]

            # Save renormalization factors
            all_betas[iter_num-1,:] = sorted(renormalized_vector_norms / delta_start, reverse=True)
            deltas_stop = all_betas[iter_num-1,0] * delta_start  # save the biggest renorm. factor

            # Re-initiate new initial conditions for the next loop
            Z0[0] = integ[0]
            for i in range(N):
                Z0[i+1] = integ[0] + e[i,:] * delta_start

        lambdas = [1.0/(T*max_iter_num) * np.sum([np.log(all_betas[j,i]) for j in range(max_iter_num)]) for i in range(N)]
        lambdas_std = [np.std([1.0/(T)*np.log(all_betas[j,i]) for j in range(max_iter_num)]) for i in range(N)]

        results = {'init_cond' : init_cond,
                   'renormalization time' : T,
                   'max iteration number' : max_iter_num,
                   'delta_start' : delta_start,
                   'average of log10(delta_stop)' : np.average(np.log10(deltas_stop)),
                   'lyapunov exponents': lambdas,
                   'lyapunov std': lambdas_std}

        self.lyapunov_spectrum.append(results)
        return results


    def calculate_PSD(self, eval_path, nperseg=1e5, window='hanning'):
        # Calculate Power Spectral Density for specified evaluated trajectory. Only one (but long) trajectory is
        # needed if we suppose the attractor has an invariant natural measure. Parameters window is an array of
        # consecutive integers (< trajectory length) that determines the window for Welch's method. It is correlated
        # to the resolution of the result.
        # Results are saved under

        if self.divergent == True:
            raise ValueError('Thise model is divergent')

        print('Calculating PSD ... eval_path={}'.format(eval_path))

        with open(eval_path, 'rb') as L05_2_inp_eval:
            ev = pickle.load(L05_2_inp_eval)
            del L05_2_inp_eval

        # first component centered around zero
        x0 = ev.y.T[:,0]- np.average(ev.y.T[:,0])

        freqs, psd = signal.welch(x=x0, fs=(ev.t[1]-ev.t[0])**(-1), nperseg=int(nperseg), window=window)

        self.PSD.append({'dt':ev.t[1]-ev.t[0],
                         't_stop':ev.t[-1],
                         'freqs':freqs,
                         'psd':psd,
                         'window':window,
                         'npesreg':nperseg,
                         'eval_path':eval_path})

        print('Done')


    def calculate_periodic_moments(self, eval_path, k_arr):
        # Return the time average of a function cos(kx) for an array of k's

        if self.divergent == True:
            raise ValueError('Thise model is divergent')

        with open(eval_path, 'rb') as L05_2_inp_eval:
            ev = pickle.load(L05_2_inp_eval)
            del L05_2_inp_eval

        dt = ev.t[1] - ev.t[0]
        T = ev.t[-1]
        f_Z_integs = np.zeros(len(k_arr))

        f = lambda k, x: np.cos(k*x)
        #f = lambda k, x: np.exp(1j*k * x)

        # for each k average the value of f over time and over all trajectories
        for i in range(len(k_arr)):
            if self.verbose:
                print('Calculating periodic moment for k={} ({}/{})'.format(k_arr[i], i+1, len(k_arr)))
            for j in range(self.N):
                f_Z_integs[i] += 1/(T*self.N) * np.trapz(y=f(k_arr[i], ev.y.T[:, j]), dx=dt)

        self.moments.append({'k_arr' : k_arr,
                             'mom_arr' : f_Z_integs,
                             'eval_path' : eval_path})


    def calculate_mutual_information(self, eval_path, time_delays, partition_num=50):
        # Mutual information as a function of time delay. Values in an array time_delays should be
        # some multiples of dt from provided evaluation. A scalar function that is used in the process
        # is the average of all components at one time instance

        if self.divergent == True:
            raise ValueError('Thise model is divergent')

        with open(eval_path, 'rb') as L05_2_inp_eval:
            ev = pickle.load(L05_2_inp_eval)

            dt = ev.t[1] - ev.t[0]
            t = ev.t
            x0 = np.sum(ev.y.T, axis=-1) / self.N # average of all components at a time instant

            del L05_2_inp_eval, ev

        print('\nCalculating Mutual Information:')
        print('dt from evaluation: ', dt)
        print('max time from evaluation: ', t[-1])
        print('partition num.: ', partition_num)
        print('\n')

        # Check if provided time delays are multiples of dt
        for T in time_delays:
            if (round(T/dt, 10))%1 != 0:  # round because of numerical error
                raise ValueError('Values in an array time_delays should be'
                                 ' some multiples of dt from provided evaluation')

        partitions_borders = np.linspace(np.min(x0), np.max(x0), partition_num + 1, endpoint=True)
        mut_inf = np.zeros(len(time_delays))

        def bin(x, y):
            # Return the mutual partition of two numbers
            xi, yi = -1, -1
            for i in range(partition_num):
                if partitions_borders[i + 1] >= x:
                    xi = i
                    break
            for i in range(partition_num):
                if partitions_borders[i + 1] >= y:
                    yi = i
                    break
            if not (xi==-1 and yi==-1):
                return xi,yi
            else:
                raise ValueError('x or y not in any partition')

        for i,T in enumerate(time_delays):
            if self.verbose:
                print('Calculating for time delay {} ({}/{})'.format(round(T,10), i+1, len(time_delays)))


            P_mn = np.zeros((partition_num, partition_num)) # Num. of events when x(t) is in bin m and x(t+T) is in bin n
            P_m = np.zeros(partition_num)  # Num. of events when x(t) is in bin n
            Ti = int(T/dt) # index that will point to time t + T

            for j in range(len(x0)-Ti):
                m, n = bin(x0[j], x0[j+Ti])
                P_m[m] += 1
                P_mn[m,n] += 1

            # Num. of events to probabilities
            P_m = P_m / np.sum(P_m)
            P_mn = P_mn / np.sum(P_mn)

            for m in range(partition_num):
                if P_m[m] != 0:
                    mut_inf[i] += - 2*P_m[m] * np.log(P_m[m])
                for n in range(partition_num):
                    if P_mn[m,n] != 0:  # we take out zeros otherwise log(0) = -inf and PlogP produces a nan
                        mut_inf[i] += P_mn[m,n] * np.log(P_mn[m,n])

        results = {'time_delays' : time_delays,
                   'mutual_info' : mut_inf,
                   'partition_num' : partition_num,
                   'eval_path' : eval_path
                   }

        self.mutual_information.append(results)

        return results


    def calculate_lyapunov_dim(self, lyap_arr):

        if self.divergent == True:
            raise ValueError('Thise model is divergent')

        N = len(lyap_arr)

        sum = 0
        for i in range(N):
            s = sum + lyap_arr[i]
            if s >= 0:
                sum = s
            else:
                break

        d_lyap = i + 1/lyap_arr[i] * sum

        return d_lyap


    def calculate_covariance_and_accuracy(self, m, a=0.01):
        # Returns the covariance and correlation matrix of parameters of the model and the accuracy vector of weights
        # a - the std of noise that is added to original x
        # m - number of perturbations

        if self.divergent == True:
            raise ValueError('Thise model is divergent')

        # numb. of features
        n_feat = int((self.N**2 + 3*self.N) / 2.0) + 1

        # x on which og regression was performed
        x = self.props['x']

        # vector of og. coefficients
        w_og = self.model.optimizer.coef_.flatten()

        w = np.zeros((m,n_feat*self.N))
        w_a = np.zeros(n_feat*self.N)

        for i in range(m):
            if self.verbose:
                print('Calculating fit ',i+1,'/',m)

            # apply noise of size a on x
            noise = np.random.normal(loc=0, scale=a, size=x.shape)
            x_pert = x + noise

            # fit model with perturbed x
            model_pert = copy.deepcopy(self.model)
            model_pert.fit(x_pert)

            # calculate relative vector of coefficients
            w_i = model_pert.optimizer.coef_.flatten() - w_og
            w[i] = w_i
            w_a += w_i


            # delete object to save space
            del model_pert

        # compute accuracy
        w_a = w_a / (m)

        # compute covariance matrix
        cov = np.cov(w.T)

        # compute correlation matrix
        corr = np.corrcoef(w.T)

        self.stats.append({'m':m,
                           'a':a,
                           'w_a':w_a,
                           'cov':cov,
                           'corr':corr})






























# # A GENERAL MODEL 3 FROM LORENZ 2005
# class LM05:
#     # Class based on the Model 3 presented in a paper by Lorenz: 'Designing Chaotic Models' (2005).
#     # For I = 1 this model simplifies to Model 2. For K = 1 this further simplifies to Model 1
#
#
#     def __init__(self, N, K, I, b, c, F, integrator_type = 'LSODA'):
#
#         # ----- Intrinsic Model Parameters -----
#         self.N = N
#         self.K = K
#         self.I = I
#         self.b = b
#         self.c = c
#         self.F = F
#         self.alpha = (3 * (self.I ** 2) + 3) / (2 * (self.I ** 3) + 4 * self.I)
#         self.beta = (2 * (self.I ** 2) + 1) / ((self.I ** 4) + 2 * (self.I ** 2))
#
#         # Integrator, 'LSODA' by default (other possible choices: 'RK45' 'RK23', 'DOP853', 'BDF')
#         self.integrator = integrator_type
#
#         # Terms in equation resented in practical form. Can be evaluated by calling .get_terms()
#         self.terms = None
#
#         # To be evaluated ( by calling .evaluate(...) ). Each item in evaluation_path is a unique string
#         # path pointing to a solve_ivp object (in a sense of a relative file path)
#         self.evaluation_paths = []
#
#         # To be evaluated ( by calling .get_lyapunov_max(...) ). When evaluated, a dictionary is created
#         # with keys: 'average', 'all generated', 'error growth', 'delta start', 'delta stop'. Each item in
#         # lyapunov_paths is a unique string that represents a relative path to an object where the result is stored
#         self.lyapunov_max_paths = []
#         self.lyapunov_spectrum = []

#         # Power Spectrum Densities. Evaluated with self.calculate_PSD. Each element in a list is a dict. with keys
#         # 'freqs', 'psd', 'window_length', 'eval_path'
#         self.PSD = []
#
#         # Periodic moments. Evaluated with self.calculate_periodic_moments. Each element is a dict. with keys k_arr',
#         # 'mom_arr', eval_path'
#         self.moments = []
#
#         # Mutual information function. Evaluated with self.calcuate_mutual_information. Each element is a dict with
#         # keys 'time_delays', 'mutual_info', 'partition_num' and 'eval_path'
#         self.mutual_information = []
#
#         # False Nearest Neighbour function. Evaluated with self.calculate_FNN. Each element is a dict with keys
#         # 'd_arr', 'fnn', 'thr_choice', 'eval_path', 'T', 'Dt', 'time_units_apart', 'td_dim_span'
#         self.FNN_v1 = []
#
#         # False Nearest Neighbour function. Evaluated with self.calculate_FNN. Each element is a dict with keys
#         # 'd_arr', 'E1_arr', 'E2_arr', 'eval_path', 'T', 'Dt', 'time_units_apart', 'td_dim_span'
#         self.FNN_v2 = []
#
#         # Print object parameters
#         self.print_parameters()
#
#         # If true, print extra info such as elapsed time when integrating etc.
#         self.verbose = False
#
#
#     def print_parameters(self):
#         print('\n#########################################\n')
#         print('N: ', self.N)
#         print('K: ', self.K)
#         print('I: ', self.I)
#         print('alpha: ', self.alpha)
#         print('beta: ', self.beta)
#         print('b: ', self.b)
#         print('c: ', self.c)
#         print('F: ', self.F)
#         print('Integrator:', self.integrator)
#         print('Evaluation paths:')
#         for path in self.evaluation_paths:
#             print('                  {}'.format(path))
#         print('Lyapunov max paths:')
#         for path in self.lyapunov_max_paths:
#             print('                {}'.format(path))
#         print('Lyapunov spectrum:')
#         for i in self.lyapunov_spectrum:
#             print(i)
#         print('PSD:')
#         for i in self.PSD:
#             print(i)
#         print('Periodic moments:')
#         for i in self.moments:
#             print(i)
#         print('FNNv1:')
#         for i in self.FNN_v1:
#             print(i)
#         print('FNNv2:')
#         for i in self.FNN_v1:
#             print(i)
#         print('\n#########################################\n')
#
#
#     def mod(self, n):
#         # just a cleaner notation for modulo operation
#
#         return n % self.N
#
#
#     def XnYn(self, n, Z):
#         # Lorenz 2005, eq. (13a, 13b); calculate Xn and Yn from Zn
#
#         Xn = 0
#
#         for i in range(-self.I, self.I+1):
#             if abs(i) == self.I:
#                 Xn += (self.alpha - self.beta * abs(i)) * Z[self.mod(n + i)] / 2
#             else:
#                 Xn += (self.alpha - self.beta * abs(i)) * Z[self.mod(n + i)]
#
#         return Xn, Z[self.mod(n)] - Xn
#
#
#     def br(self, X1, X2, K, n):
#         # the bracket formula; eq. (10) and (9) in Lorenz's 2005 paper
#
#         sum = 0
#
#         if K % 2 == 1:
#             J = int((K - 1) / 2)
#
#             for i in range(-J,J+1):
#                 for j in range(-J,J+1):
#                     sum1 = X1[self.mod(n - 2*K - i)] * X2[self.mod(n - K - j)]
#                     sum2 = X1[self.mod(n - K + j - i)] * X2[self.mod(n + K + j)]
#                     sum += sum2 - sum1
#
#         elif K % 2 == 0:
#             J = int(K / 2)
#
#             for i in range(-J, J + 1):
#                 for j in range(-J, J + 1):
#                     sum1 = X1[self.mod(n - 2 * K - i)] * X2[self.mod(n - K - j)]
#                     sum2 = X1[self.mod(n - K + j - i)] * X2[self.mod(n + K + j)]
#                     sum3 = sum2 - sum1
#                     if abs(i) == J:
#                         sum3 /= 2
#                     if abs(j) == J:
#                         sum3 /= 2
#                     sum += sum3
#
#         return sum / (K ** 2)
#
#
#     def model(self, t, Z):
#         # Representing f(Z(t)) in dZ(t)/dt = f(Z(t))
#
#         # Logging integration time
#         if self.verbose:
#             print('t = ', round(t,5))
#
#         X, Y = self.XnYn(np.arange(0, self.N), Z)
#
#         def Z_dot_n(n):
#             # Lorenz 2005, eq. (15)
#
#             return self.br(X, X, self.K, n) \
#             + (self.b ** 2) * self.br(Y, Y, 1, n) \
#             + self.c * self.br(Y, X, 1, n) \
#             - X[n] \
#             - self.b * Y[n] \
#             +  self.F
#
#         return Z_dot_n(np.arange(0, self.N))
#
#
#     def evaluate(self, init_cond, dt, t_stop):
#         # The core function of this class, called to evaluate the integral of model() function.
#         # The result is stored in self.evaluation
#
#         # ----- Integration Parameters -----
#         init_cond = np.asarray(init_cond)
#         t_eval = np.arange(0, t_stop, dt)
#
#         # Seed for reproducibility
#         np.random.seed(1000)
#
#         # Integrator keywords for solve_ivp
#         integrator_keywords = {}
#         integrator_keywords['method'] = self.integrator
#         integrator_keywords['rtol'] = 1e-12
#         integrator_keywords['atol'] = 1e-12
#
#         # Evaluation
#         evaluation = integrate.solve_ivp(fun = self.model, t_span = (t_eval[0], t_eval[-1]),
#                                          y0 = init_cond, t_eval = t_eval, **integrator_keywords)
#
#         # Save the evaluation as a separate object with a unique name
#         parameters = [self.N, self.K, self.I, self.b, self.c, self.F, dt, t_stop]
#         self.evaluation_paths.append('data/eval_{}_{}.pkl'.format(parameters,get_unique_name(init_cond.tolist())))
#         with open(self.evaluation_paths[-1], 'wb') as outp:
#             pickle.dump(evaluation, outp, pickle.HIGHEST_PROTOCOL)
#             del outp
#         del evaluation
#
#
#     def calculate_lyapunov_max(self, eval_path, delta_start=10 ** (-8), delta_stop=10 ** (0)):
#         # Calculate maximal Lyapunov exponent. This method should be used to estimate error growth time that
#         # can later be used in calculation of the whole lyapunov spectrum (because the later method requires a fixed
#         # stopping time). Method calculate_lyapunov_max is also useful if one is calculating lyap. exp. of a higher
#         # dimensional system that is computationally slow to integrate (so that we compute only one lyap. exp., not
#         # the whole spectrum). But, because it requires (cuz of the solve_ivp stopping condition) a relatively big
#         # delta_stop it is not very precise
#
#         # Load an already calculated trajectory
#         with open(eval_path, 'rb') as LM05_inp_eval:
#             evaluation = pickle.load(LM05_inp_eval)
#             del LM05_inp_eval
#
#         # Time parameters
#         t_train_span = (evaluation.t[0], evaluation.t[-1])
#         dt = evaluation.t[1] - evaluation.t[0]
#
#         # Parameters for calculating Lyapunov exp.
#         total_t = 0
#         times = []
#         normalization_factors = []
#         error_growth = []
#         lyap_enum = 0
#
#         def time_to_normalize(t, Z):
#             # A function to be passed to solve_ivp(events= )
#             # When the error (initially of norm delta_min grows to the size of delta_max) this function hits zero
#             # and the integration is to be stopped
#
#             # perturbed vector = original trajectory - perturbed trajectory
#             pert_vec = np.asarray(evaluation.y.T[int(round((total_t + t) / dt))]) - np.asarray(Z)
#             error = np.linalg.norm(pert_vec)
#             error_growth[lyap_enum].append([t, error])
#
#             if self.verbose:
#                 print('t: ', t, 'error: ', np.linalg.norm(pert_vec), 'delta_stop = ', delta_stop)
#
#             return error - delta_stop
#
#         # Seed for reproducibility
#         np.random.seed(1000)
#
#         # Integrator keywords for solve_ivp
#         integrator_keywords = {}
#         integrator_keywords['method'] = self.integrator
#         integrator_keywords['rtol'] = 1e-12
#         integrator_keywords['atol'] = 1e-12
#         time_to_normalize.terminal = True  # Stop integration when the event happen
#
#         # We start with an init cond. + perturbation vector of norm
#         # delta_min, equally big on all components
#         Z0 = evaluation.y.T[0] + np.ones(self.N) * np.sqrt(delta_start ** 2 / self.N)
#
#         while True:
#             try:
#                 error_growth.append([])
#
#                 integrate_before_normalize = integrate.solve_ivp(self.model,
#                                                                  t_train_span, Z0,
#                                                                  t_eval=evaluation.t,
#                                                                  events=time_to_normalize,
#                                                                  **integrator_keywords)
#
#                 # Time at which the event occurred (size of an error reached delta_stop)
#                 stopping_time = integrate_before_normalize.t_events[0][0]
#                 total_t += stopping_time
#                 times.append(stopping_time)
#
#                 # Caluculate the displacement (error) vector
#                 displacement = (integrate_before_normalize.y_events[0][0] - evaluation.y.T[int(round(total_t / dt))])
#                 beta = np.linalg.norm(displacement) / delta_start  # Normalization factor
#                 normalization_factors.append(beta)
#
#                 # Define a new perturbed vector as the original solution at
#                 # current time plus normalized perturbation vector
#                 Z0 = evaluation.y.T[int(round(total_t / dt))] + displacement * 1 / beta
#
#                 lyap_enum += 1
#
#             except IndexError:
#                 print('The end of evaluated trajectory reached')
#                 break
#
#         # Calculate all generated Lyapunov coefficients
#         lambdas = 1 / np.asarray(times) * np.log(np.asarray(normalization_factors))
#
#         # Dict. with output parameters
#         lyapunov_max = {'average': np.average(lambdas),
#                         'all generated': lambdas,
#                         'error growth': [np.asarray(i) for i in error_growth[:-1]],  # The last one didn't reach the end
#                         'stopping times': times,
#                         'delta start': delta_start,
#                         'delta stop': delta_stop,
#                         'corresponding evaluation path': eval_path}
#
#         # Save the evaluation as a separate object with a unique name
#         parameters = [self.N, self.K, self.I, self.b, self.c, self.F, dt,
#                       evaluation.t[-1], delta_start, delta_stop]
#         self.lyapunov_max_paths.append(
#             'data/lyap_{}_{}.pkl'.format(parameters, get_unique_name(evaluation.y.T[0].tolist())))
#         with open(self.lyapunov_max_paths[-1], 'wb') as outp:
#             pickle.dump(lyapunov_max, outp, pickle.HIGHEST_PROTOCOL)
#             del outp
#
#         del evaluation, lyapunov_max
#
#
#     def calculate_lyapunov_spectrum(self, init_cond, T, max_iter_num = 50, delta_start =10 ** (-8)):
#         # Calculate lyapunov spectrum. Parameter T should be chosen similar to ALREADY CALCULATED
#         # average 'stopping time' from calculate_lyapunov_max i.e. it is the time in which the trajectories will
#         # be normalized (Gram-Schmidt). Parameter max_iter_num is the number of times we estimate the renormalization factors
#         # needed to calculate Lyapunov exponents
#
#
#         # Initialize some parameters
#         init_cond = np.asarray(init_cond)
#         t_span = (0,  T)
#         all_betas = np.zeros((max_iter_num, self.N))
#         deltas_stop = np.zeros(max_iter_num)  # log deviations at which renormalization happens - ideally this
#                                               # number should be around 10^(-2) (numerical recommendations)
#
#         print('\n####################################################\n')
#         print('Calculating Lyapunov spectrum:')
#         print('Stopping time (T): ', T, '(=time in which the the trajectories will be normalized (Gram-Schmidt))')
#         print('Number of iterations: ', max_iter_num)
#
#
#         # Seed for reproducibility
#         np.random.seed(1000)
#
#         # Integrator keywords for solve_ivp
#         integrator_keywords = {}
#         integrator_keywords['method'] = self.integrator
#         integrator_keywords['rtol'] = 1e-12
#         integrator_keywords['atol'] = 1e-12
#
#         # We start with an N+1 different initial conditions; og. init. cond and N init.
#         # conds. perturbed with delta_start
#         Z0 = np.asarray([init_cond for i in range(self.N + 1)])
#         for i in range(1, self.N + 1):
#             Z0[i, i - 1] += delta_start
#
#         iter_num = 0
#         while iter_num < max_iter_num:
#
#             iter_num += 1
#             if self.verbose:
#                 print('Iteration: ', iter_num)
#
#             # Perform integrations to time T
#             integ = [integrate.solve_ivp(self.model,
#                                               t_span,
#                                               Z0_i,
#                                               **integrator_keywords).y.T[-1]
#                     for Z0_i in Z0]
#
#             # Calculate displacement sizes and store them as tuples with their correspondent vectors. Afterwards sort
#             # the tuples from the one with biggest to the smallest displacement. Note that sorting is not neccessary
#             # here, we only do it due to better numerical stability of the algorithm
#             displ = [[np.linalg.norm(integ_i - integ[0]), integ_i - integ[0]] for integ_i in integ[1:]]
#             displ = sorted(displ, key=lambda x: x[0], reverse=True)
#
#             renormalized_vector_norms = np.zeros(self.N)  # normalization factors
#             e = np.zeros((self.N, self.N))  # ortonormalized_basis
#
#             # Gram - Schmidt orthonormalizing
#             for i in range(self.N):
#                 e[i,:] = displ[i][1]
#                 for j in range(i):
#                     e[i,:] -= np.dot(displ[i][1], e[j,:]) * e[j,:]
#
#                 renormalized_vector_norms[i] = np.linalg.norm(e[i])
#                 e[i,:] = e[i,:] / renormalized_vector_norms[i]
#
#             # Save renormalization factors
#             all_betas[iter_num-1,:] = sorted(renormalized_vector_norms / delta_start, reverse=True)
#             deltas_stop = all_betas[iter_num-1,0] * delta_start  # save the biggest renorm. factor
#
#             # Re-initiate new initial conditions for the next loop
#             Z0[0] = integ[0]
#             for i in range(self.N):
#                 Z0[i+1] = integ[0] + e[i,:] * delta_start
#
#         lambdas = [1.0/(T*max_iter_num) * np.sum([np.log(all_betas[j,i]) for j in range(max_iter_num)]) for i in range(self.N)]
#
#         self.lyapunov_spectrum.append({'init_cond' : init_cond,
#                                        'renormalization time' : T,
#                                        'max iteration number' : max_iter_num,
#                                        'delta_start' : delta_start,
#                                        'average of log10(delta_stop)' : np.average(np.log10(deltas_stop)),
#                                        'lyapunov exponents': lambdas
#                                        })
#
#
#     def calculate_PSD(self, eval_path, window_length):
#         # Calculate Power Spectral Density for specified evaluated trajectory. Only one (but long) trajectory is
#         # needed if we suppose the attractor has an invariant natural measure. Parameters window is an array of
#         # consecutive integers (< trajectory length) that determines the window for Welch's method. It is correlated
#         # to the resolution of the result.
#         # Results are saved under
#
#         print('Calculating PSD ... eval_path={}'.format(eval_path))
#
#         window_length = np.arange(window_length)
#
#         with open(eval_path, 'rb') as LM05_inp_eval:
#             ev = pickle.load(LM05_inp_eval)
#             del LM05_inp_eval
#
#         freqs, psd = signal.welch(x=ev.y.T[:,0], fs=(ev.t[1]-ev.t[0])**(-1), window=window_length)
#
#         self.PSD.append({'freqs':freqs,
#                          'psd':psd,
#                          'window_length':window_length,
#                          'eval_path':eval_path})
#
#         print('Done\n')
#
#
#     def calculate_periodic_moments(self, eval_path, k_arr):
#         # Return the time average of a function cos(kx) for an array of k's
#
#         with open(eval_path, 'rb') as LM05_inp_eval:
#             ev = pickle.load(LM05_inp_eval)
#             del LM05_inp_eval
#
#         dt = ev.t[1] - ev.t[0]
#         T = ev.t[-1]
#         f_Z_integs = np.zeros(len(k_arr))
#
#         f = lambda k, x: np.cos(k*x)
#
#         # for each k avarage the value of f over time and over all trajectories
#         for i in range(len(k_arr)):
#             if self.verbose:
#                 print('Calculating periodic moment for k={} ({}/{})'.format(k_arr[i], i+1, len(k_arr)))
#             for j in range(self.N):
#                 f_Z_integs[i] += 1/(T*self.N) * np.trapz(y=f(k_arr[i], ev.y.T[:, j]), dx=dt)
#
#
#         self.moments.append({'k_arr' : k_arr,
#                              'mom_arr' : f_Z_integs,
#                              'eval_path' : eval_path})
#
#
#     def calculate_mutual_information(self, eval_path, time_delays, partition_num=50):
#         # Mutual information as a function of time delay. Values in an array time_delays should be
#         # some multiples of dt from provided evaluation. A scalar function that is used in the process
#         # is the average of all components at one time instance
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
#         print('\nCalculating Mutual Information:')
#         print('dt from evaluation: ', dt)
#         print('max time from evaluation: ', t[-1])
#         print('partition num.: ', partition_num)
#         print('\n')
#
#         # Check if provided time delays are multiples of dt
#         for T in time_delays:
#             if (round(T/dt, 10))%1 != 0:  # round because of numerical error
#                 raise ValueError('Values in an array time_delays should be'
#                                  ' some multiples of dt from provided evaluation')
#
#         partitions_borders = np.linspace(np.min(x0), np.max(x0), partition_num + 1, endpoint=True)
#         mut_inf = np.zeros(len(time_delays))
#
#         def bin(x, y):
#             # Return the mutual partition of two numbers
#             xi, yi = -1, -1
#             for i in range(partition_num):
#                 if partitions_borders[i + 1] >= x:
#                     xi = i
#                     break
#             for i in range(partition_num):
#                 if partitions_borders[i + 1] >= y:
#                     yi = i
#                     break
#             if not (xi==-1 and yi==-1):
#                 return xi,yi
#             else:
#                 raise ValueError('x or y not in any partition')
#
#         for i,T in enumerate(time_delays):
#             if self.verbose:
#                 print('Calculating for time delay {} ({}/{})'.format(round(T,10), i+1, len(time_delays)))
#
#
#             P_mn = np.zeros((partition_num, partition_num)) # Num. of events when x(t) is in bin m and x(t+T) is in bin n
#             P_m = np.zeros(partition_num)  # Num. of events when x(t) is in bin n
#             Ti = int(T/dt) # index that will point to time t + T
#
#             for j in range(len(x0)-Ti):
#                 m, n = bin(x0[j], x0[j+Ti])
#                 P_m[m] += 1
#                 P_mn[m,n] += 1
#
#             # Num. of events to probabilities
#             P_m = P_m / np.sum(P_m)
#             P_mn = P_mn / np.sum(P_mn)
#
#             for m in range(partition_num):
#                 if P_m[m] != 0:
#                     mut_inf[i] += - 2*P_m[m] * np.log(P_m[m])
#                 for n in range(partition_num):
#                     if P_mn[m,n] != 0:  # we take out zeros otherwise log(0) = -inf and PlogP produces a nan
#                         mut_inf[i] += P_mn[m,n] * np.log(P_mn[m,n])
#
#         self.mutual_information.append({'time_delays' : time_delays,
#                                         'mutual_info' : mut_inf,
#                                         'partition_num' : partition_num,
#                                         'eval_path' : eval_path
#                                         })
#
#
#     def calculate_FNN_v1(self, eval_path, T, Dt, R_tol=10, A_tol=1, norm='l2', time_units_apart = None, td_dim_span=(None, None)):
#         # Calculates False Nearest Neighbour function. Par. T is the time lag used to construct time-delay
#         # coordinates, Dt is the spacing in time when delay coordinates are constructed and td_dim is a tuple (min, max)
#         # dimension of time delay coordinates vector space (where we estimate FNN). Both T as delta_t should be
#         # multiples of provided dt (from eval), and aldo T should be a multiple of delta_t. Param time_units_apart
#         # defines the number of time units two compared points should be apart when finding a NN. It is set by default
#         # such that two points can be at min Dt time unit close when searching for NN. A scalar function that is used in the
#         # process is the average of all components at one time instance. td_dim_span[0]=td_dim_min must be at least 1.
#         # R_tol and A_tol are method intrinsic parameters (Determining embedding dimension for phase-space
#         # reconstruction using a geometrical construction, M.b. Kennel, R. Brown, H.D.I. Abarbanel, 1992)
#         # Additional parameter to the ones proposed in the article is the 'norm' of the distance between two points. In
#         # the original article l2 norm is used, but one can choose here also dimension-normalized l2 ('l2/d') and the
#         # supremum ('max') norm
#
#
#         # Set default value of td_dim
#         if td_dim_span == (None, None):
#             td_dim_min = 1
#             td_dim_max = self.N + 1
#         else:
#             if td_dim_span[0] < 1:
#                 raise ValueError('Min time delay dimension cannot be smaller then 1')
#             td_dim_min = td_dim_span[0]
#             td_dim_max = td_dim_span[1] + 1
#
#         if time_units_apart == None:
#             time_units_apart = Dt
#
#         with open(eval_path, 'rb') as LM05_inp_eval:
#             ev = pickle.load(LM05_inp_eval)
#
#             dt = ev.t[1] - ev.t[0]
#             t = ev.t
#             x0 = np.sum(ev.y.T, axis=-1) / self.N # average of all components at a time instant
#
#             # for i in range(len(x0)):
#             #     x0[i] += np.random.uniform()/100
#
#             del LM05_inp_eval, ev
#
#         print('\nCalculating FNNv1. Parameters:')
#         print('Time delay (T): ', T)
#         print('dt from evaluation: ', dt)
#         print('max time from evaluation: ', t[-1])
#         print('new dt:=Dt: ', Dt)
#         print('time_units_apart: ', time_units_apart)
#         print('R_tol: ', R_tol)
#         print('A_tol: ', A_tol)
#         print('norm: ', norm)
#         print('\n')
#
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
#         if t[-1] < time_units_apart / 2:
#             print('Warning: t_max < time_units_apart / 2')
#
#         indices_apart = int(time_units_apart / Dt)
#
#         # Estimate the variance of the signal x
#         var = np.var(x0)
#
#         # generate time delay vector function coordinates of dimension td_dim
#         l = int((len(x0) - Ti * (td_dim_max - 1)) / Dti)
#         print('num. of datapoints: ', l)
#         print('\n')
#
#         TD = np.zeros((l, td_dim_max))
#         for i in range(l):
#             for j in range(td_dim_max):
#                 TD[i,j] = x0[i*Dti + j*Ti]
#
#
#         def get_NN(d):
#             # returns the indices of NN for dim. of delayed coord. space d, the distance squared
#             # between these two points, and the distance squared between points in d+1 dim.
#             # (returns [i, NN_i,  dist,  dist_of_dplus1] for each index i)
#
#             # Initialize arrays
#             NN_ind_arr = np.zeros((l, 4))
#             TDd = TD[:,:d]
#             TDdplus1 = TD[:,d]
#
#             # Find NN in dim. of time delay d and save the  dist. of every pair
#             for i, pi in enumerate(TDd):
#                 NNj = None
#                 c = 10**10  # initial  distance to NN
#
#                 for j, pj in enumerate(TDd):
#                     if np.abs(i - j) > indices_apart:
#
#                         if norm == 'max':
#                             c_new = np.linalg.norm(pi-pj, ord=np.inf)
#                         elif norm == 'l2':
#                             c_new = np.linalg.norm(pi-pj) ** 2
#                         elif norm == 'l2/d':
#                             c_new = np.linalg.norm(pi-pj) ** 2 / d
#                         else:
#                             raise ValueError("Parameter 'norm' should be one of ('max', 'l2', 'l2/d')")
#
#                         if c_new < c:
#                             c = c_new
#                             NNj = j
#
#                 NN_ind_arr[i] = [i,NNj, c, 0]
#
#             # Calculate the increase in dist. of every pair for dim. of time delay vector space d+1
#             for NN_pair in NN_ind_arr:
#                 i, j = int(NN_pair[0]), int(NN_pair[1])
#                 NN_pair[3] = np.abs(TDdplus1[i] - TDdplus1[j])
#
#             return NN_ind_arr
#
#         # Initialize return arrays
#         fnn = []
#         d_arr = []
#         threshold_choice = [] # for each d, if R_tol * NNd[i,2] was always chosen for threshold the value is 1
#                               # and 0 if times A_tol * var was always chosen to be the threshold (and linear inbetween)
#
#         # Attractor size bound
#         A = A_tol * var
#
#         # Calculate fnn for every d
#         for d in np.arange(td_dim_min, td_dim_max):
#
#             if self.verbose:
#                 print('Calculating for dim. {}/{}'.format(d, td_dim_max-1))
#
#             NNd = get_NN(d)
#
#             score = 0
#             thr = 0
#
#             NNdplus1_avr = 0
#             R_avr = 0
#
#             for i in range(l):
#
#                 R = R_tol * NNd[i,2]
#
#                 NNdplus1_avr += NNd[i,3]
#                 R_avr += R
#
#                 threshold = np.max([R, A])
#                 if threshold == R:
#                     thr += 1
#
#                 if NNd[i,3] > threshold:
#                     score += 1
#
#             NNdplus1_avr /= l
#             R_avr /= l
#             # print('##################',NNdplus1_avr, R_avr, A)
#
#
#             d_arr.append(d)
#             fnn.append(score / l)
#             threshold_choice.append(thr / l)
#
#             if self.verbose:
#                 print('fnn(d): ', fnn[-1])
#                 print('threshold choice: ', threshold_choice[-1])
#
#             # break loop if fnn(d) = 0
#             if fnn[-1] == 0:
#                 break
#
#
#         self.FNN_v1.append({'d_arr' : np.asarray(d_arr),
#                         'fnn' : np.asarray(fnn),
#                         'thr_choice' : threshold_choice,
#                         'eval_path' : eval_path,
#                         'T' : T,
#                         'Dt' : Dt,
#                         'time_units_apart' : time_units_apart,
#                         'td_dim_span' : td_dim_span})
#
#
#     def calculate_FNN_v2(self, eval_path, T, Dt, norm = 'l2', time_units_apart = None, td_dim_span=(None, None)):
#         # Calculates False Nearest Neighbour according to the approach proposed in 'Practical method for determining
#         # the minimum embedding dimension of a scalar time series' by Liangyue Cao (Physica D, 1997). Par. T is the
#         # time lag used to construct time-delay coordinates, Dt is the spacing in time when delay coordinates are
#         # constructed and td_dim is a tuple (min, max) dimension of time delay coordinates vector space (where we
#         # estimate FNN). Both T as delta_t should be multiples of provided dt (from eval), and aldo T should be a
#         # multiple of delta_t. Param time_units_apart defines the number of time units two compared points should be
#         # apart when finding a NN. It is set by default such that two points can be at min Dt time unit close when
#         # searching for NN. A scalar function that is used in the process is the average of all components at one time
#         # instance. td_dim_span[0]=td_dim_min must be at least 1.
#         # Additional parameter to the ones proposed in the article is the 'norm' of the distance between two points. In
#         # the original article supremum ('max') norm is used, but one can choose here also dimension-normalized l2
#         # ('l2/d') and the regular l2 ('l2') norm
#
#
#         # Set default value of td_dim
#         if td_dim_span == (None, None):
#             td_dim_min = 1
#             td_dim_max = self.N + 3
#         else:
#             if td_dim_span[0] < 1:
#                 raise ValueError('Min time delay dimension cannot be smaller then 1')
#             td_dim_min = td_dim_span[0]
#             td_dim_max = td_dim_span[1] + 1
#
#         if time_units_apart == None:
#             time_units_apart = Dt
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
#         print('\nCalculating FNNv2. Parameters:')
#         print('Time delay (T): ', T)
#         print('dt from evaluation: ', dt)
#         print('max time from evaluation: ', t[-1])
#         print('new dt:=Dt: ', Dt)
#         print('time_units_apart: ', time_units_apart)
#         print('norm: ', norm)
#         print('\n')
#
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
#         if t[-1] < time_units_apart / 2:
#             print('Warning: t_max < time_units_apart / 2')
#
#         indices_apart = int(time_units_apart / Dt)
#
#
#         # generate time delay vector function coordinates of dimension td_dim
#         l = int((len(x0) - Ti * (td_dim_max - 1)) / Dti)
#         print('num. of datapoints: ', l)
#         print('\n')
#
#         TD = np.zeros((l, td_dim_max))
#         for i in range(l):
#             for j in range(td_dim_max):
#                 TD[i,j] = x0[i*Dti + j*Ti]
#
#
#         def get_NN(d):
#             # returns the indices of NN for dim. of delayed coord. space d, the distance squared
#             # between these two points, and the distance squared between points in d+1 dim.
#             # (returns [i, NN_i,  dist,  dist_of_dplus1] for each index i)
#
#             # Initialize arrays
#             NN_ind_arr = np.zeros((l, 4))
#             TDd = TD[:,:d]
#             TDdplus1 = TD[:,d]
#
#             # Find NN in dim. of time delay d and save the  dist. of every pair
#             for i, pi in enumerate(TDd):
#                 NNj = None
#                 c = 10**10  # initial  distance to NN
#
#                 for j, pj in enumerate(TDd):
#                     if np.abs(i - j) > indices_apart:
#
#                         if norm == 'max':
#                             c_new = np.linalg.norm(pi-pj, ord=np.inf)
#                         elif norm == 'l2':
#                             c_new = np.linalg.norm(pi-pj) ** 2
#                         elif norm == 'l2/d':
#                             c_new = np.linalg.norm(pi-pj) ** 2 / d
#                         else:
#                             raise ValueError("Parameter 'norm' should be one of ('max', 'l2', 'l2/d')")
#
#                         if c_new < c:
#                             c = c_new
#                             NNj = j
#
#                 NN_ind_arr[i] = [i,NNj, c, 0]
#
#             # Calculate the increase in dist. of every pair for dim. of time delay vector space d+1
#             for NN_pair in NN_ind_arr:
#                 i, j = int(NN_pair[0]), int(NN_pair[1])
#                 dplus1dist = np.abs(TDdplus1[i] - TDdplus1[j])
#                 if dplus1dist > NN_pair[2]:
#                     NN_pair[3] = dplus1dist
#                 else:
#                     NN_pair[3] = NN_pair[2]
#
#             return NN_ind_arr
#
#         # Initialize return arrays
#         E_arr = []
#         E_star_arr = []
#         d_arr = []
#
#         # Calculate fnn for every d
#         for d in np.arange(td_dim_min, td_dim_max):
#
#             if self.verbose:
#                 print('Calculating for dim. {}/{}'.format(d, td_dim_max-1))
#
#             NNd = get_NN(d)
#
#             E_d = 0
#             E_star_d = 0
#
#             for i in range(l):
#
#                 a_d = NNd[i,3] / NNd[i,2]
#                 E_d += a_d
#                 E_star_d += NNd[i,3]
#
#             d_arr.append(d)
#             E_arr.append(E_d / l)
#             E_star_arr.append(E_star_d / l)
#
#         d_arr = d_arr[:-1]
#         E1_arr = [E_arr[i+1]/E_arr[i] for i in range(len(d_arr))]
#         E2_arr = [E_star_arr[i+1]/E_star_arr[i] for i in range(len(d_arr))]
#
#         print(d_arr)
#         print(E1_arr)
#         print(E2_arr)
#
#
#         self.FNN_v2.append({'d_arr' : np.asarray(d_arr),
#                         'E1_arr' : np.asarray(E1_arr),
#                         'E2_arr' : np.asarray(E2_arr),
#                         'eval_path' : eval_path,
#                         'T' : T,
#                         'Dt' : Dt,
#                         'time_units_apart' : time_units_apart,
#                         'td_dim_span' : td_dim_span})
#
#
# def calculate_FNN_v3(self, eval_path, T, Dt, norm='l2', Dt_n_apart=None, td_dim_span=(None, None)):
#     # Calculates False Nearest Neighbour function. Par. T is the time lag used to construct time-delay
#     # coordinates, Dt is the spacing in time when delay coordinates are constructed and td_dim is a tuple (min, max)
#     # dimension of time delay coordinates vector space (where we estimate FNN). Both T as delta_t should be
#     # multiples of provided dt (from eval), and aldo T should be a multiple of delta_t. Param Dt_n_apart defines
#     # the number of Dt units two compared points should be apart when finding a NN. It is set by default such that
#     # two points can be at min 1 time unit close when searching for NN. A scalar function that is used in the
#     # process is the average of all components at one time instance.
#
#     # Set default value of td_dim
#     if td_dim_span == (None, None):
#         td_dim_min = 1
#         td_dim_max = self.N + 5
#     else:
#         td_dim_min = td_dim_span[0]
#         td_dim_max = td_dim_span[1]
#
#     if Dt_n_apart == None:
#         Dt_n_apart = 10
#
#     with open(eval_path, 'rb') as LM05_inp_eval:
#         ev = pickle.load(LM05_inp_eval)
#
#         dt = ev.t[1] - ev.t[0]
#         t = ev.t
#         x0 = np.sum(ev.y.T, axis=-1) / self.N  # average of all components at a time instant
#
#         # x0 = np.random.normal(size=len(x0))
#         # x0 = np.sin(np.arange(len(x0))*0.01)
#
#         del LM05_inp_eval, ev
#
#     print('\nCalculating FNN. Parameters:')
#     print('Time delay: ', T)
#     print('dt from evaluation: ', dt)
#     print('new dt:=Dt: ', Dt)
#
#     # intigers pointing to the element equivalent to T and Dt
#     Ti = int(T / dt)
#     Dti = int(Dt / dt)
#
#     # Check if provided T/dt, delta_t/dt and T/delta_t are all intigers
#     if (round(Ti, 10)) % 1 != 0:  # round because of numerical error
#         raise ValueError('Time_delay should be'
#                          ' some multiple of dt from provided evaluation')
#     if (round(Dti, 10)) % 1 != 0:  # round because of numerical error
#         raise ValueError('Par. delta_t should be'
#                          ' some multiple of dt from provided evaluation')
#     if (round(T / Dt, 10)) % 1 != 0:  # round because of numerical error
#         raise ValueError('Par. T should be'
#                          ' some multiple of delta_t from provided evaluation')
#     if (round(len(x0) / Dti, 10)) % 1 != 0:  # round because of numerical error
#         raise ValueError('Number of points from provided evaluation should'
#                          ' be dividable by Dt/dt')
#
#     # generate time delay vector function coordinates of dimension td_dim
#     l = int((len(x0) - Ti * (td_dim_max - 1)) / Dti)
#     TD = np.zeros((l, td_dim_max))
#     for i in range(l):
#         for j in range(td_dim_max):
#             TD[i, j] = x0[i * Dti + j * Ti]
#
#     def get_NN(d):
#         # returns the indices of NN for dim. of delayed coord. space d
#
#         NN_ind_arr = np.zeros((l, 2))
#         TDd = TD[:, :d]
#
#         for i, pi in enumerate(TDd):
#             NNj = None
#             c = 10 ** 10  # initial distance to NN
#
#             for j, pj in enumerate(TDd):
#                 if np.abs(i - j) > Dt_n_apart:
#                     if norm == 'l2':
#                         c_new = np.linalg.norm(pi - pj)
#                     elif norm == 'sup':
#                         c_new = np.linalg.norm(pi-pj, ord=np.inf)
#
#                     #print(c_new)
#
#                     if c_new < c:
#                         c = c_new
#                         NNj = j
#
#             NN_ind_arr[i] = [i, NNj]
#             print(i,NNj)
#
#         return (NN_ind_arr)
#
#     fnn = np.zeros(td_dim_max - td_dim_min - 1)
#     d_arr = np.arange(td_dim_min, td_dim_max - 1)
#     NNd = get_NN(td_dim_min)
#
#     for d in d_arr:
#
#         if self.verbose:
#             print('Calculating for dim. {}/{}'.format(d, td_dim_max - 1))
#
#         NNdplus1 = get_NN(d + 1)
#
#         score = 0
#         for i in range(l):
#             if NNd[i, 1] == NNdplus1[i, 1]:
#                 score += 1
#
#         fnn[d - td_dim_min] = 1 - score / l
#         if self.verbose:
#             print('fnn(d): ', fnn[d - td_dim_min])
#
#         NNd = NNdplus1
#
#     self.FNN_v3.append({'d_arr': d_arr,
#                      'fnn': fnn,
#                      'eval_path': eval_path,
#                      'T': T,
#                      'Dt': Dt,
#                      'Dt_n_apart': Dt_n_apart,
#                      'td_dim': td_dim_max})
#
#
# def calculate_min_dist(self, eval_path):
#     #
#
#     # with open(eval_path, 'rb') as L05_2_inp_eval:
#     #     ev = pickle.load(L05_2_inp_eval)
#     #
#     #     dt = ev.t[1] - ev.t[0]
#     #     dt_i = int(1/dt)
#     #     t = ev.t
#     #     l = len(t)
#     #     x0 = np.sum(ev.y.T, axis=-1) / self.N # average of all components at a time instant
#     #
#     #     del L05_2_inp_eval, ev
#     #
#     # TD = np.zeros((l, td_dim_max))
#     # for i in range(l):
#     #     for j in range(td_dim_max):
#     #         TD[i, j] = x0[i * Dti + j * Ti]
#
#     pass
#
#
# def calculate_BCD(self, eval_path, eps_arr):
#     # Box Counting Dimension. Takes in
#
#     # Load data
#     with open(eval_path, 'rb') as L05_2_inp_eval:
#         ev = pickle.load(L05_2_inp_eval)
#
#         x = ev.y.T
#         l = len(ev.y.T)
#
#         del L05_2_inp_eval, ev
#
#
#     # check for the smallest distance between first point and its nearest points that is not too close in time
#     # - this dist. must (later) not be smaller than the smallest epsilon
#     min_dist = 1e10
#     for i in range(0,l-2):
#         dist = np.linalg.norm(x[i]-x[l-1])
#         if dist < min_dist:
#             min_dist = dist
#
#     print('min_dist: ', min_dist)
#
#     del x
#
#     N_arr = []
#     d_arr = []
#
#     # For each epsilon, calculate into which box a point falls and save the box position. Afterwards,
#     # turn the array of all boxes into a set (only different elements
#     for eps in eps_arr:
#
#         if eps < min_dist:
#             print('eps={} < min_dist between time consecutive points ={}'.format(eps,min_dist))
#         else:
#
#             # Load data
#             with open(eval_path, 'rb') as L05_2_inp_eval:
#                 ev = pickle.load(L05_2_inp_eval)
#
#                 x = ev.y.T
#                 l = len(ev.y.T)
#
#                 del L05_2_inp_eval, ev
#
#             set = np.unique(x // eps, axis=0)
#
#             N = len(set)
#             d = np.log(N) / np.log(1/eps)
#
#             del set
#
#             print('eps={}, N={}, d={}'.format(eps, N, d))
#
#             N_arr.append(N)
#             d_arr.append(d)
#
#     eps_arr = eps_arr[:len(N_arr)]
#
#     print(eps_arr)
#     print(N_arr)
#     print(d_arr)
#
#     self.BCD.append({'d' : d_arr,
#                      'eps_arr' : eps_arr,
#                      'N_arr' : N_arr})
#
#
# def calculate_information_dim(self, eval_path, partition_num=1000):
#     # Calculate information dimension of the first component of trajectory
#
#     # Load data
#     with open(eval_path, 'rb') as L05_2_inp_eval:
#         ev = pickle.load(L05_2_inp_eval)
#
#         x0 = ev.y[0] # first component
#         l = len(x0)
#
#         del L05_2_inp_eval, ev
#
#     mn, mx = np.min(x0), np.max(x0)
#     eps = (mx - mn) / partition_num
#     bins_borders = np.linspace(mn, mx, partition_num + 1, endpoint=True)
#
#     bins = np.zeros(partition_num)
#
#     for i,x_i in enumerate(x0):
#         if self.verbose:
#             print('{}/{}'.format(i,l))
#
#         for j in range(partition_num):
#             if bins_borders[j + 1] >= x_i:
#                 bins[j] += 1
#                 break
#
#     partitions = bins/l
#
#     print(partitions)
#
#     d = 0
#     for p in partitions:
#         if p != 0:
#             d += p*np.log(p)
#
#     d = d / (self.N * np.log(eps))
#
#     print(d)
#
