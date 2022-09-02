from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pysindy as ps
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from pysindy.optimizers import SINDyOptimizer
import numpy as np
import models
import sys
import time


# ignore user warnings
import warnings
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)




# Build the energy-preserving quadratic nonlinearity constraints
def make_constraints(r):
    # r = phase space size

    # number of functions of 1st and 2nd order
    N = int((r ** 2 + 3 * r) / 2.0) # note there is no +1 term (we take care of this with unconstrained_constant_terms_matrix
    # number of constraints
    p = int(r * (r + 1) * (r + 2) / 6.0)

    constraint_zeros = np.zeros(p)
    constraint_matrix = np.zeros((p, r * N))
    unconstrained_constant_terms_matrix = np.zeros((p,r))

    q = 0

    # Set coefficients adorning terms like a_i^3 to zero
    for i in range(r):
        constraint_matrix[q, r * (N - r) + i * (r + 1)] = 1.0
        q = q + 1

    # Set coefficients adorning terms like a_ia_j^2 to be antisymmetric
    for i in range(r):
        for j in range(i + 1, r):
            constraint_matrix[q, r * (N - r + j) + i] = 1.0
            constraint_matrix[q, r * (r + j - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)] = 1.0
            q = q + 1
    for i in range(r):
         for j in range(0, i):
            constraint_matrix[q, r * (N - r + j) + i] = 1.0
            constraint_matrix[q, r * (r + i - 1) + j + r * int(j * (2 * r - j - 3) / 2.0)] = 1.0
            q = q + 1

    # Set coefficients adorning terms like a_ia_ja_k to be antisymmetric
    for i in range(r):
        for j in range(i + 1, r):
            for k in range(j + 1, r):
                constraint_matrix[q, r * (r + k - 1) + i + r * int(j * (2 * r - j - 3) / 2.0)] = 1.0
                constraint_matrix[q, r * (r + k - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)] = 1.0
                constraint_matrix[q, r * (r + j - 1) + k + r * int(i * (2 * r - i - 3) / 2.0)] = 1.0
                q = q + 1

    # there are no constraints on linear terms
    constraint_matrix = np.hstack([unconstrained_constant_terms_matrix, constraint_matrix])

    return constraint_zeros, constraint_matrix


##################################################################################
##################################################################################
##################################################################################

# Model parameters
pars = {'N' : 20,
        'K' : 2,
        'F' : 30
        }

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-15
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-10

# Seed for reproducibility
np.random.seed(1000)

# Special library ordering needed again if constraints are used
# library_functions = [lambda : 1, lambda x: x, lambda x, y: x * y, lambda x: x ** 2]
# library_function_names = [lambda : 1, lambda x: x, lambda x, y: x + y, lambda x: x + x]
# sindy_library = ps.CustomLibrary(
#     library_functions=library_functions,
#     function_names=library_function_names
# )
sindy_library = ps.PolynomialLibrary() # It's modified to be ordered correctly regarding the constraints

# Differentiation methods
diffs = {
    'SFD': ps.SmoothedFiniteDifference(),
    'Spline': ps.SINDyDerivative(kind='spline', s=1e-0),
    'FD' : ps.FiniteDifference(order=3,d=2,drop_endpoints=True),
}

# Some global parameters
n_targets = pars['N']    # = r = phase space size
n_features = int((n_targets**2 + 3*n_targets)/2 + 1) # = number of functions (=10 for n_targets=3)
feature_names = ['x{}'.format(i) for i in range(n_targets)]
make_predictions = True # make predictions?
make_graphs = True      # make graphs?

noise = 0.0           # mean Gaussian
cov_off_diag_vs_diag_ampl = 0
noise_type = 1 # 1, 2

num_of_all_data_points = 100000#zadt0.01 #30000zadt0.005split10 # number of all data points leaded (take the last part of loaded data points)
sparsification_factor = 10 # take every sparsification_factor point of the loaded sample

cv_type = 'time_split_series' #'time_split_series' or 'predefined_split'
slice_factor = 9 #10  # the data will be divided into slice_factor sets of training data and 1 test data set i.e.
                  # the data will be divided into (slice_factor+1) equally big sets
test_size = 1000 # num. of points for the test size


# SINDy parameters
diff_method = 'SFD' # # 'Spline' or 'SFD' or 'FD'
max_iter = 30

# STLSQ parameters
param_grid_STLSQ = {
    "optimizer__threshold": [0.001, 0.01, 0.1],
    "optimizer__alpha":  [0.001, 0.01, 0.1],
    "optimizer__max_iter" : [max_iter] #[i for i in range(1,30)]
}

# SC3 parameters
regularizer_SR3 = 'l0'
param_grid_SR3 = {
    "optimizer__threshold": [0.01], #[0.001, 0.01, 0.1],
    "optimizer__nu": [0.01], #[0.1, 1, 10], #[0.1, 0.1, 10]
    "optimizer__max_iter" : [max_iter]
}

# ConstrainedSR3 parameters
regularizer_consSR3 = 'l0'
param_grid_consSR3 = {
    "optimizer__threshold": [0.001, 0.01, 0.1],
    "optimizer__nu": [0.1, 0.1, 10],
    "optimizer__max_iter" : [max_iter]
}

# TrappingSR3 parameters
eta = 1e10
param_grid_trapSR3 = {
    # "optimizer__eta": [1], # searchgrid not passing on this value
    "optimizer__threshold": [0.01], #[0.01, 0.001],
    "optimizer__gamma": [-1],
    "optimizer__max_iter" : [max_iter]
}


################ Prepare input data ##########################################

# Load evaluated trajectory
eval_path = "data/L05_2_{'N': 20, 'K': 2, 'F': 30}/eval_[20, 2, 30, 0.001, 10000]_d70a3db09d46e5f3472bf12b4303610c042d5df4.pkl"
# eval_path = "data/eval_{'N': 20, 'K': 2, 'F': 30, 'dt': 0.001, 't1': 1000.0}_f65e47e20531df16d919cbbfd3ce001f1405b727_1659731820.6279635.pkl"
#eval_path = "eval"
with open(eval_path, 'rb') as inp_eval:
    ev = pickle.load(inp_eval)
    del inp_eval

# make training and testing data by splitting all loaded data into chunks
if len(ev.t) < num_of_all_data_points:
    raise ValueError('Number of all loaded data points (={}) cannot be smaller'
                     'than num_of_used_data_points (={})'.format(len(ev.t), num_of_all_data_points))
t = ev.t[:num_of_all_data_points:sparsification_factor]
x = ev.y.T[-num_of_all_data_points::sparsification_factor]
print(x.shape)
del ev
dt = t[1] - t[0]
num_of_used_data_points = int(num_of_all_data_points / sparsification_factor)
if cv_type == 'time_split_series':
    if test_size > int(num_of_used_data_points / (slice_factor+1)):
        raise ValueError('test_size too big')

# -------------- add noise on data --------------------

if noise_type == 1:
    # noise of type [[1 a a ... a],   of dimension N
    #                [a 1 a ... a],          and a = cov_off_diag_vs_diag_ampl
    #                [...........],
    #                [a ... a 1 a],
    #                [a ... a a 1]]
    cov = np.ones((pars['N'], pars['N'])) * cov_off_diag_vs_diag_ampl + np.diag(np.ones(pars['N'])
                                                                                * (1-cov_off_diag_vs_diag_ampl))
elif noise_type == 2:
    # noise of type [[1 a 0 ... 0],   of dimension N
    #                [a 1 0 ... 0],          and a = cov_off_diag_vs_diag_ampl
    #                [0 0 1 ... 0],
    #                [...........],
    #                [0 ... 0 0 1]]
    cov = np.diag(np.ones(pars['N']))
    cov[0,1], cov[1,0] = cov_off_diag_vs_diag_ampl, cov_off_diag_vs_diag_ampl

print('Covariance matrix:')
print(cov)
print('\n')
mu = np.random.multivariate_normal(np.zeros(pars['N']), cov, size=len(x)) * noise
x_og = x
x = x + mu

# -----------------------------------------------------

if cv_type == 'predefined_split':
    cv = PredefinedSplit(np.append([-1 for i in range(len(x) - test_size)], [0 for i in range(len(x) - test_size, len(x))]))
    print('\n####################################################\n')
    print('Diff. method: ', diff_method)
    print('Noise: ', noise, '(mean gaussian)')
    print('cov_off_diag_vs_diag_ampl: ',  cov_off_diag_vs_diag_ampl)
    print('noise_type: ', noise_type)
    print('dt: ', dt)
    print('Num. of data points: ', num_of_used_data_points)
    print('Test size: ', test_size)
    print('Max iter: ', max_iter)
elif cv_type == 'time_split_series':
    cv = TimeSeriesSplit(n_splits=slice_factor,  test_size=test_size)
    print('\n####################################################\n')
    print('Diff. method: ', diff_method)
    print('Noise: ', noise, '(mean gaussian)')
    print('cov_off_diag_vs_diag_ampl: ',  cov_off_diag_vs_diag_ampl)
    print('noise_type: ', noise_type)
    print('dt: ', dt)
    print('Num. of data points: ', num_of_used_data_points)
    print('slice_factor: {}'.format(slice_factor), '(=num. of training data sets)')
    print('Chunk size: ', (num_of_used_data_points - test_size) / slice_factor)
    print('Test size): ', test_size)
    print('Max iter: ', max_iter)
else:
    raise ValueError("cv_type not 'predefined_split' or 'time_split_series'")



################## Parameter Search ###########################################

# STLSQ Poly3
if False:

    print('\n####################################################\n')
    print('STLSQ model:')
    for key, val in param_grid_STLSQ.items():
        print('{}: {}'.format(key, val))
    print('\n')

    opt = ps.STLSQ(max_iter=max_iter)
    model = ps.SINDy(t_default=dt,
                     feature_names=feature_names,
                     optimizer=opt,
                     feature_library=ps.PolynomialLibrary(3),
                     differentiation_method=diffs[diff_method],
                     )
    search = GridSearchCV(
        model,
        param_grid_STLSQ,
        cv=TimeSeriesSplit(n_splits=slice_factor),
        verbose=4
    )
    search.fit(x) #, quiet=True)

    print("Best parameters:", search.best_params_)
    search.best_estimator_.print()
    print(search.cv_results_)

    results = {'best_est' : search.best_estimator_,
               'best_params' : search.best_params_,
               'best_score' : search.best_score_,
               'cv_results' : search.cv_results_,
               'n_splits' : search.n_splits_,
               'noise_matrix': mu,
               'x' : x,
               'x_og': x_og,
               'dt' : t[1]-t[0],
               'test_size' : test_size,
               'num_of_points': num_of_used_data_points,
               'max_iter' : max_iter,
               'dif_meth' : diff_method,
               'noise' : [noise, cov_off_diag_vs_diag_ampl, noise_type]}

    # Save results
    add_pars = {'mu':[noise, cov_off_diag_vs_diag_ampl, noise_type],
                'dt':dt,
                'test_size':test_size,
                'all_point':num_of_used_data_points,
                'max_it':max_iter,
                'df':diff_method}
    if cv_type == 'time_split_series':
        add_pars['slice_factor'] = slice_factor
    with open('data/L05_2_{}/L05_2_SINDy_STLSQ_Poly3_search_{}_{}_{}_{}_{}.pkl'.format(pars, pars, cv_type,
                                                                 add_pars,
                                                                 models.get_unique_name(x[0].tolist()),time.time()), 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)
        del outp
        del search, results

# STLSQ Poly2
if True:

    print('\n####################################################\n')
    print('STLSQ model:')
    for key, val in param_grid_STLSQ.items():
        print('{}: {}'.format(key, val))
    print('\n')

    opt = ps.STLSQ(max_iter=max_iter)
    model = ps.SINDy(t_default=dt,
                     feature_names=feature_names,
                     optimizer=opt,
                     feature_library=sindy_library,
                     differentiation_method=diffs[diff_method],
                     )
    search = GridSearchCV(
        model,
        param_grid_STLSQ,
        cv=cv,
        verbose=4
    )
    search.fit(x) #, quiet=True)

    print("Best parameters:", search.best_params_)
    search.best_estimator_.print()
    print(search.cv_results_)

    results = {'best_est' : search.best_estimator_,
               'best_params' : search.best_params_,
               'best_score' : search.best_score_,
               'cv_results' : search.cv_results_,
               'n_splits' : search.n_splits_,
               'noise_matrix': mu,
               'x' : x,
               'x_og': x_og,
               'dt' : t[1]-t[0],
               'test_size' : test_size,
               'num_of_points' : num_of_used_data_points,
               'max_iter' : max_iter,
               'dif_meth' : diff_method,
               'noise' : [noise, cov_off_diag_vs_diag_ampl, noise_type]}

    # Save results
    add_pars = {'mu':[noise, cov_off_diag_vs_diag_ampl, noise_type],
                'dt':dt,
                'test_size':test_size,
                'all_point':num_of_used_data_points,
                'max_it':max_iter,
                'df':diff_method}
    if cv_type == 'time_split_series':
        add_pars['slice_factor'] = slice_factor
    with open('data/L05_2_{}/L05_2_SINDy_STLSQ_search_{}_{}_{}_{}_{}.pkl'.format(pars, pars, cv_type,
                                                                 add_pars,
                                                                 models.get_unique_name(x[0].tolist()), str(time.time())), 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)
        del outp
        del search, results

# SR3
if False:

    print('\n####################################################\n')
    print('SR3 model, no constraints:')
    print('regularizer: ', regularizer_SR3)
    for key, val in param_grid_SR3.items():
        print('{}: {}'.format(key, val))
    print('\n')

    opt = ps.SR3(thresholder=regularizer_SR3)
    model = ps.SINDy(t_default=dt,
                     feature_names=feature_names,
                     optimizer=opt,
                     feature_library=sindy_library,
                     differentiation_method=diffs[diff_method],
                     )
    search = GridSearchCV(
        model,
        param_grid_SR3,
        cv=cv,
        verbose=4
    )
    search.fit(x) #, quiet=True)

    print("Best parameters:", search.best_params_)
    search.best_estimator_.print()
    print(search.cv_results_)

    results = {'best_est' : search.best_estimator_,
               'best_params' : search.best_params_,
               'best_score' : search.best_score_,
               'cv_results' : search.cv_results_,
               'n_splits' : search.n_splits_,
               'noise_matrix': mu,
               'x' : x,
               'x_og': x_og,
               'dt' : t[1]-t[0],
               'test_size' : test_size,
               'num_of_points': num_of_used_data_points,
               'max_iter' : max_iter,
               'dif_meth' : diff_method,
               'noise' : [noise, cov_off_diag_vs_diag_ampl, noise_type]}

    # Save results
    add_pars = {'mu':[noise, cov_off_diag_vs_diag_ampl, noise_type],
                'dt':dt,
                'test_size':test_size,
                'all_point':num_of_used_data_points,
                'max_it':max_iter,
                'df':diff_method}
    if cv_type == 'time_split_series':
        add_pars['slice_factor'] = slice_factor
    with open('data/L05_2_{}/L05_2_SINDy_SR3_search_{}_{}_{}_{}_{}.pkl'.format(pars, pars, cv_type,
                                                               add_pars,
                                                               models.get_unique_name(x[0].tolist()),str(time.time())), 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)
        del outp
        del search, results

# Constrained SR3
if False:

    print('\n####################################################\n')
    print('Constrained SR3 model, energy preserving quadratic nonliniarities:')
    print('regularizer: ', regularizer_consSR3)
    for key, val in param_grid_consSR3.items():
        print('{}: {}'.format(key, val))
    print('\n')

    # Make constraints
    constraint_zeros, constraint_matrix = make_constraints(n_targets)

    opt = ps.ConstrainedSR3(thresholder=regularizer_consSR3,
                            constraint_rhs=constraint_zeros,
                            constraint_lhs=constraint_matrix,
                            constraint_order="feature",
                            max_iter=max_iter)
    model = ps.SINDy(t_default=dt,
                     feature_names=feature_names,
                     optimizer=opt,
                     feature_library=sindy_library,
                     differentiation_method=diffs[diff_method],
                     )
    search = GridSearchCV(
        model,
        param_grid_consSR3,
        cv=cv,
        verbose=4
    )
    search.fit(x)  # , quiet=True)

    print("Best parameters:", search.best_params_)
    search.best_estimator_.print()
    print(search.cv_results_)


    results = {'best_est' : search.best_estimator_,
               'best_params' : search.best_params_,
               'best_score' : search.best_score_,
               'cv_results' : search.cv_results_,
               'n_splits' : search.n_splits_,
               'noise_matrix' : mu,
               'x' : x,
               'x_og': x_og,
               'dt' : t[1]-t[0],
               'test_size' : test_size,
               'num_of_points': num_of_used_data_points,
               'max_iter' : max_iter,
               'dif_meth' : diff_method,
               'noise' : [noise, cov_off_diag_vs_diag_ampl, noise_type]}

    # Save results
    add_pars = {'mu':[noise, cov_off_diag_vs_diag_ampl, noise_type],
                'dt':dt,
                'test_size':test_size,
                'all_point':num_of_used_data_points,
                'max_it':max_iter,
                'df':diff_method}
    if cv_type == 'time_split_series':
        add_pars['slice_factor'] = slice_factor
    with open('data/L05_2_{}/L05_2_SINDy_ConsSR3_search_{}_{}_{}_{}_{}.pkl'.format(pars, pars, cv_type,
                                                                   add_pars,
                                                                   models.get_unique_name(x[0].tolist()),str(time.time())),
              'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)
        del outp
        del search, results

# Trapping SR3
if False:

    print('\n####################################################\n')
    print('Trapping SR3 model, energy preserving quadratic nonliniarities:')
    print('eta: ', eta)
    for key, val in param_grid_trapSR3.items():
        print('{}: {}'.format(key, val))
    print('\n')

    # Make constraints
    constraint_zeros, constraint_matrix = make_constraints(n_targets)

    opt = ps.TrappingSR3(thresholder='l1',
                         constraint_rhs=constraint_zeros,
                         constraint_lhs=constraint_matrix,
                         constraint_order="feature",
                         max_iter=max_iter,
                         eta=eta,
                         relax_optim=False
                         )
    model = ps.SINDy(t_default=dt,
                     feature_names=feature_names,
                     optimizer=opt,
                     feature_library=sindy_library,
                     differentiation_method=diffs[diff_method],
                     )

    search = GridSearchCV(
        model,
        param_grid_trapSR3,
        cv=cv,
        verbose=4,
    )
    search.fit(x) #, quiet=True)

    print("Best parameters:", search.best_params_)
    search.best_estimator_.print()
    print(search.cv_results_)

    results = {'best_est' : search.best_estimator_,
               'best_params' : search.best_params_,
               'best_score' : search.best_score_,
               'cv_results' : search.cv_results_,
               'n_splits' : search.n_splits_,
               'noise_matrix': mu,
               'eta' : eta,
               'x' : x,
               'x_og': x_og,
               'dt' : t[1]-t[0],
               'test_size' : test_size,
               'num_of_points': num_of_used_data_points,
               'max_iter' : max_iter,
               'dif_meth' : diff_method,
               'noise' : [noise, cov_off_diag_vs_diag_ampl, noise_type]}

    # Save results
    add_pars = {'eta':eta,
                'mu':[noise, cov_off_diag_vs_diag_ampl, noise_type],
                'dt':dt,
                'test_size':test_size,
                'all_point':num_of_used_data_points,
                'max_it':max_iter,
                'df':diff_method}
    if cv_type == 'time_split_series':
        add_pars['slice_factor'] = slice_factor
    with open('data/L05_2_{}/L05_2_SINDy_TrapSR3_search_{}_{}_{}_{}_{}.pkl'.format(pars, pars, cv_type,
                                                                   add_pars,
                                                                   models.get_unique_name(x[0].tolist()),str(time.time())), 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)
        del outp
        del search, results


print('\n----------------------------')
print(  '-------- Finished ----------')
print('----------------------------\n')




