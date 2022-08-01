import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pysindy as ps
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec
import random as rd

# ignore user warnings
import warnings
import pickle
warnings.filterwarnings("ignore")


# Build the energy-preserving quadratic nonlinearity constraints
def make_constraints(r):
    q = 0
    N = int((r ** 2 + 3 * r) / 2.0) # note there is no +1 term (we take care of this with unconstrained_constant_terms_matrix
    p = int(r * (r + 1) * (r + 2) / 6.0)

    constraint_zeros = np.zeros(p)
    constraint_matrix = np.zeros((p, r * N))
    unconstrained_constant_terms_matrix = np.zeros((p,r))

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

# Use optimal m, and calculate eigenvalues(PW) to see if identified model is stable
def check_stability(r, Xi, sindy_opt, mean_val):
    N = int((r ** 2 + 3 * r) / 2.0)
    opt_m = sindy_opt.m_history_[-1]
    PL_tensor_unsym = sindy_opt.PL_unsym_
    PL_tensor = sindy_opt.PL_
    PQ_tensor = sindy_opt.PQ_
    mPQ = np.tensordot(opt_m, PQ_tensor, axes=([0], [0]))
    P_tensor = PL_tensor - mPQ
    As = np.tensordot(P_tensor, Xi, axes=([3, 2], [0, 1]))
    eigvals, eigvecs = np.linalg.eigh(As)
    print('optimal m: ', opt_m)
    print('As eigvals: ', np.sort(eigvals))
    max_eigval = np.sort(eigvals)[-1]
    min_eigval = np.sort(eigvals)[0]
    L = np.tensordot(PL_tensor_unsym, Xi, axes=([3, 2], [0, 1]))
    Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
    d = np.dot(L, opt_m) + np.dot(np.tensordot(Q, opt_m, axes=([2], [0])), opt_m)
    Rm = np.linalg.norm(d) / np.abs(max_eigval)
    Reff = Rm / mean_val
    print('Estimate of trapping region size, Rm = ', Rm)
    print('Normalized trapping region size, Reff = ', Reff)


# Plot the SINDy trajectory, trapping region, and ellipsoid where Kdot >= 0
def trapping_region(r, x_test_pred, Xi, sindy_opt, filename):

    # Need to compute A^S from the optimal m obtained from SINDy algorithm
    N = int((r ** 2 + 3 * r) / 2.0)
    opt_m = sindy_opt.m_history_[-1]
    PL_tensor_unsym = sindy_opt.PL_unsym_
    PL_tensor = sindy_opt.PL_
    PQ_tensor = sindy_opt.PQ_
    mPQ = np.tensordot(opt_m, PQ_tensor, axes=([0], [0]))
    P_tensor = PL_tensor - mPQ
    As = np.tensordot(P_tensor, Xi, axes=([3, 2], [0, 1]))
    eigvals, eigvecs = np.linalg.eigh(As)
    print('optimal m: ', opt_m)
    print('As eigvals: ', eigvals)

    # Extract maximum and minimum eigenvalues, and compute radius of the trapping region
    max_eigval = np.sort(eigvals)[-1]
    min_eigval = np.sort(eigvals)[0]

    # Should be using the unsymmetrized L
    L = np.tensordot(PL_tensor_unsym, Xi, axes=([3, 2], [0, 1]))
    Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
    d = np.dot(L, opt_m) + np.dot(np.tensordot(Q, opt_m, axes=([2], [0])), opt_m)
    Rm = np.linalg.norm(d) / np.abs(max_eigval)

    # Make 3D plot illustrating the trapping region
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(8, 8))
    Y = np.zeros(x_test_pred.shape)
    Y = x_test_pred - opt_m * np.ones(x_test_pred.shape)

    Y = np.dot(eigvecs, Y.T).T
    plt.plot(Y[:, 0], Y[:, 1], Y[:, -1], 'k',
             label='SINDy model prediction with new initial condition',
             alpha=1.0, linewidth=3)
    h = np.dot(eigvecs, d)

    alpha = np.zeros(r)
    for i in range(r):
        if filename == 'Von Karman' and (i == 2 or i == 3):
            h[i] = 0
        alpha[i] = np.sqrt(0.5) * np.sqrt(np.sum(h ** 2 / eigvals) / eigvals[i])

    shift_orig = h / (4.0 * eigvals)

    # draw sphere in eigencoordinate space, centered at 0
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    x = Rm * np.cos(u) * np.sin(v)
    y = Rm * np.sin(u) * np.sin(v)
    z = Rm * np.cos(v)

    ax.plot_wireframe(x, y, z, color="b",
                      label=r'Trapping region estimate, $B(m, R_m)$',
                      alpha=0.5, linewidth=0.5)
    ax.plot_surface(x, y, z, color="b", alpha=0.05)
    ax.view_init(elev=0., azim=30)

    # define ellipsoid
    rx, ry, rz = np.asarray([alpha[0], alpha[1], alpha[-1]])

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Add this piece so we can compare with the analytic Lorenz ellipsoid,
    # which is typically defined only with a shift in the "y" direction here.
    if filename == 'Lorenz Attractor':
        shift_orig[0] = 0
        shift_orig[-1] = 0
    x = rx * np.outer(np.cos(u), np.sin(v)) - shift_orig[0]
    y = ry * np.outer(np.sin(u), np.sin(v)) - shift_orig[1]
    z = rz * np.outer(np.ones_like(u), np.cos(v)) - shift_orig[-1]

    # Plot ellipsoid
    ax.plot_wireframe(x, y, z, rstride=5, cstride=5, color='r',
                      label='Ellipsoid of positive energy growth',
                      alpha=1.0, linewidth=0.5)

    if filename == 'Lorenz Attractor':
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0

        # define analytic ellipsoid in original Lorenz state space
        rx, ry, rz = [np.sqrt(beta * rho), np.sqrt(beta * rho ** 2), rho]

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # ellipsoid in (x, y, z) coordinate to -> shifted by m
        x = rx * np.outer(np.cos(u), np.sin(v)) - opt_m[0]
        y = ry * np.outer(np.sin(u), np.sin(v)) - opt_m[1]
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + rho - opt_m[-1]

        # Transform into eigencoordinate space
        xyz = np.tensordot(eigvecs, np.asarray([x, y, z]), axes=[1, 0])
        x = xyz[0, :, :]
        y = xyz[1, :, :]
        z = xyz[2, :, :]

        # Plot ellipsoid
        ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='g',
                          label=r'Lorenz analytic ellipsoid',
                          alpha=1.0, linewidth=1.5)

    # Adjust plot features and save
    plt.legend(fontsize=16, loc='upper left')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()
    plt.show()


# Plot errors between m_{k+1} and m_k and similarly for the model coefficients
def make_progress_plots(r, sindy_opt):
    W = np.asarray(sindy_opt.history_)
    M = np.asarray(sindy_opt.m_history_)
    dW = np.zeros(W.shape[0])
    dM = np.zeros(M.shape[0])
    for i in range(1,W.shape[0]):
        dW[i] = np.sum((W[i, :, :] - W[i - 1, :, :]) ** 2)
        dM[i] = np.sum((M[i, :] - M[i - 1, :]) ** 2)
    plt.figure()
    print(dW.shape, dM.shape)
    plt.semilogy(dW, label=r'Coefficient progress, $\|\xi_{k+1} - \xi_k\|_2^2$')
    plt.semilogy(dM, label=r'Vector m progress, $\|m_{k+1} - m_k\|_2^2$')
    plt.xlabel('Algorithm iterations', fontsize=16)
    plt.ylabel('Errors', fontsize=16)
    plt.legend(fontsize=14)
    PWeigs = np.asarray(sindy_opt.PWeigs_history_)
    plt.figure()
    for j in range(r):
        if np.all(PWeigs[:, j] > 0.0):
            plt.semilogy(PWeigs[:, j],
                         label=r'diag($P\xi)_{' + str(j) + str(j) + '}$')
        else:
            plt.plot(PWeigs[:, j],
                     label=r'diag($P\xi)_{' + str(j) + str(j) + '}$')
        plt.xlabel('Algorithm iterations', fontsize=16)
        plt.legend(fontsize=12)
        plt.ylabel(r'Eigenvalues of $P\xi$', fontsize=16)


# Plot first three modes in 3D for ground truth and SINDy prediction
def make_3d_plots(x_test, x_test_pred, filename):
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(8, 8))
    if filename == 'VonKarman':
        ind = -1
    else:
        ind = 2
    plt.plot(x_test[:, 0], x_test[:, 1], x_test[:, ind],
             'r', label='true x')
    plt.plot(x_test_pred[:, 0], x_test_pred[:, 1], x_test_pred[:, ind],
             'k', label='pred x')
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()
    plt.legend(fontsize=14)
    plt.show()


# Plot the SINDy fits of X and Xdot against the ground truth
def make_fits(r, t, xdot_test, xdot_test_pred, x_test, x_test_pred, filename):
    fig = plt.figure(figsize=(16, 8))
    spec = gridspec.GridSpec(ncols=2, nrows=r, figure=fig, hspace=0.0, wspace=0.0)
    for i in range(r):
        plt.subplot(spec[i, 0]) #r, 2, 2 * i + 2)
        plt.plot(t, xdot_test[:, i], 'r',
                 label=r'true $\dot{x}_' + str(i) + '$')
        plt.plot(t, xdot_test_pred[:, i], 'k--',
                 label=r'pred $\dot{x}_' + str(i) + '$')
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.legend(fontsize=12)
        if i == r - 1:
            plt.xlabel('t', fontsize=18)
        plt.subplot(spec[i, 1])
        plt.plot(t, x_test[:, i], 'r', label=r'true $x_' + str(i) + '$')
        plt.plot(t, x_test_pred[:, i], 'k--', label=r'pred $x_' + str(i) + '$')
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.legend(fontsize=12)
        if i == r - 1:
            plt.xlabel('t', fontsize=18)

    plt.show()


# Make Lissajou figures with ground truth and SINDy model
def make_lissajou(r, x_train, x_test, x_train_pred, x_test_pred, filename):
    fig = plt.figure(figsize=(8, 8))
    spec = gridspec.GridSpec(ncols=r, nrows=r, figure=fig, hspace=0.0, wspace=0.0)
    for i in range(r):
        for j in range(i, r):
            plt.subplot(spec[i, j])
            plt.plot(x_train[:, i], x_train[:, j],linewidth=1)
            plt.plot(x_train_pred[:, i], x_train_pred[:, j], 'k--', linewidth=1)
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                plt.ylabel(r'$x_' + str(i) + r'$', fontsize=18)
            if i == r - 1:
                plt.xlabel(r'$x_' + str(j) + r'$', fontsize=18)
        for j in range(i):
            plt.subplot(spec[i, j])
            plt.plot(x_test[:, j], x_test[:, i], 'r', linewidth=1)
            plt.plot(x_test_pred[:, j], x_test_pred[:, i], 'k--', linewidth=1)
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                plt.ylabel(r'$x_' + str(i) + r'$', fontsize=18)
            if i == r - 1:
                plt.xlabel(r'$x_' + str(j) + r'$', fontsize=18)
    plt.show()



##################################################################################
##################################################################################
##################################################################################

# Model parameters
parameters = {'N' : 4,
              'K' : 1,
              'I' : 1,
              'b' : 1,
              'c' : 1,
              'F' : 15
              }

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-15
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-10

# Seed for reproducibility
np.random.seed(1000)

# Special library ordering needed again if constraints are used
library_functions = [lambda : 1, lambda x: x, lambda x, y: x * y, lambda x: x ** 2]
library_function_names = [lambda : 1, lambda x: x, lambda x, y: x + y, lambda x: x + x]
sindy_library = ps.CustomLibrary(
    library_functions=library_functions,
    function_names=library_function_names
)

# Some global parameters
n_targets = parameters['N']    # = r = phase space size
n_features = int((n_targets**2 + 3*n_targets)/2 + 1) # = number of functions (=10 for n_targets=3)
feature_names = ['x{}'.format(i) for i in range(n_targets)]
noise = 0.0            # mean Gaussian
make_predictions = True # make predictions?
make_graphs = True     # make graphs?
slice_factor = 10  # 1/slice_factor is the portion of all loaded data we take to be our training data (e.g. the last 1/slice_factor of data)
sparsification_factor = 1  # 1/(slice_factor * sparsification_factor) is the portion of all loaded data we take to be our test data

# General SINDy parameters
threshold = 0.1
max_iter = 100
regularizer = 'l0'  # for regular and constrained SINDy (Trapping has 'l1' by default)

# Constrained SINDy parameters
nu = 0.01

# Trapping SINDy parameters
eta = 0.1
gamma = -1


################ Prepare input data ##########################################

# Load evaluated Lorenz05 model 3
with open('data/LM05_{}.pkl'.format(parameters), 'rb') as inp:
    L = pickle.load(inp) # Lorenz_model_3_evaluated
    L.print_parameters()

    eval_path_num = -1
    eval_path = L.evaluation_paths[eval_path_num]

    with open(eval_path, 'rb') as LM05_inp_eval:
        ev = pickle.load(LM05_inp_eval)
        print(ev)
        print('\n########################################\n')

        del LM05_inp_eval
        del L

# Save loaded data
t_all = ev.t
x_all = ev.y.T
print('Number of all loaded data points: {}'.format(len(t_all)))
print('dt of loaded samples: ', t_all[1] - t_all[0])

# make training and testing data by splitting all loaded data into chunks
dt = (t_all[1] - t_all[0]) * sparsification_factor
t_train = t_all[-int(len(t_all) / slice_factor) : : sparsification_factor]
x_train = x_all[-int(len(t_all) / slice_factor) : : sparsification_factor]
t_test = t_all[-2*int(len(t_all) / slice_factor) : -int(len(t_all) / slice_factor) : sparsification_factor]
x_test = x_all[-2*int(len(t_all) / slice_factor) : -int(len(t_all) / slice_factor) : sparsification_factor]
print('slice_factor: {}'.format(slice_factor))
print('sparsification_factor: {}'.format(sparsification_factor))
print('Number of training (= num. of test) data points: {} (= all loaded data points / (slice_factor * sparsification_factor))'.format(len(t_test)))

# add noise on data
x_train = x_train + np.random.normal(loc=0, scale=noise, size=x_train.shape)

###############################################################################

print('\n####################################################\n')
print('threshold: ', threshold)
print('nu: ', nu, '(determines the level of relaxation)')
print('noise: ', noise, '(mean gaussian)')
print('regularizer: ', regularizer)
print('dt: ', dt)
print('Num. of data points: ', int(len(t_train)))
print('\n####################################################\n')

##### UNCONSTRAINED SINDy (PolyLibrary) ###########
if False:

    print('SR3 model, no constraints:')

    opt = ps.SR3(threshold=threshold,
                 thresholder=regularizer)
    model = ps.SINDy(feature_names=feature_names,
                     optimizer=opt,
                     feature_library=sindy_library,
                     differentiation_method=ps.SmoothedFiniteDifference())
    model.fit(x_train, t=dt)
    model.print()

    if make_predictions:

        # make predictions
        Xi = model.coefficients().T
        xdot_test = model.differentiate(x_test, t=t_test)
        xdot_test_pred = model.predict(x_test)
        x_train_pred = model.simulate(x_train[0, :], t_train, integrator_kws=integrator_keywords)
        x_test_pred = model.simulate(x_test[0, :], t_test, integrator_kws=integrator_keywords)
        E_pred = np.linalg.norm(x_test - x_test_pred) / np.linalg.norm(x_test)

        if make_graphs:
            # plotting and analysis
            make_fits(n_targets, np.arange(0,len(xdot_test_pred))*dt, xdot_test, xdot_test_pred, x_test, x_test_pred, 'lorenz05')
            make_lissajou(n_targets, x_train, x_test, x_train_pred, x_test_pred, 'lorenz05')
            make_3d_plots(x_test, x_test_pred, 'lorenz05')

        print('Frobenius Error = ', E_pred)
        mean_val = np.mean(x_test_pred, axis=0)
        mean_val = np.sqrt(np.sum(mean_val ** 2))

        # # compute relative Frobenius error in the model coefficients
        # Xi_meanfield = np.zeros(Xi.shape)
        # Xi_meanfield[:n_targets, :n_targets] = np.asarray([[0.01, -1, 0], [1, 0.01, 0], [0, 0, -1]]).T
        # Xi_meanfield[n_targets + 1, 0] = -1
        # Xi_meanfield[n_targets + 2, 1] = -1
        # Xi_meanfield[n_targets + 3, 2] = 1
        # Xi_meanfield[n_targets + 4, 2] = 1
        # coef_pred = np.linalg.norm(Xi_meanfield - Xi) / np.linalg.norm(Xi_meanfield)
        # print('Frobenius coefficient error = ', coef_pred)

        # Compute time-averaged dX/dt error
        deriv_error = np.zeros(xdot_test.shape[0])
        for i in range(xdot_test.shape[0]):
            deriv_error[i] = np.dot(xdot_test[i, :] - xdot_test_pred[i, :],
                                    xdot_test[i, :] - xdot_test_pred[i, :]) / np.dot(
                xdot_test[i, :], xdot_test[i, :])
        print('Time-averaged derivative error = ', np.nanmean(deriv_error))

##### CONSTRAINED SINDy (energy preserving quadratic nonliniarities (feature) constraints, CostumLibrary) #############
if False:

    print(
        'Constrained SR3 model, energy preserving quadratic nonliniarities'
        ' (constraint_order="feature", library=CostumLibrary):')

    # Make constraints
    constraint_zeros, constraint_matrix = make_constraints(n_targets)

    opt = ps.ConstrainedSR3(constraint_rhs=constraint_zeros,
                            constraint_lhs=constraint_matrix,
                            threshold=threshold,
                            thresholder=regularizer,
                            constraint_order="feature",
                            nu=nu,
                            verbose=True,
                            max_iter=max_iter)
    model = ps.SINDy(feature_names=feature_names,
                     optimizer=opt,
                     feature_library=sindy_library,
                     differentiation_method=ps.SmoothedFiniteDifference())
    model.fit(x_train, t=dt)
    model.print()

    if make_predictions:

        # make predictions
        Xi = model.coefficients().T
        xdot_test = model.differentiate(x_test, t=t_test)
        xdot_test_pred = model.predict(x_test)
        x_train_pred = model.simulate(x_train[0, :], t_train, integrator_kws=integrator_keywords)
        x_test_pred = model.simulate(x_test[0, :], t_test, integrator_kws=integrator_keywords)
        E_pred = np.linalg.norm(x_test - x_test_pred) / np.linalg.norm(x_test)

        if make_graphs:
            # plotting and analysis
            make_fits(n_targets, np.arange(0,len(xdot_test_pred))*dt, xdot_test, xdot_test_pred, x_test, x_test_pred, 'lorenz05')
            make_lissajou(n_targets, x_train, x_test, x_train_pred, x_test_pred, 'lorenz05')
            make_3d_plots(x_test, x_test_pred, 'lorenz05')

        print('Frobenius Error = ', E_pred)
        mean_val = np.mean(x_test_pred, axis=0)
        mean_val = np.sqrt(np.sum(mean_val ** 2))

        # # compute relative Frobenius error in the model coefficients
        # Xi_meanfield = np.zeros(Xi.shape)
        # Xi_meanfield[:n_targets, :n_targets] = np.asarray([[0.01, -1, 0], [1, 0.01, 0], [0, 0, -1]]).T
        # Xi_meanfield[n_targets + 1, 0] = -1
        # Xi_meanfield[n_targets + 2, 1] = -1
        # Xi_meanfield[n_targets + 3, 2] = 1
        # Xi_meanfield[n_targets + 4, 2] = 1
        # coef_pred = np.linalg.norm(Xi_meanfield - Xi) / np.linalg.norm(Xi_meanfield)
        # print('Frobenius coefficient error = ', coef_pred)

        # Compute time-averaged dX/dt error
        deriv_error = np.zeros(xdot_test.shape[0])
        for i in range(xdot_test.shape[0]):
            deriv_error[i] = np.dot(xdot_test[i, :] - xdot_test_pred[i, :],
                                    xdot_test[i, :] - xdot_test_pred[i, :]) / np.dot(
                xdot_test[i, :], xdot_test[i, :])
        print('Time-averaged derivative error = ', np.nanmean(deriv_error))


##### TRAPPING SINDy (energy preserving quadratic nonliniarities (featuer) constraints, CostumLibrary) #############
if True:

    print('Trapping SR3 model, energy preserving quadratic nonliniarities'
          ' (constraint_order="feature", library=CostumLibrary):')

    # Make constraints
    constraint_zeros, constraint_matrix = make_constraints(n_targets)

    opt = ps.TrappingSR3(constraint_rhs=constraint_zeros,
                         constraint_lhs=constraint_matrix,
                         threshold=threshold,
                         thresholder='l1',
                         constraint_order="feature",
                         gamma=gamma,
                         max_iter=max_iter,
                         verbose=True,
                            )
    model = ps.SINDy(feature_names=feature_names,
                     optimizer=opt,
                     feature_library=sindy_library,
                     differentiation_method=ps.SmoothedFiniteDifference())
    model.fit(x_train, t=dt)
    model.print()

    if make_predictions:

        # make predictions
        Xi = model.coefficients().T
        xdot_test = model.differentiate(x_test, t=t_test)
        xdot_test_pred = model.predict(x_test)
        x_train_pred = model.simulate(x_train[0, :], t_train, integrator_kws=integrator_keywords)
        x_test_pred = model.simulate(x_test[0, :], t_test, integrator_kws=integrator_keywords)
        E_pred = np.linalg.norm(x_test - x_test_pred) / np.linalg.norm(x_test)

        if make_graphs:
            # plotting and analysis
            make_fits(n_targets, np.arange(0,len(xdot_test_pred))*dt, xdot_test, xdot_test_pred, x_test, x_test_pred, 'lorenz05')
            make_lissajou(n_targets, x_train, x_test, x_train_pred, x_test_pred, 'lorenz05')
            make_3d_plots(x_test, x_test_pred, 'lorenz05')

        print('Frobenius Error = ', E_pred)
        mean_val = np.mean(x_test_pred, axis=0)
        mean_val = np.sqrt(np.sum(mean_val ** 2))
        check_stability(n_targets, Xi, opt, mean_val)

        # # compute relative Frobenius error in the model coefficients
        # Xi_meanfield = np.zeros(Xi.shape)
        # Xi_meanfield[:n_targets, :n_targets] = np.asarray([[0.01, -1, 0], [1, 0.01, 0], [0, 0, -1]]).T
        # Xi_meanfield[n_targets + 1, 0] = -1
        # Xi_meanfield[n_targets + 2, 1] = -1
        # Xi_meanfield[n_targets + 3, 2] = 1
        # Xi_meanfield[n_targets + 4, 2] = 1
        # coef_pred = np.linalg.norm(Xi_meanfield - Xi) / np.linalg.norm(Xi_meanfield)
        # print('Frobenius coefficient error = ', coef_pred)

        # Compute time-averaged dX/dt error
        deriv_error = np.zeros(xdot_test.shape[0])
        for i in range(xdot_test.shape[0]):
            deriv_error[i] = np.dot(xdot_test[i, :] - xdot_test_pred[i, :],
                                    xdot_test[i, :] - xdot_test_pred[i, :]) / np.dot(
                xdot_test[i, :], xdot_test[i, :])
        print('Time-averaged derivative error = ', np.nanmean(deriv_error))



