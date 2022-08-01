
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pysindy as ps
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec


# ignore user warnings
import warnings
warnings.filterwarnings("ignore")



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



def make_constraints_og(r):
    q = 0
    N = int((r ** 2 + 3 * r) / 2.0)
    p = r + r * (r - 1) + int(r * (r - 1) * (r - 2) / 6.0)
    constraint_zeros = np.zeros(p)
    constraint_matrix = np.zeros((p, r * N))

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


# Lorenz model
def lorenz(t, x, sigma=10, beta=2.66667, rho=28):
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]

##################################################################################
##################################################################################
##################################################################################

print('Original Lorenz63 model:')
print("(x)' = - 10 x + 10 y + 10")
print("(y)' = 28 x - 1 y - 1 x z")
print("(z)' = - 2.66667 z + 1 x y")

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

# Global parameters
n_targets = 3    # = r = phase space size
#n_features_og = int((n_targets**2 + 3*n_targets)/2) # = number of functions for library without cons. terms (=9 for n_targets=3)
n_features = int((n_targets**2 + 3*n_targets)/2 + 1) # = number of functions for library with cons. term (=10 for n_targets=3)
feature_names = ['x', 'y', 'z']

# SINDy parameters
threshold = 0.1
nu = 1
noise = 0
regularizer = 'l0'
max_iter = 1000
eta =  2e-1
gamma = -1


# make training and testing data
dt = 0.01
T = 100
t = np.arange(0, T, dt)
t_span = (t[0], t[-1])
x0_train = [5.94857223, -0.12379429, 31.24273752]
x_train = solve_ivp(lorenz, t_span, x0_train, t_eval=t, **integrator_keywords).y.T
x0_test = (np.random.rand(3) - 0.5) * 30
x_test = solve_ivp(lorenz, t_span, x0_test, t_eval=t, **integrator_keywords).y.T

# add noise on data
x_train = x_train + np.random.normal(loc=0, scale=noise, size=x_train.shape)

# make predictions?
make_predictions = True
# make graphs?
make_graphs = True

# verbose?
verbose = True


print('####################################################')
print('threshold: ', threshold)
print('nu: ', nu, '(determines the level of relaxation)')
print('noise: ', noise, '(mean gaussian)')
print('regularizer: ', regularizer)
print('dt: ', dt)
print('T: ', T, '(stopping time)')
print('Num. of data points: ', int(len(x_train)))
print('####################################################')

##### UNCONSTRAINED SINDy (PolyLibrary) ###########
if False:

    print('SR3 model, no constraints:')

    opt = ps.SR3(threshold=threshold,
                 thresholder=regularizer)
    model = ps.SINDy(feature_names=feature_names,
                     optimizer=opt,
                     feature_library=ps.PolynomialLibrary(degree=2))
    model.fit(x_train, t=dt)
    model.print()

    if make_predictions:

        # make predictions
        Xi = model.coefficients().T
        xdot_test = model.differentiate(x_test, t=t)
        xdot_test_pred = model.predict(x_test)
        x_train_pred = model.simulate(x_train[0, :], t, integrator_kws=integrator_keywords)
        x_test_pred = model.simulate(x_test[0, :], t, integrator_kws=integrator_keywords)

        if make_graphs:
            # plotting and analysis
            make_fits(n_targets, t, xdot_test, xdot_test_pred, x_test, x_test_pred, 'lorenz')
            make_lissajou(n_targets, x_train, x_test, x_train_pred, x_test_pred, 'lorenz')
            make_3d_plots(x_test, x_test_pred, 'lorenz')

##### CONSTRAINED SINDy (some arbitrary (target) constraints, PolyLibrary) #############
if False:

    print('Constrained SR3 model, some arbitrary equality constraints '
          '(constraint_order="target", , library=PolynomialLibrary(degree=2)):')

    # xi matrix -> vectorized (constraint_order="target", library=PolynomialLibrary(degree=2))
    # n_targets = 3
    #
    #  x'   y'   z' | feature_num x   feature_num y    feature_num z
    # --------------|------------------------------------------------
    #  1    1    1  |      0          n_features + 0    2*n_features + 0
    #  x    x    x  |      1          n_features + 1    2*n_features + 1
    #  y    y    y  |      2          n_features + 2    2*n_features + 2
    #  z    z    z  |      3          n_features + 3    2*n_features + 3
    #  x2   x2   x2 |      4          n_features + 4    2*n_features + 4
    #  xy   xy   xy |      5          n_features + 5    2*n_features + 5
    #  xz   xz   xz |      6          n_features + 6    2*n_features + 6
    #  y2   y2   y2 |      7          n_features + 7    2*n_features + 7
    #  yz   yz   yz |      8          n_features + 8    2*n_features + 8
    #  z2   z2   z2 |      9          n_features + 9    2*n_features + 9


    # Define constraints
    constraint_rhs = np.asarray([0, 1, 28])
    constraint_lhs = np.zeros((3, n_targets*n_features))

    # Coefficients of x and y in the first equation (for x') must be opposite equal
    constraint_lhs[0, 1 + 0*n_features] = 1
    constraint_lhs[0, 2 + 0*n_features] = 1

    # Coefficient of xy in the third equation must be = 1
    constraint_lhs[1, 5 + 2*n_features] = 1

    # Coefficient of xz in the second equation must be = 1
    constraint_lhs[2, 6 + 1*n_features] = 1


    opt = ps.ConstrainedSR3(constraint_rhs=constraint_rhs,
                            constraint_lhs=constraint_lhs,
                            threshold=threshold,
                            thresholder=regularizer,
                            constraint_order="target",
                            nu=nu)
    model = ps.SINDy(feature_names=feature_names,
                     optimizer=opt,
                     feature_library=ps.PolynomialLibrary(degree=2))
    model.fit(x_train, t=dt)
    model.print()

    if make_predictions:

        # make predictions
        Xi = model.coefficients().T
        xdot_test = model.differentiate(x_test, t=t)
        xdot_test_pred = model.predict(x_test)
        x_train_pred = model.simulate(x_train[0, :], t, integrator_kws=integrator_keywords)
        x_test_pred = model.simulate(x_test[0, :], t, integrator_kws=integrator_keywords)

        if make_graphs:
            # plotting and analysis
            make_fits(n_targets, t, xdot_test, xdot_test_pred, x_test, x_test_pred, 'lorenz')
            make_lissajou(n_targets, x_train, x_test, x_train_pred, x_test_pred, 'lorenz')
            make_3d_plots(x_test, x_test_pred, 'lorenz')

##### CONSTRAINED SINDy (some arbitrary (feature) constraints, PolyLibrary) #############
if False:

    print('Constrained SR3 model, some arbitrary equality constraints '
          '(constraint_order="feature", library=PolynomialLibrary(degree=2)):')

    # xi matrix -> vectorized (constraint_order="feature", library=PolynomialLibrary(degree=2)))
    # n_targets = 3
    #
    #  x'   y'   z' | feature_num x     feature_num y     feature_num z
    # --------------|------------------------------------------------
    #  1    1    1  | 0                 1                 2
    #  x    x    x  | 0 + n_targets     1 + n_targets     2 + n_targets
    #  y    y    y  | 0 + 2*n_targets   1 + 2*n_targets   2 + 2*n_targets
    #  z    z    z  | 0 + 3*n_targets   1 + 3*n_targets   2 + 3*n_targets
    #  x2   x2   x2 | 0 + 4*n_targets   1 + 4*n_targets   2 + 4n_targets
    #  xy   xy   xy | 0 + 5*n_targets   1 + 5*n_targets   2 + 5n_targets
    #  xz   xz   xz | 0 + 6*n_targets   1 + 6*n_targets   2 + 6n_targets
    #  y2   y2   y2 | 0 + 7*n_targets   1 + 7*n_targets   2 + 7n_targets
    #  yz   yz   yz | 0 + 7*n_targets   1 + 7*n_targets   2 + 7n_targets
    #  z2   z2   z2 | 0 + 9*n_targets   1 + 9*n_targets   2 + 9n_targets


    # Define constraints
    constraint_rhs = np.asarray([0, 1, 28])
    constraint_lhs = np.zeros((3, n_targets*n_features))

    # Coefficients of x and y in the first equation (for x') must be opposite equal
    constraint_lhs[0, 1*n_targets + 0] = 1
    constraint_lhs[0, 2*n_targets + 0] = 1

    # Coefficient of xy in the third equation must be = 1
    constraint_lhs[1, 5*n_targets + 2] = 1

    # Coefficient of xz in the second equation must be = 1
    constraint_lhs[2, 6*n_targets + 1] = 1


    opt = ps.ConstrainedSR3(constraint_rhs=constraint_rhs,
                            constraint_lhs=constraint_lhs,
                            threshold=threshold,
                            thresholder=regularizer,
                            constraint_order="feature",
                            nu=nu)
    model = ps.SINDy(feature_names=feature_names,
                     optimizer=opt,
                     feature_library=ps.PolynomialLibrary(degree=2))
    model.fit(x_train, t=dt)
    model.print()

    if make_predictions:
        # make predictions
        Xi = model.coefficients().T
        xdot_test = model.differentiate(x_test, t=t)
        xdot_test_pred = model.predict(x_test)
        x_train_pred = model.simulate(x_train[0, :], t, integrator_kws=integrator_keywords)
        x_test_pred = model.simulate(x_test[0, :], t, integrator_kws=integrator_keywords)

        if make_graphs:
            # plotting and analysis
            make_fits(n_targets, t, xdot_test, xdot_test_pred, x_test, x_test_pred, 'lorenz')
            make_lissajou(n_targets, x_train, x_test, x_train_pred, x_test_pred, 'lorenz')
            make_3d_plots(x_test, x_test_pred, 'lorenz')

##### CONSTRAINED SINDy (some arbitrary (feature) constraints, CostumLibrary) #############
if False:

    print('Constrained SR3 model, some arbitrary equality constraints '
          '(constraint_order="feature", library=PolynomialLibrary(degree=2)):')

    # xi matrix -> vectorized (constraint_order="feature", library=PolynomialLibrary(degree=2)))
    # n_targets = 3
    #
    #  x'   y'   z' | feature_num x     feature_num y     feature_num z
    # --------------|------------------------------------------------
    #  1    1    1  | 0                 1                 2
    #  x    x    x  | 0 + n_targets     1 + n_targets     2 + n_targets
    #  y    y    y  | 0 + 2*n_targets   1 + 2*n_targets   2 + 2*n_targets
    #  z    z    z  | 0 + 3*n_targets   1 + 3*n_targets   2 + 3*n_targets
    #  xy   xy   xy | 0 + 4*n_targets   1 + 4*n_targets   2 + 4*n_targets
    #  xz   xz   xz | 0 + 5*n_targets   1 + 5*n_targets   2 + 5*n_targets
    #  yz   yz   yz | 0 + 6*n_targets   1 + 6*n_targets   2 + 6*n_targets
    #  x2   x2   x2 | 0 + 7*n_targets   1 + 7*n_targets   2 + 7*n_targets
    #  y2   y2   y2 | 0 + 7*n_targets   1 + 7*n_targets   2 + 7*n_targets
    #  z2   z2   z2 | 0 + 9*n_targets   1 + 9*n_targets   2 + 9*n_targets


    # Define constraints
    constraint_rhs = np.asarray([1, 1, 1])
    constraint_lhs = np.zeros((3, n_targets*n_features))

    # Coefficients of x and y in the first equation (for x') must be opposite equal
    constraint_lhs[0, 0 + 3*7] = 1
    constraint_lhs[1, 1 + 3*8] = 1
    constraint_lhs[2, 2 + 3*9] = 1
    # constraint_lhs[0, 4 * n_targets + 0] = 1
    # constraint_lhs[0, 5 * n_targets + 0] = 1

    # # Coefficient of xy in the third (2) equation must be = 1
    # constraint_lhs[1, 4*n_targets + 2] = 1
    #
    # # Coefficient of xz in the second (1) equation must be = 1
    # constraint_lhs[2, 5*n_targets + 1] = 1


    opt = ps.ConstrainedSR3(constraint_rhs=constraint_rhs,
                            constraint_lhs=constraint_lhs,
                            threshold=threshold,
                            thresholder=regularizer,
                            constraint_order="feature",
                            nu=nu)
    model = ps.SINDy(feature_names=feature_names,
                     optimizer=opt,
                     feature_library=sindy_library)
    model.fit(x_train, t=dt)
    model.print()

    if make_predictions:

        # make predictions
        Xi = model.coefficients().T
        xdot_test = model.differentiate(x_test, t=t)
        xdot_test_pred = model.predict(x_test)
        x_train_pred = model.simulate(x_train[0, :], t, integrator_kws=integrator_keywords)
        x_test_pred = model.simulate(x_test[0, :], t, integrator_kws=integrator_keywords)

        if make_graphs:
            # plotting and analysis
            make_fits(n_targets, t, xdot_test, xdot_test_pred, x_test, x_test_pred, 'lorenz')
            make_lissajou(n_targets, x_train, x_test, x_train_pred, x_test_pred, 'lorenz')
            make_3d_plots(x_test, x_test_pred, 'lorenz')

##### CONSTRAINED SINDy (energy preserving quadratic nonliniarities (feature) constraints, CostumLibrary) #############
if False:

    print('Constrained SR3 model, energy preserving quadratic nonliniarities '
          '(constraint_order="feature", library=CostumLibrary):')

    # xi matrix -> vectorized (constraint_order="feature", library=CostumLibrary))
    # n_targets = 3
    #
    #  x'   y'   z' | feature_num x     feature_num y     feature_num z
    # --------------|------------------------------------------------
    #  1    1    1  | 0                 1                 2
    #  x    x    x  | 0 + n_targets     1 + n_targets     2 + n_targets
    #  y    y    y  | 0 + 2*n_targets   1 + 2*n_targets   2 + 2*n_targets
    #  z    z    z  | 0 + 3*n_targets   1 + 3*n_targets   2 + 3*n_targets
    #  xy   xy   xy | 0 + 4*n_targets   1 + 4*n_targets   2 + 4*n_targets
    #  xz   xz   xz | 0 + 5*n_targets   1 + 5*n_targets   2 + 5*n_targets
    #  yz   yz   yz | 0 + 6*n_targets   1 + 6*n_targets   2 + 6*n_targets
    #  x2   x2   x2 | 0 + 7*n_targets   1 + 7*n_targets   2 + 7*n_targets
    #  y2   y2   y2 | 0 + 7*n_targets   1 + 7*n_targets   2 + 7*n_targets
    #  z2   z2   z2 | 0 + 9*n_targets   1 + 9*n_targets   2 + 9*n_targets


    # Define constraints
    constraint_zeros, constraint_matrix = make_constraints(n_targets)

    opt = ps.ConstrainedSR3(constraint_rhs=constraint_zeros,
                            constraint_lhs=constraint_matrix,
                            threshold=threshold,
                            thresholder=regularizer,
                            constraint_order="feature",
                            nu=nu)
    model = ps.SINDy(feature_names=feature_names,
                     optimizer=opt,
                     feature_library=sindy_library)
    model.fit(x_train, t=dt)
    model.print()

    if make_predictions:

        # make predictions
        Xi = model.coefficients().T
        xdot_test = model.differentiate(x_test, t=t)
        xdot_test_pred = model.predict(x_test)
        x_train_pred = model.simulate(x_train[0, :], t, integrator_kws=integrator_keywords)
        x_test_pred = model.simulate(x_test[0, :], t, integrator_kws=integrator_keywords)

        if make_graphs:
            # plotting and analysis
            make_fits(n_targets, t, xdot_test, xdot_test_pred, x_test, x_test_pred, 'lorenz')
            make_lissajou(n_targets, x_train, x_test, x_train_pred, x_test_pred, 'lorenz')
            make_3d_plots(x_test, x_test_pred, 'lorenz')

##### TRAPPING SINDy (energy preserving quadratic nonliniarities (feature) constraints, CostumLibrary) #############
if True:

    print('Trapping SR3 model, energy preserving quadratic nonliniarities '
          '(constraint_order="feature", library=CostumLibrary):')

    # xi matrix -> vectorized (constraint_order="feature", library=CostumLibrary))
    # n_targets = 3
    #
    #  x'   y'   z' | feature_num x     feature_num y     feature_num z
    # --------------|------------------------------------------------
    #  1    1    1  | 0                 1                 2
    #  x    x    x  | 0 + n_targets     1 + n_targets     2 + n_targets
    #  y    y    y  | 0 + 2*n_targets   1 + 2*n_targets   2 + 2*n_targets
    #  z    z    z  | 0 + 3*n_targets   1 + 3*n_targets   2 + 3*n_targets
    #  xy   xy   xy | 0 + 4*n_targets   1 + 4*n_targets   2 + 4*n_targets
    #  xz   xz   xz | 0 + 5*n_targets   1 + 5*n_targets   2 + 5*n_targets
    #  yz   yz   yz | 0 + 6*n_targets   1 + 6*n_targets   2 + 6*n_targets
    #  x2   x2   x2 | 0 + 7*n_targets   1 + 7*n_targets   2 + 7*n_targets
    #  y2   y2   y2 | 0 + 7*n_targets   1 + 7*n_targets   2 + 7*n_targets
    #  z2   z2   z2 | 0 + 9*n_targets   1 + 9*n_targets   2 + 9*n_targets


    # Define constraints


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
    model.fit(x_train, t=dt, quiet=True)
    model.print()

    if make_predictions:

        # make predictions
        Xi = model.coefficients().T
        xdot_test = model.differentiate(x_test, t=t)
        xdot_test_pred = model.predict(x_test)
        x_train_pred = model.simulate(x_train[0, :], t, integrator_kws=integrator_keywords)
        x_test_pred = model.simulate(x_test[0, :], t, integrator_kws=integrator_keywords)

        if make_graphs:
            # plotting and analysis
            make_fits(n_targets, t, xdot_test, xdot_test_pred, x_test, x_test_pred, 'lorenz')
            make_lissajou(n_targets, x_train, x_test, x_train_pred, x_test_pred, 'lorenz')
            make_3d_plots(x_test, x_test_pred, 'lorenz')

        E_pred = np.linalg.norm(x_test - x_test_pred) / np.linalg.norm(x_test)
        print('Frobenius Error = ', E_pred)
        mean_val = np.mean(x_test_pred, axis=0)
        mean_val = np.sqrt(np.sum(mean_val ** 2))
        check_stability(n_targets, Xi, opt, mean_val)

        # compute relative Frobenius error in the model coefficients
        Xi_meanfield = np.zeros(Xi.shape)
        Xi_meanfield[:n_targets, :n_targets] = np.asarray([[0.01, -1, 0], [1, 0.01, 0], [0, 0, -1]]).T
        Xi_meanfield[n_targets + 1, 0] = -1
        Xi_meanfield[n_targets + 2, 1] = -1
        Xi_meanfield[n_targets + 3, 2] = 1
        Xi_meanfield[n_targets + 4, 2] = 1
        coef_pred = np.linalg.norm(Xi_meanfield - Xi) / np.linalg.norm(Xi_meanfield)
        print('Frobenius coefficient error = ', coef_pred)

        # Compute time-averaged dX/dt error
        deriv_error = np.zeros(xdot_test.shape[0])
        for i in range(xdot_test.shape[0]):
            deriv_error[i] = np.dot(xdot_test[i, :] - xdot_test_pred[i, :],
                                    xdot_test[i, :] - xdot_test_pred[i, :])  / np.dot(
                                    xdot_test[i, :], xdot_test[i, :])
        print('Time-averaged derivative error = ', np.nanmean(deriv_error))





























































def calculate_mutual_information(time_delays, partition_num=30):
    # Mutual information as a function of time delay. Values in an array time_delays should be
    # some multiples of dt from provided evaluation. A scalar function that is used in the process
    # is the average of all components at one time instance


    x0 = np.sum(x_train, axis=-1) / 3 # average of all components at a time instant

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
        if verbose:
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

    print({'time_delays' : time_delays,
           'mutual_info' : mut_inf,
           'partition_num' : partition_num,
           })

    plt.plot(time_delays, mut_inf)
    plt.show()


def calculate_FNN(T, Dt, R_tol=10, A_tol=1, time_units_apart = None, td_dim_span=(None, None)):
        # Calculates False Nearest Neighbour function. Par. T is the time lag used to construct time-delay
        # coordinates, Dt is the spacing in time when delay coordinates are constructed and td_dim is a tuple (min, max)
        # dimension of time delay coordinates vector space (where we estimate FNN). Both T as delta_t should be
        # multiples of provided dt (from eval), and aldo T should be a multiple of delta_t. Param time_units_apart
        # defines the number of Dt units two compared points should be apart when finding a NN. It is set by default
        # such that two points can be at min 10 time unit close when searching for NN. A scalar function that is used in the
        # process is the average of all components at one time instance. td_dim_span[0]=td_dim_min must be at least 1.
        # R_tol and A_tol are method intrinsic parameters (Determining embedding dimension for phase-space
        # reconstruction using a geometrical construction, M.b. Kennel, R. Brown, H.D.I. Abarbanel, 1992)


        # Set default value of td_dim
        if td_dim_span == (None, None):
            td_dim_min = 1
            td_dim_max = 3 + 3
        else:
            if td_dim_span[0] < 1:
                raise ValueError('Min time delay dimension cannot be smaller then 1')
            td_dim_min = td_dim_span[0]
            td_dim_max = td_dim_span[1] + 1

        if time_units_apart == None:
            time_units_apart = Dt


        x0 = x_train[0]
        print(x0)

        # for i in range(len(x0)):
        #     x0[i] += np.random.uniform()/100

        print('\nCalculating FNN. Parameters:')
        print('Time delay (T): ', T)
        print('dt from evaluation: ', dt)
        print('max time from evaluation: ', t[-1])
        print('new dt:=Dt: ', Dt)
        print('time_units_apart: ', time_units_apart)
        print('R_tol: ', R_tol)
        print('A_tol: ', A_tol)

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

        # Estimate the variance of the signal x
        var = np.var(x0)

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
                        c_new = np.linalg.norm(pi-pj)
                        if c_new < c:
                            c = c_new
                            NNj = j

                NN_ind_arr[i] = [i,NNj, c, 0]

            # Calculate the increase in dist. of every pair for dim. of time delay vector space d+1
            for NN_pair in NN_ind_arr:
                i, j = int(NN_pair[0]), int(NN_pair[1])
                NN_pair[3] = np.abs(TDdplus1[i] - TDdplus1[j])

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

            if verbose:
                print('Calculating for dim. {}/{}'.format(d, td_dim_max-1))

            NNd = get_NN(d)

            score = 0
            thr = 0

            NNdplus1_avr = 0
            R_avr = 0

            for i in range(l):

                R = R_tol * NNd[i,2]

                NNdplus1_avr += NNd[i,3]
                R_avr += R

                threshold = np.max([R, A])
                if threshold == R:
                    thr += 1

                if NNd[i,3] > threshold:
                    score += 1

            NNdplus1_avr /= l
            R_avr /= l
            print('##################',NNdplus1_avr, R_avr, A)


            d_arr.append(d)
            fnn.append(score / l)
            threshold_choice.append(thr / l)

            if verbose:
                print('fnn(d): ', fnn[-1])
                print('threshold choice: ', threshold_choice[-1])

            # break loop if fnn(d) = 0
            if fnn[-1] == 0:
                break


        print({'d_arr' : np.asarray(d_arr),
               'fnn' : np.asarray(fnn),
               'thr_choice' : threshold_choice,
               'T' : T,
               'Dt' : Dt,
               'time_units_apart' : time_units_apart,
               'td_dim_span' : td_dim_span})


def calculate_FNN_met2(T, Dt, Dt_n_apart = None, td_dim_span=(None,None)):
        # Calculates False Nearest Neighbour function. Par. T is the time lag used to construct time-delay
        # coordinates, Dt is the spacing in time when delay coordinates are constructed and td_dim is a tuple (min, max)
        # dimension of time delay coordinates vector space (where we estimate FNN). Both T as delta_t should be
        # multiples of provided dt (from eval), and aldo T should be a multiple of delta_t. Param Dt_n_apart defines
        # the number of Dt units two compared points should be apart when finding a NN. It is set by default such that
        # two points can be at min 1 time unit close when searching for NN. A scalar function that is used in the
        # process is the average of all components at one time instance.

        # Set default value of td_dim
        if td_dim_span == (None, None):
            td_dim_min = 1
            td_dim_max = 3 + 2
        else:
            td_dim_min = td_dim_span[0]
            td_dim_max = td_dim_span[1]

        if Dt_n_apart == None:
            Dt_n_apart = 10

        x0 = np.sum(x_train, axis=-1) / 3 # average of all components at a time instant


        print('\nCalculating FNN. Parameters:')
        print('Time delay: ', T)
        print('dt from evaluation: ', dt)
        print('new dt:=Dt: ', Dt)

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


        # generate time delay vector function coordinates of dimension td_dim
        l = int((len(x0) - Ti * (td_dim_max - 1)) / Dti)
        print('Num.of points: ', l)
        TD = np.zeros((l, td_dim_max))
        for i in range(l):
            for j in range(td_dim_max):
                TD[i,j] = x0[i*Dti + j*Ti]


        def get_NN(d):
            # returns the indices of NN for dim. of delayed coord. space d

            NN_ind_arr = np.zeros((l, 2))
            TDd = TD[:,:d]

            for i, pi in enumerate(TDd):
                NNj = None
                c = 10**10  # initial distance to NN

                for j, pj in enumerate(TDd):
                    if np.abs(i - j) > Dt_n_apart:
                        c_new = np.linalg.norm(pi-pj)
                        if c_new < c:
                            c = c_new
                            NNj = j

                NN_ind_arr[i] = [i,NNj]

            return(NN_ind_arr)


        fnn = np.zeros(td_dim_max - td_dim_min)
        d_arr = np.arange(td_dim_min, td_dim_max)
        NNd = get_NN(td_dim_min)

        for d in d_arr:

            if verbose:
                print('Calculating for dim. {}/{}'.format(d, td_dim_max-1))

            NNdplus1 = get_NN(d+1)

            score = 0
            for i in range(l):
                if NNd[i,1] == NNdplus1[i,1]:
                    score += 1

            fnn[d - td_dim_min] = 1 - score / l
            if verbose:
                print('fnn(d): ', fnn[d - td_dim_min])

            NNd = NNdplus1

        print({'d_arr' : d_arr,
                        'fnn' : fnn,
                        'T' : T,
                        'Dt' : Dt,
                        'Dt_n_apart' : Dt_n_apart,
                        'td_dim' : td_dim_max})


# calculate_mutual_information(np.arange(0,5,0.01)) # 0.55

# calculate_FNN(0.55, 0.05)

#c alculate_FNN_met2(0.55, 0.05)


















