import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoLarsCV
from sklearn import preprocessing

# M: the number of MC replicates
# phi: the probability of dropping a feature in an observation. This is a tuning parameter for penalization / regularization.
# X: design matrix (observations / input matrix of model)
# y: output vector
def dropout_data(X, y, M, phi):
    def regularized_data():
        Z = np.random.binomial(n=1, p=1 - phi, size=X.shape)
        return np.multiply(X, Z)

    new_X = np.concatenate([regularized_data() for i in range(M)], axis=0) / (1 - phi)
    new_y = np.tile(y, M)
    return new_X, new_y


# M: the number of MC replicates
# sigma: standard deviation of Normally distributed noise. This is a tuning parameter for penalization / regularization.
# X: design matrix (observations / input matrix of model)
# y: output vector
def noise_data(X, y, M, sigma):
    def regularized_data():
        Z = np.random.normal(0, sigma, size=X.shape)
        return np.add(X, Z)

    new_X = np.concatenate([regularized_data() for i in range(M)], axis=0)
    new_y = np.tile(y, M)
    return new_X, new_y


# sample from d-dimensional ball
# r: radius
def sample_dball(d, r):
    norm = np.random.normal(0, 1, d + 2)
    standardized = norm / np.sum(norm ** 2) ** (0.5) * r
    return standardized[0:d]


# sample perturbation matrix delta from M in Robust method
# X: design matrix
# c: vector of column bounds for delta. For example, c1 is the norm bound for the first column of delta.
# S: number of perturbation matrix to generate
def sample_perturbed_data(X, c, S=1000):
    n, p = X.shape

    def sample_delta():
        M = np.empty(shape=(n, p))
        for i in range(p):
            M[:, i] = sample_dball(n, r=np.sqrt(c[i]))
        return M

    perturbed_data = [sample_delta() + X for i in range(S)]

    return perturbed_data


# c: vector of upper bounds for column norm of perturbation matrix M
# model: black box model
# perturbed_data: list of perturbed data X+delta generated from "sample_perturbed_data" function
# method: 1 to use method 1, 2 to use method 2
def worse_case_perturb_X(model, X, y, perturbed_data):
    max_loss = -1
    max_loss_pert_X = None
    blackbox = model(X, y)
    for each_X in perturbed_data:
        loss = np.sum((y - blackbox(each_X)) ** 2)
        if loss > max_loss:
            max_loss = loss
            max_loss_pert_X = each_X
    return max_loss_pert_X


# c: a vector (c1, c2, ..., cp) indicating avg perturbation to an observation of a predictor Xi. c_i * n is the total perturbation to a column of Xi
# model: black box model
# M: number of perturbation matrices (delta) to sample
# S: number of iterations
def robust_reg(c, model, X_train, y_train, M=2000, S=100):
    n, p = X_train.shape
    c_total = c * n
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    for i in range(1, S + 1):
        perturbed_data = sample_perturbed_data(X_train, c_total, S=M)
        worse_X = worse_case_perturb_X(model, X_train, y_train, perturbed_data)
        X_train = X_train + (worse_X - X_train) / i

    def scale_wrapper(X):
        X_scaled = scaler.transform(X)
        return model(X_train, y_train)(X_scaled)

    return scale_wrapper


# model: a function that takes in X & y as input, and return another function. The returned function takes in X and return predicted y.
# X: X_train, design matrix
# y: y_train, response
# M: number of replicates for Dropout & Noise Addition method. If "Robust", it is the number of deltas (perturbation matrices) sample
# M (continued): Recommend M>=1000 for "Dropout" & "Noise Addition"; M>=2000 for "Robust"
# cv_K: K for K-fold cross-validation in optimal parameter search
# reg_method: one of ("Dropout", "NoiseAddition" or "Robust")
# max_iter: maximum number of iterations for parameter search (phi for dropout, sigma2 for noise addition, c for robust)
# S: number of iterations for "Robust" regularisation method
def blackbox_tuner(model, X, y, M=1000, cv_K=5, metric='MSE', reg_method='dropout', max_iter=100, S=100):
    # Setups & sanity check
    reg_method = reg_method.lower()
    assert reg_method in ('dropout', 'noiseaddition',
                          'robust'), 'Regularization method is not implemented! Please use "Dropout", "NoiseAddition" or "Robust"'
    assert metric.lower() in (
    'mse', 'mae', 'mad'), 'Evaluation Metric not implemented! Please use "MSE" or "MAE"(or "MAD") '
    metric_func = mean_squared_error if metric.lower() == 'mse' else mean_absolute_error  # set metric function
    param_bound = 0.5 if reg_method == 'dropout' else (
        1 if reg_method == 'noiseaddition' else 0.05)  # upper bound of hyperparameters during tuning

    if reg_method in ('dropout', 'noiseaddition'):
        assert M is not None, "Please specify variable M for the number of Monte Carlo replicates!"
        X_mod_func = dropout_data if reg_method == 'dropout' else noise_data  # set the function to modify X

        # standardize X for noise injection
        if reg_method == 'noiseaddition':
            scaler = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)

        # K-fold CV loss
        def objective_func(param):
            kf = KFold(n_splits=cv_K, shuffle=True).split(X)
            fold_loss = list()
            for _, (train_index, test_index) in enumerate(kf):
                X_train = X[train_index, :]
                y_train = y[train_index]
                X_train, y_train = X_mod_func(X_train, y_train, M, param)

                X_test = X[test_index, :]
                y_test = y[test_index]

                fold_loss.append(metric_func(model(X_train, y_train)(X_test), y_test))
            return np.mean(fold_loss)

            # search for best parameters

        param_list = list()
        params_array = np.linspace(0.00001, param_bound, max_iter)
        for p in params_array:
            param_list.append(objective_func(p))
        best_param = params_array[np.array(param_list).argmin()]

        # modified datasets using the best parameter
        X_mod, y_mod = X_mod_func(X, y, M, best_param)

        # return trained model wrapperd in scaler
        if reg_method == 'noiseaddition':
            def scale_wrapper(X):
                X_scaled = scaler.transform(X)
                return model(X_mod, y_mod)(X_scaled)

            return scale_wrapper
        else:
            return model(X_mod, y_mod)
    else:
        assert M is not None, "Please specify M for the number of samples for perturbation delta!"

        # Scale X
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)

        # return approximation of worse case loss given a vector c
        def worse_case_loss(c, X_train, y_train, X_test, y_test):
            mod = robust_reg(c, model, X_train, y_train, M=M, S=S)
            worse_case_loss = metric_func(mod(X_test), y_test)
            return worse_case_loss

        # K-fold CV loss for a vector c: the function takes in c and return CV error
        def objective_func(c):
            kf = KFold(n_splits=cv_K, shuffle=True).split(X)
            fold_loss = list()
            for _, (train_index, test_index) in enumerate(kf):
                X_train = X[train_index, :]
                y_train = y[train_index]

                X_test = X[test_index, :]
                y_test = y[test_index]

                fold_loss.append(worse_case_loss(c, X_train, y_train, X_test, y_test))
            return np.mean(fold_loss)

        # search for best parameters, param (call it x) here refers to c = (x, x, ..., x). Ideally we should have different values for c_i
        # but it could take too long to compute
        n, p = X.shape
        param_list = list()  # record CV loss for a param x
        params_array = np.linspace(0.00001, param_bound, max_iter)  # upper bound is set to 0.1
        for p in params_array:
            param_c = np.array([p] * n)
            obj = objective_func(param_c)
            param_list.append(obj)
        best_param = params_array[np.array(param_list).argmin()]
        best_c = np.array([best_param] * n)
        best_mod = robust_reg(best_c, model, X, y, M=M, S=S)

        # wrap trained model in scaler
        def scale_wrapper(X):
            X_scaled = scaler.transform(X)
            return best_mod(X_scaled)

        return scale_wrapper


# Example wrapper for sklearn LR model to make it compatible with our "blackbox_tuner". It should be self-tuning as described by HW.
def linear_reg(X, y):
    reg = LinearRegression().fit(X, y)
    return reg.predict


# Example wrapper for sklearn LR model to make it compatible with our "blackbox_tuner". It should be self-tuning as described by HW.
def ridge_reg(X, y):
    reg = Ridge().fit(X, y)
    return reg.predict


# Example wrapper for sklearn LR model to make it compatible with our "blackbox_tuner". It should be self-tuning as described by HW.
def lasso(X, y):
    reg = LassoLarsCV().fit(X, y)
    return reg.predict
