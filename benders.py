import numpy as np
import picos as pic
import numpy.linalg as la


class quad_function(object):
    """
    Quadratic Function Class:       f(x) = 0.5 x'Hx + c'x + d + f'y,    x: continuous n dim,    y: binary q dim
    H: Hessian Matrix
    c = Gradient Vector for continuous variables
    d = Constant term
    f = Gradient Vector for binary variables
    """

    def __init__(self, H, c, d, f):
        self.H = H
        self.c = c
        self.d = d
        self.f = f
        self.n = H.shape[0]
        self.q = f.shape[0]

    def __str__(self):
        return "quadratic function:\n0.5 x'{}x + c'{} + d + f'{}".format(self.H, self.c, self.d, self.f)

    def compute_gradient(self, x):
        """
        compute gradient with respect to the continuous variable
        :param x: n*1 vector at which the gradient is computed
        :return: gradient vector
        """
        return self.H @ x + self.c

    def compute_function(self, x, y):
        """
        computes the value of the function at the given points x and y
        :param x: continuous vec
        :param y: binary vec
        :return: f(x)
        """
        f = 0.5 * x.T @ self.H @ x + self.c.T @ x + self.d + self.f.T @ y
        return f[0][0]

    def condition(self):
        """
        provides the condition number of the hessian matrix
        :return: condition number
        """
        return la.cond(self.H)


class primal_problem(object):

    def __init__(self, A, B, b, M):
        """
        solves the primal problem. g(x) = Ax - b
        :param A: constraint matrix for continuous vec
        :param B: constraint matrix for discrete vec
        :param b: constant vector b
        """
        self.A = A
        self.B = B * M
        self.b = b
        self.m = A.shape[0]

    def solve(self, fcn, y):
        problem = pic.Problem()
        x = problem.add_variable('x', fcn.n, vtype='continuous')
        const = []
        problem.set_objective('min', 0.5 * x.T * fcn.H * x + fcn.c.T * x + fcn.d)
        const.append(problem.add_constraint(self.A * x + self.B @ y <= self.b))
        # problem.set_option('solver', 'gurobi')
        d = problem.solve(verbosity = 0)
        x = np.array(list(d.primals.values())[0])
        # dual_var = np.array(d['duals'][0])
        return x.reshape(fcn.n, 1), fcn.compute_function(x, y)


def cut_generation(x, Lambda, obj, fcn, const_data):
    opt_cut = fcn.f.T + Lambda.T @ const_data.B
    const_opt_cut = + obj + Lambda.T @ (const_data.A @ x - const_data.b)

    return opt_cut, const_opt_cut


def solve_master(fcn, k, opt_cuts, opt_constant):
    const = []
    master = pic.Problem()

    miu = master.add_variable('miu', 1, vtype='continuous')
    y = master.add_variable('y', fcn.q, vtype='binary')

    master.set_objective('min', miu)

    for cuts in range(len(opt_cuts)):
        const.append(master.add_constraint(-miu + opt_cuts[cuts] * y <= - opt_constant[cuts]))
    const.append(master.add_constraint(np.ones((fcn.q, 1)).T * y <= k))
    # master.set_option('timelimit', 1)
    d = master.solve(verbose=0)
    miu = d['primals']['miu']
    y = np.array(d['primals']['y']).reshape(fcn.q, 1)
    return miu, y

def solve_master_outer(fcn, sp, xk, cuts_f, cuts_grad, k, eig,FF):
    P = pic.Problem()

    miu = P.add_variable('miu', 1, vtype = 'continuous')
    x = P.add_variable('x', fcn.n, vtype = 'continuous')
    y = P.add_variable('y', fcn.n, vtype='binary')
    kk = len(xk) - 1
    aa = xk[kk]
    # P.set_objective('min', miu * 0)
    P.set_objective('min', miu )
    const = []
    for cut in range(len(cuts_f)):

        const.append(P.add_constraint(miu >= cuts_f[cut] + cuts_grad[cut].T * x - cuts_grad[cut].T @ xk[cut]))
#         const.append(P.add_constraint(miu >= cuts_f[cut] + cuts_grad[cut].T * x - cuts_grad[cut].T @ xk[cut] + (x.T
#                                                                             - xk[cut].T) * (eig/2) * (x - xk[cut])))
    # const.append(P.add_constraint(miu <= FF[len(FF) - 1]))
    const.append(P.add_constraint(np.ones((fcn.q, 1)).T * y <= k))
    const.append(P.add_constraint(sp.A * x + sp.B * y <= sp.b))
    # P.set_option('timelimit'=  1)
    d = P.solve(verbose=0)
    miu = d.value
    y = np.array(list(d.primals.values())[2]).reshape(fcn.q, 1)
    return miu, y
