import benders as bd
import numpy as np
import numpy.linalg as la
import time
def outer_approx(H, c, d, f, A, B, b, k, M, y):
    q = f.shape[0]

    p = bd.quad_function(H, c, d, f)

    sp = bd.primal_problem(A, B, b, M)
    # y = np.zeros((q, 1))
    upper_bound = 1e100
    lower_bound = - upper_bound
    prim_sol = []
    cuts_f = []
    cuts_grad = []
    iteration = 0
    eps = 1e-3
    rel_error = 0.25
    rel_gap = 1
    max_iter = 500
    Ub = []
    Lb = []
    master_time = []
    primal_time = []
    eigen_values = la.eig(H)
    eig = eigen_values[0].min()
    FF = []
    alpha = 0.2



    # while (rel_gap >= rel_error):
    while (upper_bound - lower_bound >= eps) and (iteration <= max_iter):
        ff = alpha * lower_bound + (1- alpha) * upper_bound
        FF.append(ff)
        iteration += 1
        start = time.time()
        x, obj_value = sp.solve(p, y)
        ps = time.time() - start
        cuts_f.append(obj_value)
        cuts_grad.append(p.compute_gradient(x))
        prim_sol.append(x)
        start = time.time()
        miu, y = bd.solve_master_outer(p, sp, prim_sol, cuts_f, cuts_grad, k, eig,FF)
        ms = time.time() - start
        upper_bound = min(obj_value, upper_bound)
        lower_bound = miu
        Ub.append(upper_bound)
        Lb.append(lower_bound)
        master_time.append(ms)
        primal_time.append(ps)
        rel_gap =  abs(upper_bound - lower_bound)
        print('iter: {0:3d}  ub: {1:8.3f}  lb:{2:8.3f}  gap:{3:2.2f}  primal time: {4:3.3f}  master time: {5:3.3f}s  total: {6:3.3f}s'.format(iteration, upper_bound, lower_bound, rel_gap, ps, ms, ms + ps))
    x, Lambda, obj_value = sp.solve(p, y)
    solution = {'x': x, 'y': y, 'UB': Ub, 'LB': Lb, 'obj': upper_bound, 'ms': master_time, 'ps': primal_time}
    print('outer approximation algorithm was terminated successfully')

    return solution
