import benders as bd
import numpy as np
import numpy.linalg as las


import time
def benders_solver(H, c, d, f, A, B, b, k, M, y):
    q = f.shape[0]

    p = bd.quad_function(H, c, d, f)

    sp = bd.primal_problem(A, B, b, M)

    # y = np.zeros((q, 1))

    upper_bound = 1e100
    lower_bound = - upper_bound

    benders_cuts = []
    benders_cuts_support = []
    iteration = 0
    eps = 1e-3
    max_iter = 500
    rel_error = 0.25
    rel_gap = 1
    Ub = []
    Lb = []
    master_time = []
    primal_time = []
    # while (rel_gap >= rel_error):
    while (upper_bound - lower_bound >= eps) and (iteration <= max_iter):
        iteration += 1
        start = time.time()
        x, Lambda, obj_value = sp.solve(p, y)
        ps = time.time() - start
        opt_cut, const_opt_cut = bd.cut_generation(x, Lambda, obj_value, p, sp)
        benders_cuts.append(opt_cut)
        benders_cuts_support.append(const_opt_cut)
        start = time.time()
        miu, y = bd.solve_master(p, k, benders_cuts, benders_cuts_support)
        ms = time.time() - start
        lower_bound = miu[0]
        upper_bound = min(obj_value, upper_bound)
        Ub.append(upper_bound)
        Lb.append(lower_bound)
        master_time.append(ms)
        primal_time.append(ps)
        rel_gap = abs(upper_bound - lower_bound)
        print('iter: {0:3d}  ub: {1:8.3f}  lb:{2:8.3f}  gap:{3:2.2f}  primal time: {4:3.3f}  master time: {5:3.3f}s  total: {6:3.3f}s'.format(iteration, upper_bound, lower_bound, rel_gap, ps, ms, ms + ps))
    x, Lambda, obj_value = sp.solve(p, y)
    solution = {'x': x, 'y': y, 'UB': Ub, 'LB': Lb, 'obj': upper_bound, 'ms': master_time, 'ps': primal_time}
    print('Benders was terminated successfully')
    return solution
