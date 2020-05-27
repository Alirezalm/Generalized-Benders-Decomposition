#from GBD import benders_solver
import numpy as np
import numpy.linalg as la
from outer_approx import outer_approx
import matplotlib.pyplot as plt
import picos as pic
import time
#import scipy.sparse as  spa
n = 30
m = 1000
k = 7
# n = 3
# m = 5

# data = [[0.5, 0.2, 0.9],[0.1, 0.5, 0.2], [0.5,0.8,0.35],[0.33,0.25,0.85], [0.38, 0.22, 0.14]]
# data = np.array(data)
# response = np.array([[0.22],[0.57],[0.26],[0.37],[0.55]])

q = n

data = np.random.randn(m, n)

response = np.random.rand(m, 1)
#
# data = spa.random(m,n,0.25)
# response = spa.random(m,1,0.30)
# data = data.todense()
# response = response.todense()
ans = la.lstsq(data, response, rcond=None)
M = la.norm(ans[0], np.inf)
H = data.T @ data

################


##################
# M = 0.005
c = - data.T @ response

f = np.zeros((q, 1))
d = 0.5 * response.T @ response

A = np.concatenate((np.eye(n, n), -np.eye(n, n)), axis=0)

B = -  np.concatenate((np.eye(n, n), np.eye(n, n)), axis=0)

b = np.zeros((A.shape[0], 1))
print('Commercial solver is starting...')

start_time = time.time()
BB = M * B
P = pic.Problem()
x = P.add_variable('x', n, 'continuous')
y = P.add_variable('y', n, 'binary')
P.set_objective('min', 0.5 * x.T * H * x + c.T * x + d + f.T * y)
const = [P.add_constraint(A * x + BB * y <= b), P.add_constraint(np.ones((n, 1)).T * y <= k)]
print(' ')
print('gurobi objective value: {}' .format(P.solve(verbose=1).value))
print('Problem was solved by gurobi')
print('====================================================')
print("time elapsed: {:.2f}s".format(time.time() - start_time))
print('====================================================')
print(' ')
a = list(ans[0])
aa = []
y = np.zeros((n, 1))
for i in range(k):
    kk = a.index(max(a))
    # a.pop(kk)
    a[kk] = -np.inf
    aa.append(kk)
    y[kk] = 1
# print('Benders is starting...')
# start_time = time.time()
# sol = benders_solver(H, c, d, f, A, B, b, k, M, y)
# print('benders objective value: {}' .format(sol['obj']))
# print('====================================================')
# print("time elapsed: {:.2f}s".format(time.time() - start_time))
# print('====================================================')
# ub = sol['UB']
# lb = sol['LB']
# master_time = sol['ms']
# primal_time = sol['ps']
print('outer approximation is starting...')
start_time = time.time()
sol = outer_approx(H, c, d, f, A, B, b, k, M, y)
print('outer approximation objective value: {}' .format(sol['obj']))
print('====================================================')
print("time elapsed: {:.2f}s".format(time.time() - start_time))
print('====================================================')
ub_oa = sol['UB']
lb_oa = sol['LB']
master_time_oa = sol['ms']
primal_time_oa = sol['ps']

# plt.plot(ub)
# plt.plot(lb)
# plt.title('Benders')
# plt.show()

plt.figure()
plt.plot(ub_oa)
plt.plot(lb_oa)
plt.title('outer approximation')
plt.show()

# plt.figure()
# plt.plot(master_time)
# plt.plot(primal_time)
# plt.title('benders')
# plt.legend(['master','primal'])
# plt.show()


plt.figure()
plt.plot(master_time_oa)
plt.plot(primal_time_oa)
plt.title('outer approximation')
plt.legend(['master','primal'])
plt.show()
#
#
