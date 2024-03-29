import sub.minimization as mymin
import numpy as np

np.random.seed(0)
#np.random.seed(315054)

Np = 100
Nf = 4

A = np.random.randn(Np, Nf)
w = np.random.randn(Nf, 1)  # random column vector
print(f"Real value of w : {w}")

y = A@w

m = mymin.SolveLLS(y, A)
m.run()
m.print_result('LLS')
m.plot_w_hat('LLS')  # plot w_hat (inherited method)

Nit = 1000
gamma = 1e-3
g = mymin.SolveGrad(y, A)
g.run(gamma, Nit)
g.print_result('Gradient algorithm')
logx = 0
logy = 1
g.plot_err('Gradient algorithmn: square error', logy, logx)  # inherited method
g.plot_w_hat('Gradient algorithmn')  # inherited method

g = mymin.steepestDescentAlgorithm(y, A)
g.run()
g.print_result('Steepest descent: ')
g.plot_err('Steepest descent: square error', logy, logx)
g.plot_w_hat('Steepest descent')