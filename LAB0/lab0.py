import sub.minimization as mymin
import numpy as np

# np.random.seed(50)
np.random.seed(315054)

Np = 100
Nf = 4

A = np.random.randn(Np, Nf)
w = np.random.randn(Nf, 1)  # random column vector
y = A@w
m = mymin.SolveLLS(y, A)
m.run()
m.print_result('LLS')
m.plot_w_hat('LLS')  # plot w_hat (inherited method)

Nit = 1000
gamma = 1e-5
g = mymin.SolveGrand(y, A)
g.run(gamma, Nit)
g.print_result('Gradient algorithm')
logx = 0
logy = 1
g.plot_err('Gradient algorithmn: square error', logy, logx)  # inherited method
g.plot_w_hat('Gradient algorithmn')  # inherited method
