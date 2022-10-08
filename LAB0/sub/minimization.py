import numpy as np
import matplotlib.pyplot as plt


class SolveMinProbl:
    """
    This class is used to solve mi
    """

    def __init__(self, y=np.ones((3,)), A=np.eye(3)):
        self.matr = A
        self.NP = y.shape[0]
        self.Nf = A.shape[1]
        self.vect = y
        self.sol = np.zeros((self.Nf, ), dtype=float)
        return

    def plot_w_hat(self, title='Solution'):
        w_hat = self.sol
        n = np.arange(self.Nf)
        plt.figure()
        plt.plot(n, w_hat)
        plt.xlabel('n')
        plt.ylabel('$\hat{w}(n)$')
        plt.title('Title')
        plt.grid()
        plt.show()
        return

    def print_result(self, title):
        print(title, ': ')
        print('The optimum weight vector is: ')
        print(self.sol)
        return


class SolveLLS(SolveMinProbl):
    """
    This class is used to solve Linear Least Squares problems
    """

    def run(self):
        A = self.matr
        y = self.vect
        w_hat = np.linalg.inv(A.T@A)@(A.T@y)
        self.sol = w_hat
        self.min = np.linalg.norm(A@w_hat-y)**2
        return


class SolveGrad(SolveMinProbl):
    """
    -----------------------------------------------
    This class is used to solve iteratively 
    minimization problems using the gradient method
    -----------------------------------------------
    The termination condition is given by a maximum 
    number of iterations whose default value is 100
    -----------------------------------------------
    """

    def run(self, gamma=1e-3, Nit=100):
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        y = self.vect

        w = np.random.rand(self.Nf,)

        for it in range(Nit):
            grad = 2*A.T@(A@w - y)
            w = w - gamma*grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(A@w - y)**2

        self.sol = w
        self.min = self.err[it, 1]

    def plot_err(self, title='Square error', logy=0, logx=0):
        """
        This function allows to plot the error 
        value at each iteration of the gradient method
        """
        err = self.err
        plt.figure()

        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1])
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 1):
            plt.semilogx(err[:, 1], err[:, 1])
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.show()
        return


# %% This section will only run if the file 'minimization.py'
# is run "directly" (i.e., if it is called as % python3 minimization.py)

if __name__ == '__main__':
    Np = 100
    Nf = 4

    A = np.random.randn(Np, Nf)
    w = np.random.randn(Nf,)
    y = A@w
    m = SolveLLS(y, A)
    m.run()
    m.print_result('LLS')
    m.plot_w_hat('LLS')
