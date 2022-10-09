# LINEAR LEAST SQUARE
#%%
import numpy as np
Np = 5
Nf = 4
A = np.random.randn(Np,Nf)
w = np.random.randn(Nf,)
y = A@w     #Suppose we don't know y
#%%
ATA = A.T@A   #ATA (A transpose A)
ATAinv = np.linalg.inv(ATA)
ATy = A.T@y
w_hat = ATAinv@ATy

print("w_hat = ", w_hat)
print("True w = ", w)

e = y - A@w_hat
errsqnorm = np.linalg.norm(e)**2
print(e)
print("###########################")
print(errsqnorm)
# %%
#%% plot
import matplotlib.pyplot as plt
plt.figure()# create a new figure
plt.plot(w_hat, label='w_hat')# plot w_hat with a line
plt.plot(w,'o', label='w')# plot w with markers (circles)
plt.xlabel('n')# label on the x-axis
plt.ylabel('w(n)')# label on the y-axis
plt.legend()# show the legend
plt.grid() # show the grid
plt.title('Comparison between w and w_hat')# set the title
plt.show()# show the figure on the screen

#test github