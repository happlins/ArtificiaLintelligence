import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-whitegrid")
fig = plt.figure()
ax = plt.axes()
x = np.linspace(0,10,1000)
ax.plot(x,np.sin(x))
ax.plot(x,np.cos(x))
plt.show()