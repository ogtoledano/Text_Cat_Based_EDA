## Import the packages
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


grid=["0.1", "0.01", "0.001", "0.0001", "1e-05", "1e-06", "1e-07", "1e-08"]
accuracy_mean=[0.40194175, 0.93203883, 0.94368932, 0.38446602, 0.35339806,0.22330097, 0.13980583, 0.26213592]

plt.plot(grid,accuracy_mean,color="black")
plt.scatter(grid,accuracy_mean,color="black")
plt.xlabel("Learning rate")
plt.ylabel("Accuracy")
plt.title("Effects of learning rate in accuracy on BBC Sport dataset")
plt.grid(True)
plt.show()

grid=["0.1", "0.01", "0.001", "0.0001", "1e-05", "1e-06", "1e-07", "1e-08"]
accuracy_mean=[0.58812741, 0.91763769, 0.92931121, 0.78574652, 0.51822429,0.30327007, 0.38248441, 0.304]

plt.plot(grid,accuracy_mean,color="black")
plt.scatter(grid,accuracy_mean,color="black")
plt.xlabel("Learning rate")
plt.ylabel("Accuracy")
plt.title("Effects of learning rate in accuracy on YSC dataset")
plt.grid(True)
plt.show()


