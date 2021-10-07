#%Positive
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
y = np.array([ 77.61,78.54,78.17,75.11,71.49,73.71,77.31,72.12,74.40,75.38,82.64,78.95])
plt.plot(x, y, 'o')
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)
plt.title("%Positive", size=20)
plt.xlabel("Months", size=13)
plt.ylabel("% of Reviews", size=13)

plt.show()