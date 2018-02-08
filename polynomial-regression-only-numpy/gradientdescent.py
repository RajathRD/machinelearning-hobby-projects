import numpy as np
import matplotlib.pyplot as plt

N = 100
num_w = 4
# Gradient Descent Algorithm
def gradient_desc(x,w,y):
    rate = 0.00000001
    augmented_x = np.array([x**0])
    for i in range(1,num_w):
        augmented_x = np.vstack([augmented_x, x**i ])
    iterations = 100000000

    for i in range(0,iterations):
        ybar = np.dot(w,augmented_x)
        for j in range(0,num_w):
            dldw = np.sum ( (ybar - y ) * augmented_x[j] )
            w[j] = w[j] - rate * dldw

    return w


# Generate Data
samples = np.random.normal(0, 0.01, N)
x = np.linspace(0,2*np.pi,N)
y = np.sin(x/3) + np.cos(2*x)

w = np.ones(num_w)
weights = gradient_desc(x,w,y)

augmented_x = np.array([x**0])
for i in range(1,num_w):
    augmented_x = np.vstack([augmented_x, x**i ])

print ("Weights: ",weights) # actual weights

prediction = np.dot(weights,augmented_x)
print ("Prediction: ",prediction[0]) # to check if NaN
plt.plot (x,y,x,prediction)
plt.show()
