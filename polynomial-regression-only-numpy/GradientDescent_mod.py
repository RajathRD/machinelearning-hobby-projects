import numpy as np
import matplotlib.pyplot as plt

N = 100
num_w = 2 #6
# Gradient Descent Algorithm
def gradient_desc(x,w,y):
    rate = 0.000000001 # 9+1
    augmented_x = np.array([x**0])
    for i in range(1,num_w+1):
        augmented_x = np.vstack([augmented_x, x**i ])
    iterations = 1000000 #6

    for i in range(0,iterations):
        ybar = np.dot(w,augmented_x)
        w = w - ( rate * ( np.dot( ( ybar - y ),augmented_x.T ) )/ (2*N) )
    return w
# Generate Data
def yh(v0, theta, h0, t):
    return -4.9*t*t + v0*np.cos(np.radians(theta))*t + np.array([h0]*np.size(t))
    
def xh(v0, theta, t):
    return v0*np.sin(np.radians(theta))*t

initial_velocity = 100
theta = 45
initial_height = 100 

time = np.linspace(0, 10, num=100)
noise = np.random.normal(0,3, 100)
y = yh(initial_velocity, theta, initial_height, time) + noise
x = xh(initial_velocity, theta, time) + noise

w = np.random.normal(0, 1, num_w+1)
weights = gradient_desc(x,w,y)
print (weights)

augmented_x = np.array([x**0])
for i in range(1,num_w):
    augmented_x = np.vstack([augmented_x, x**i ])

print ("Weights: ",weights) # actual weights

prediction = np.dot(weights,augmented_x)
print ("Prediction: ",prediction[0]) # to check if NaN
plt.plot (x,y,x,prediction)
plt.show()
