# Import numpy package
import numpy as np

# Create numpy.array from list
x = np.array([2, 5, 0])
print(x)

x = np.array([[2, 5, 0],
              [3, 5, 1],
              [3, 7, 8]])
print(x)

# Zero matrices
x = np.zeros(3)
print(x)

x = np.zeros((3, 3))
print(x)

# Matrices filled with 1
x = np.ones(3)
print(x)

x = np.ones((3, 3))
print(x)

# Identity matrix
x = np.eye(3)
print(x)

# Ranges
x = np.arange(10)
print(x)

x = np.arange(2, 10, dtype=np.float)
print(x)

x = np.linspace(1.0, 4.0, 10)
print(x)

# Random matrices
x = np.random.rand(3, 3)
print(x)

# Slicing
x = np.random.rand(3, 3)
print(x)
print(x[:, 0])
print(x[1, :])
x[:, 0] = x[:, 1]
print(x)

# Transposition
x = np.random.rand(3, 3)
print(x)
print(x.transpose())

# Save to file
x = np.random.rand(10, 2)
np.savetxt("x.csv", x, delimiter=',', fmt="%.5g")
x = np.loadtxt("x.csv", delimiter=',')
print(x)

# Elementwise operations
a = np.random.rand(3)
b = np.random.rand(3)
print(a)
print(b)
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(np.dot(a, b))

X = np.random.rand(3, 3)

print(X*a)
print(np.dot(X, a))

# Solve linear matrix equation
A = np.random.rand(3, 3)
b = np.random.rand(3)
x = np.linalg.solve(A, b)

print(np.dot(A, x) - b)

# Inverse matrix
A = np.random.rand(3, 3)
Ainv = np.linalg.inv(A)
print(np.dot(A, Ainv))