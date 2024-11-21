import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1, cache=True)
# X = mnist['data'].to_numpy()[:10000]
# y = mnist['target'].to_numpy(dtype=int)[:10000]

# Extract features and labels
X = mnist.data
y = mnist.target

# Normalize features to [0, 1]
X = X / 255.

# Split data into training and testing sets
split_idx = 1
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Define the harmonic coding function
def harmonic_coding(X, intermediate_dim):
    # Compute harmonic progression
    harmonic_progression = np.arange(1, intermediate_dim + 1)

    # Compute encoding matrix A
    A = np.zeros((intermediate_dim, X.shape[1]))
    for i in range(intermediate_dim):
        for j in range(X.shape[1]):
            A[i, j] = np.sin(np.pi * harmonic_progression[i] * (j + 1) / (X.shape[1] + 1))

    # Encode data
    encoded_data = np.dot(X, A.T)

    return encoded_data

# Define the harmonic decoding function
def harmonic_decoding(encoded_data, intermediate_dim):
    # Compute harmonic progression
    harmonic_progression =list(np.arange(1, intermediate_dim + 1))

    # Compute decoding matrix B
    B = np.zeros((encoded_data.shape[1], intermediate_dim))
    for i in range(encoded_data.shape[1]):
        for j in range(intermediate_dim):
            for m in range(intermediate_dim):
                if m != j:
                    B[i, j] = np.prod((harmonic_progression[m] - 1) / (harmonic_progression[m] - harmonic_progression[j]))

    # Decode data
    decoded_data = np.dot(encoded_data, B.T)

    return decoded_data

# Encode training data
intermediate_dim = 3
a = np.array([[1,2,3],[4,5,6]])
print(a.shape)
# encoded_train = harmonic_coding(X_train, intermediate_dim)
encoded_train = harmonic_coding(a, intermediate_dim)
print(1)

# Decode training data
decoded_train = harmonic_decoding(encoded_train, intermediate_dim)
print(decoded_train.shape)
print(decoded_train)
# decoded_train = decoded_train.reshape(X_train.shape)
print(2)

# Compute reconstruction error
# train_recon_error = np.mean(np.square(X_train - decoded_train))
train_recon_error = np.mean(np.square(a - decoded_train))
print('Train reconstruction error:', train_recon_error)
print(3)
quit()

idx = np.random.randint(0, X_train.shape[0])
fig, ax = plt.subplots(1, 2)
ax[0].imshow(X_train[idx].reshape(28, 28), cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(decoded_train[idx].reshape(28, 28), cmap='gray')
ax[1].set_title('Reconstructed')
plt.show()

# # Encode test data
# encoded_test = harmonic_coding(X_test, intermediate_dim)
#
# # Decode test data
# decoded_test = harmonic_decoding(encoded_test, intermediate_dim)
#
# # Compute reconstruction error
# test_recon_error = np.mean((X_test - decoded_test)**2)
# print('Test reconstruction error:', test_recon_error)

# Plot a random original and reconstructed digit from the test set
# idx = np.random.randint(0, X_test.shape[0])
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(X_test[idx].reshape(28, 28), cmap='gray')
# ax[0].set_title('Original')
# ax[1].imshow(decoded_test[idx].reshape(28, 28), cmap='gray')
# ax[1].set_title('Reconstructed')
# plt.show()