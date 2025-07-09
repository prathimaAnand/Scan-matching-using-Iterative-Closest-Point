import numpy as np
import matplotlib.pyplot as plt

# Inputs
iterations = 30
dmax = 0.25
R0 = np.eye(3)
t0 = np.zeros((3, 1))

# Estimate point correspondences within distance threshold
def EstimatePointCorrespondences(X, Y, t, R, dmax):
    correspondences = []
    for i in range(len(X)):
        x = X[i].reshape(3, 1)
        y_diff = Y.T - (R @ x + t)
        ynorm = np.linalg.norm(y_diff, axis=0)
        yNormMin = np.argmin(ynorm)
        if ynorm[yNormMin] < dmax:
            correspondences.append((i, yNormMin))
    return correspondences

# Compute optimal rotation and translation using SVD
def ComputeOptimalRigidRegistration(X, Y, C):
    J = len(C)
    sumX = np.zeros((3,))
    sumY = np.zeros((3,))
    for m, n in C:
        sumX += X[m]
        sumY += Y[n]
    xC = sumX / J
    yC = sumY / J

    W = np.zeros((3, 3))
    for m, n in C:
        xDev = X[m] - xC
        yDev = (Y[n] - yC).reshape(3, 1)
        W += yDev @ xDev.reshape(1, 3)
    W /= J

    U, _, Vt = np.linalg.svd(W)
    R = U @ Vt
    t = yC.reshape(3, 1) - R @ xC.reshape(3, 1)
    return t, R

# ICP main function
def ICP(Xpts, Ypts, num_ICP_iters, dmax):
    t, R = t0.copy(), R0.copy()
    for _ in range(num_ICP_iters):
        corres = EstimatePointCorrespondences(Xpts, Ypts, t, R, dmax)
        t, R = ComputeOptimalRigidRegistration(Xpts, Ypts, corres)

    # Compute RMSE
    Xt = Xpts.T
    Yt = Ypts.T
    corres = EstimatePointCorrespondences(Xpts, Ypts, t, R, dmax)
    RMSE_total = 0
    for m, n in corres:
        x = Xt[:, [m]]
        y = Yt[:, [n]]
        diff = y - (R @ x + t)
        RMSE_total += np.linalg.norm(diff)
    RMSE = np.sqrt(RMSE_total / len(corres))
    return t, R, corres, RMSE

# Load point clouds from files
XCloudSet = np.loadtxt("pclX.txt", delimiter=" ")
YCloudSet = np.loadtxt("pclY.txt", delimiter=" ")

# Run ICP
finalTMat, finalRMat, finalC, RMSE = ICP(XCloudSet, YCloudSet, iterations, dmax)

# Transform XCloudSet using final R and t
xTranspose = XCloudSet.T
yTranspose = YCloudSet.T
newXCloudSet = finalRMat @ xTranspose + finalTMat

# Plot original and registered point clouds
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
ax.scatter3D(yTranspose[0], yTranspose[1], yTranspose[2], color='b', s=0.8, label='Target (Y)')
ax.scatter3D(newXCloudSet[0], newXCloudSet[1], newXCloudSet[2], color='y', s=0.8, label='Transformed Source (X)')
ax.view_init(12, 11)
ax.legend()
plt.show()

# Output final results
print('Rotational Matrix (R) is:\n', finalRMat)
print('Translation Matrix (t) is:\n', finalTMat)
print('RMSE Error is:\n', RMSE)
