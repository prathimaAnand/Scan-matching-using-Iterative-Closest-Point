import numpy as np
import matplotlib.pyplot as plt

#Inputs
R = 0
t = 0
iterations = 30
dmax = 0.25
R0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
t0 = np.array([[0], [0], [0]])

#Calculating the correspondences
def EstimatePointCorrespondences(X, Y, t, R, dmax):
Correspondences = []
for i in range(len(X)):
x = X[i].reshape(3, 1)
y = (Y.T - (R.dot(x) + t))
ynorm = np.linalg.norm(y, axis=0)
yNormMin = np.argmin(ynorm)
if (ynorm[yNormMin] < dmax):
Correspondences.append((i, yNormMin))
return Correspondences

#Computing the Optimal Rigid Registration
def ComputeOptimalRigidRegistration(X, Y, C):
#calculating pointcloud centroids
xC = 0
yC = 0
#computing cross-covariance matrix W
W = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
sumX = np.matrix([0, 0, 0])
sumY = np.matrix([0, 0, 0])
J = len(C)
for m, n in C:
sumX = sumX + X[m]
sumY = sumY + Y[n]
xC = sumX / J
yC = sumY / J
for m, n in C:
xDev = X[m] - xC
yDev = np.transpose(Y[n] - yC)
W = W + yDev.dot(xDev)
W = W / J
U, S, Vt = np.linalg.svd(W)
#Construcing optimal rotation
rot = U.dot(Vt)
# recovering optimal translation
tran = yC.T - rot.dot(xC.T)
return tran, rot

def ICP(Xpts,Ypts,num_ICP_iters,dmax):
# Initialization
iterCount = 0
corres = 0
while iterCount < num_ICP_iters:
if (iterCount == 0):
corres = EstimatePointCorrespondences(Xpts,Ypts,t0,R0,dmax)
else:
corres = EstimatePointCorrespondences(Xpts,Ypts,t,R,dmax)
t, R = ComputeOptimalRigidRegistration(Xpts, Ypts, corres)
iterCount += 1
Xt = np.transpose(Xpts)
Yt = np.transpose(Ypts)
RMSE_total = 0
RMSE = 0
corres = EstimatePointCorrespondences(Xpts,Ypts,t,R,dmax)
for m, n in corres:
x = Xt[:, [m]]
y = Yt[:, [n]]
y_norm = np.linalg.norm(y - (R.dot(x) + t))
RMSE_total += y_norm
RMSE = np.sqrt((RMSE_total) / len(corres))
return (t, R, corres, RMSE)


XCloudSet= np.loadtxt("pclX.txt", delimiter = " ")
YCloudSet = np.loadtxt("pclY.txt", delimiter = " ")
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
xtranspose = XCloudSet.T
yTranspose = YCloudSet.T
finalTMat, finalRMat, finalC, RMSE = ICP(XCloudSet, YCloudSet, iterations, dmax)
# Parameters for the estimated rigid transformation
print('Rotational Matrix (R) is :\n {} '.format(finalRMat))
print('Translation Matrix (t) is :\n {} '.format(finalTMat))
newXCloudSet = 0
newXCloudSet = finalRMat.dot(xtranspose) + finalTMat
# Plotting the final co-registered pointclouds
ax.scatter3D(yTranspose[0], yTranspose[1], yTranspose[2],color = 'b' ,s=0.8)
ax.scatter3D(newXCloudSet[0], newXCloudSet[1], newXCloudSet[2], color = 'y',s=0.8)
ax.view_init(12, 11)
plt.show()
# RMSE for the estimated point correspondences
print('RMSE Error is: \n {} '.format(RMSE))

Output:
Rotational Matrix (R) is :
[[ 0.95126601 -0.15043058 -0.26919069]
[ 0.22323628 0.9381636 0.26460276]
[ 0.21274056 -0.31180074 0.92602471]]
Translation Matrix (t) is :
[[ 0.49661487]
[-0.29392971]
[ 0.29645004]]
