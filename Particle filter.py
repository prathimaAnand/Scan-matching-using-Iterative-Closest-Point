import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


vleft= 1.5
vright=2
rBot=0.25
wBot=0.5
stdVelL=0.05
stdVelR=0.05
stdMeasurement=0.1
velLPrime = []
velRPrime = []
numParticles = 1000


def ParticleFilterPropagate(X0,T0,T1):
num1 = (np.random.randn(numParticles)*stdVelL)
num2 = (np.random.randn(numParticles)*stdVelR)
velLPrime = vleft + num1
velRPrime = vright + num2
# Initialize empty particle set
Xt1 = np.zeros(shape=(3,numParticles),dtype=float)
for i in range (numParticles):
v1 = velRPrime[i] - velLPrime[i]
v2 = velRPrime[i] + velLPrime[i]
a = (rBot/wBot) * v1
b = 0.5 * rBot * v2
velMap = np.matrix([[0,-a,b],[a,0,0],[0,0,0]])
x = X0[0,i]
y = X0[1,i]
theta = X0[2,i]
xPoseInit = np.matrix([[np.cos(theta),-np.sin(theta),x],[np.sin(theta),np.cos(theta),y],[0,0,1]])
poseMat = xPoseInit.dot(scipy.linalg.expm(T1*velMap))
Xt1[0,i],Xt1[1,i],Xt1[2,i] = poseMat[0,2], poseMat[1,2], np.arccos(poseMat[0,0])
mean=np.mean(Xt1)
covariance=np.cov(Xt1)
# printing the Mean and Covariance values
print("Mean = ",mean)
print("Covariance = ",covariance)
return (Xt1)


def ParticleFilterUpdate(X1,Zt):
weights = np.ones(shape=(1,numParticles))
#Calculating importance weights
for k in range(numParticles):
diff_matrix = Zt - np.matrix([[X1[0,k]],[X1[1,k]]])
diffM = diff_matrix.T.dot(diff_matrix)
std = (-0.5/np.power(stdMeasurement,2))
sq = np.sqrt(2*np.pi)*stdMeasurement
weights[0,k] = (1/sq)*scipy.linalg.expm(std* diffM)
weights = weights/np.sum(weights)
pos = (np.arange(numParticles) + np.random.random()) / numParticles
indexes = np.zeros(numParticles,int)
cumulativeSum = np.cumsum(weights)
i = 0
j = 0
while i < numParticles and j < numParticles:
if pos[i] < cumulativeSum[j]:
indexes[i] = j
i += 1
else:
j += 1
X1[0,:] = X1[0,indexes]
X1[1,:] = X1[1,indexes]
X1[2,:] = X1[2,indexes]
#print(X1)
return(X1)


fig = plt.figure()
ax = plt.axes()
Value = ParticleFilterPropagate(np.zeros(shape=(3,numParticles),dtype=float),0,5)
ax.scatter(Value[0],Value[1])
Value = ParticleFilterUpdate(Value,np.matrix([[1.6561],[1.2847]]))
ax.scatter(Value[0],Value[1])
Value = ParticleFilterPropagate(np.zeros(shape=(3,numParticles),dtype=float),0,10)
ax.scatter(Value[0],Value[1])
Value = ParticleFilterUpdate(Value,np.matrix([[1.0505],[3.1059]]))
ax.scatter(Value[0],Value[1])
Value = ParticleFilterPropagate(np.zeros(shape=(3,numParticles),dtype=float),0,15)
ax.scatter(Value[0],Value[1])
Value = ParticleFilterUpdate(Value,np.matrix([[-0.9875],[3.2118]]))
ax.scatter(Value[0],Value[1])
Value = ParticleFilterPropagate(np.zeros(shape=(3,numParticles),dtype=float),0,20)
ax.scatter(Value[0],Value[1])
Value = ParticleFilterUpdate(Value,np.matrix([[-1.6450],[1.1978]]))
ax.scatter(Value[0],Value[1])
plt.show()



Output:
Mean = 1.3629707981546275
Covariance = [[ 0.01942727 -0.01583271 -0.02383308]
[-0.01583271 0.01600098 0.02179355]
[-0.02383308 0.02179355 0.0310925 ]]
Mean = 2.211068959551827
Covariance = [[ 0.37427348 0.047193 -0.19799047]
[ 0.047193 0.01945962 -0.02281638]
[-0.19799047 -0.02281638 0.10713777]]
Mean = 1.5970423375270797
Covariance = [[0.43364617 0.47324319 0.18276479]
[0.47324319 0.75265943 0.35627823]
[0.18276479 0.35627823 0.19027694]]
Mean = 0.4976193593063523
Covariance = [[ 0.35047701 -0.32835498 -0.23760601]
[-0.32835498 1.59366344 0.81731789]
[-0.23760601 0.81731789 0.44201675]]
