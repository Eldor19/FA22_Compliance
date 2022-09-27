import numpy as np
import fcm



indicator = np.loadtxt("indicator1(0).txt", delimiter=', ')
compliance = np.loadtxt("compliance1(0).txt", delimiter=', ')

for i in range(1, 18):
    indicator = np.append(indicator, np.loadtxt("indicator1("+str(i)+").txt", delimiter=', '), axis=0)
    compliance = np.append(compliance, np.loadtxt("compliance1("+str(i)+").txt", delimiter=', '), axis=0)
    


# recomputation of the compliance


DirichletVoxels = np.array([[1, 24], [47, 24]])
NeumannVoxelsX = np.array([])
NeumannVoxelsY = np.array([[24,24]])

n = 50

# boundary condition extraction from voxels
DirichletBCs = []
for iVoxel, jVoxel in DirichletVoxels:
    DirichletBCs += fcm.eft(iVoxel, jVoxel, n)
NeumannBCs = []
for iVoxel, jVoxel in NeumannVoxelsX:
    NeumannBCs += np.array(fcm.eft(iVoxel, jVoxel, n))[:8:2].tolist()
for iVoxel, jVoxel in NeumannVoxelsY:
    NeumannBCs += np.array(fcm.eft(iVoxel, jVoxel, n))[1:8:2].tolist()

NeumannBCs = np.transpose(np.concatenate((np.array(NeumannBCs),np.ones(len(NeumannBCs),dtype=int))).reshape((2,len(NeumannBCs)))).tolist()


# material parameters
E = 100. 
nu = 0.3
alpha = 1e-6 


complianceComputed = np.zeros(len(compliance))

for i in range(len(compliance)):
    indicatori = (indicator[i].reshape((n,n)))
    
    # assembly of fcm problem and solving
    K = fcm.globalStiffnessMatrix(E, nu, indicatori, alpha, n)
    K = fcm.applyHomogeneousDirichletBCs(K, DirichletBCs)
    F = fcm.globalForceVectorFromNeumannBCs(NeumannBCs, n)
    U = fcm.solve(K, F)

    complianceComputed[i] = np.transpose(F)@U


np.savetxt("compliance.txt", complianceComputed, newline='\n')
np.savetxt("indicator.txt", indicator, delimiter=', ')
      


#import matplotlib.pyplot as plt
#
#x = np.linspace(0,1,50)
#y = np.linspace(0,1,50)
#x, y = np.meshgrid(x, y)
    
#for i in range(110,130):
#    fig, ax = plt.subplots()
#    ax.pcolormesh(x, y, np.transpose(indicator[i].reshape((50,50))))
#    plt.gca().set_aspect('equal')
#    plt.show()





