import numpy as np
import matplotlib.pylab as plt
import dump

NA = 6.023e23 # Avogadro's number
K = 1.3e-23 # Boltzmann constant

    
## Boundary condition to check if position does not exceed wall coordinates
def BoundaryCheck(pos,vel,box):
    ndims = len(box)
    for i in range(0,ndims):
        vel[(pos[:,i]<=box[i][0]) | (pos[:,i]>=box[i][1])]*=-1
        

## Integrating Langevin's equation (uses forward Euler Scheme)
def Integrate(pos,vel,mass,forces,dt):
    pos += vel*dt
    vel += forces*dt/mass[np.newaxis].T
        

## Force calculation on all atoms
def CalculateForces(vel,mass,temp,dt,relax):
    natoms,ndims = vel.shape
    sigma = np.sqrt(2*mass*K*temp/(dt*relax))
    noise = np.random.randn(natoms,ndims)*sigma[np.newaxis].T
    
    forces = -vel*mass[np.newaxis].T/relax + noise
    
    return forces
    




def run(**args):
    natoms,radius,mass,dt,relax = args['natoms'],args['radius'],args['mass'],args['dt'],args['relax']
    temp,nsteps,box,ofname,freq = args['temp'],args['nsteps'],args['box'],args['ofname'],args['freq']
    
    
    mass = np.ones(natoms)*mass/NA
    radius = np.ones(natoms)*radius
    ndims = len(box)
    
    pos = np.random.rand(natoms,ndims)
    vel = np.random.rand(natoms,ndims)
    
    for i in range(0,ndims):
        pos[:,i] = box[i][0] + (box[i][1]-box[i][0])*pos[:,i]
    
    steps = 0
    output = []
    
    while(steps<=nsteps):
        steps = steps+1
        
        forces = CalculateForces(vel,mass,temp,dt,relax)
        Integrate(pos,vel,mass,forces,dt)
        BoundaryCheck(pos,vel,box)
        
        inst_temp = np.sum(np.dot(mass,(vel - vel.mean(axis=0))**2))/(K*ndims*natoms)
        output.append([dt*steps,inst_temp])
        
        if (steps%freq==0):
            dump.WriteOutput(ofname,natoms,steps,box,radius=radius,pos=pos,v=vel)
        
    return np.array(output)
    


if __name__ == '__main__':
    ## Define parameters
    parameters = {
            'natoms' : 1000,
            'radius' : 120e-12,
            'mass' : 1e-3,
            'dt' : 1e-15,
            'relax': 1e-13,
            'temp' : 300,
            'nsteps' : 10000,
            'freq' : 100,
            'box' : ((0,1e-8),(0,1e-8),(0,1e-8)),
            'ofname' : 'traj-hydrogen.dump'
            }
    
    output = run(**parameters)
    
    plt.plot(output[:,0]*1e12, output[:,1])
    plt.xlabel('scaled time')
    plt.ylabel('Temperature of system')
    plt.title('Thermostat readings')
    plt.show()

