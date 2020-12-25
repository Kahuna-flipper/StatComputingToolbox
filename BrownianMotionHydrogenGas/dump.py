import numpy as np

def WriteOutput(ofname,natoms,steps,box,**data):
    
    with open(ofname,'a') as fw:
        
        fw.write('ITEM: TIMESTEP\n')
        fw.write('{}\n'.format(steps))
        
        fw.write('ITEM: NUMBER OF ATOMS\n')
        fw.write('{}\n'.format(natoms))
        
        fw.write('ITEM: BOX BOUNDS f f f\n')
        for b in box:
            fw.write('{} {}\n'.format(*b))
        
        axis = ('x','y','z')
        for i in range(len(axis)-len(box)):
            fw.write('0 0\n')
        
        
        keys = list(data.keys())
        
        
        for key in keys:
            ismatrix = len(data[key].shape)>1
            if ismatrix:
                rw,col = data[key].shape
                
                for i in range(0,col):
                    
                    if(key=='pos'):
                        data['{}'.format(axis[i])] = data[key][:,i]
                    else : 
                        data['{}_{}'.format(key,axis[i])] = data[key][:,i]
                        
                del data[key]
                
        keys = data.keys()
        
        fw.write('ITEM: ATOMS' + (' {} '*len(keys)).format(*keys) + '\n') 
        
        output = []
        
        for key in keys:
            output = np.hstack((output,data[key]))
            
        if len(output):
            np.savetxt(fw,output.reshape((natoms,len(data)),order='F'))
                        
                
                        
                   
                    
                
        