#Read bin type binary file
import numpy as np
import struct

def readp(filename):
        file=open(filename,'rb')
        frame=struct.unpack('i',file.read(4))[0]
        num_atoms=struct.unpack('i',file.read(4))[0]
        position=np.zeros(shape=[frame,num_atoms])
        for i in range(0,frame):
                for j in range(0,num_atoms):
                        position[i][j]=struct.unpack('f',file.read(4))[0]
        return frame,num_atoms,position
