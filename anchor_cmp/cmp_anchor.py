
import numpy as np

for i in range(5):
    dy_anchor_0 = np.load('dy_anchor/npy/anchor_out_{}_0.npy'.format(i))
    dy_anchor_1 = np.load('dy_anchor/npy/anchor_out_{}_1.npy'.format(i))
    #print('dy_anchor_0', dy_anchor_0.shape)
    
    st_anchor = np.load('../npy_anchor/anchor_anchor_generator_{}.tmp_0.npy'.format(i))
    
    #print('st_anchor', st_anchor.shape)
    
    print(np.sum(np.abs(st_anchor - dy_anchor_0)))