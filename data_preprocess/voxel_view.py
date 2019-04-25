import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

voxels = np.load(os.path.join('../data/dataset/02691156_val_voxels.npz')).items()[0][1]

data = voxels[10].transpose((1,2,0))[::-1,:,:]

import binvox_rw

v = binvox_rw.Voxels(data, dims=[32,32,32],translate=[0,0,0],scale=1, axis_order='xyz')

with open('output.binvox','w',encoding='utf-8') as f:
    v.write(f)

print(data.shape)


ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程

for x in range(32):
    for y in range(32):
        for z in range(32):
            if data[x,y,z]:
                ax.scatter(x, y, z, c='y')  # 绘制数据点

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()