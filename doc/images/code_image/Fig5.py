''' Plot karin noise for several SWH as a fonction of the across track direction '''
import matplotlib.pylab as plt
import numpy
import matplotlib as mpl
import swotsimulator.rw_data as rw_data
from netCDF4 import Dataset
import params as p


karin_data = rw_data.file_karin(file=p.karin_file)
fig = plt.figure(figsize=(12,9))
tloc=0.11
tfont=20
stitle = 'Karin noise standard deviation as a function of distance from nadir'
for i in range(0, 8):
     swh = i
     karin_data.read_karin(i)
     hsdt = karin_data.hsdt
     x_ac = karin_data.x_ac
     plt.plot(x_ac, hsdt, label = str(int(swh)) + ' m')
     plt.legend()
plt.ylabel('Karin std')
plt.xlabel('across track (km)')
plt.title(stitle, y=-tloc, fontsize=tfont) #size[1])
plt.savefig('Fig5.png')
