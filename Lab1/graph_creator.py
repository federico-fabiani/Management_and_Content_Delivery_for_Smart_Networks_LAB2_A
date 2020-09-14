import matplotlib.pyplot as plt
import numpy as np

# d1 = np.load('0.35_MM1_delays.npy')
# d2 = np.load('0.60_MM1_delays.npy')
# d3 = np.load('0.85_MM1_delays.npy')
# d4 = np.load('0.95_MM1_delays.npy')

d1 = np.load('0.35_MM25_delays.npy')
d2 = np.load('0.60_MM25_delays.npy')
d3 = np.load('0.85_MM25_delays.npy')
d4 = np.load('0.95_MM25_delays.npy')


plt.figure()
kwargs = dict(alpha=0.5, bins=50, density=True, stacked=True)
plt.hist(d1, **kwargs, color='g', label='0.35')
plt.hist(d2, **kwargs, color='b', label='0.60')
plt.hist(d3, **kwargs, color='r', label='0.85')
plt.hist(d4, **kwargs, color='y', label='0.95')
plt.gca().set(title='Delay frequency', xlabel='delay', ylabel='frequency')
plt.legend(title='Load')
plt.xlim(0, 35)
plt.show()


# b1 = [244376/417454, 114243/417454, 45778/417454, 13057/417454]
# b2 = [105455/417454, 105238/417454, 102880/417454, 103880/417454]
#
# plt.figure()
# plt.gca().set(title='Work distributed among servers', xlabel='server', ylabel='%working time')
# plt.plot(b1, marker="X", label='First preferred')
# plt.plot(b2, marker="h", label='Random choice')
# plt.legend()
# plt.xticks(range(4), labels=['1', '2', '3', '4'])
# plt.show()


