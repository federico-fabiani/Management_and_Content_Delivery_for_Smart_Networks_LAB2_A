import numpy as np
import matplotlib.pyplot as plt

NHOURS = 24
NSCS = 10
time = 3600 * NHOURS

plugged_ev_RES = np.load('plugged_ev' + '_' + 'RESIDENTIAL.npy')
serving_delays_RES = np.load('serving_delays' + '_' + 'RESIDENTIAL.npy')
arrivedVehicles_RES = np.load('arrived_vehicles' + '_' + 'RESIDENTIAL.npy')
arrivedCustomers_RES = np.load('arrived_customers' + '_' + 'RESIDENTIAL.npy')
busyT_RES = np.load('busyT' + '_' + 'RESIDENTIAL.npy')
powerGridUsage_RES = np.load('powerGridUsage' + '_' + 'RESIDENTIAL.npy')

plugged_ev_BUS = np.load('plugged_ev' + '_' + 'BUSINESS.npy')
serving_delays_BUS = np.load('serving_delays' + '_' + 'BUSINESS.npy')
arrivedVehicles_BUS = np.load('arrived_vehicles' + '_' + 'BUSINESS.npy')
arrivedCustomers_BUS = np.load('arrived_customers' + '_' + 'BUSINESS.npy')
busyT_BUS = np.load('busyT' + '_' + 'BUSINESS.npy')
powerGridUsage_BUS = np.load('powerGridUsage' + '_' + 'BUSINESS.npy')

plt.figure()
plt.gca().set(title='ev plugged over time', xlabel='hour', ylabel='N_ev')
plt.plot(plugged_ev_RES[:, 0], plugged_ev_RES[:, 1])
plt.plot(plugged_ev_BUS[:, 0], plugged_ev_BUS[:, 1])
plt.legend(['Residential', 'Business'], loc=10)
plt.xticks(range(0, int(time), 3600), labels=[str(int(x / 3600)) for x in range(0, int(time), 3600)])
plt.show()

plt.figure()
plt.gca().set(title='serving time distribution', xlabel='s', ylabel='frequency')
plt.hist(serving_delays_RES, bins=50, alpha=0.7)
plt.hist(serving_delays_BUS, bins=50, alpha=0.7)
plt.legend(['Residential', 'Business'])
plt.show()

plt.figure()
plt.gca().set(title='ev arrivals distribution', xlabel='time', ylabel='ev')
plt.plot(arrivedVehicles_RES, marker="X")
plt.plot(arrivedVehicles_BUS, marker="h")
plt.grid()
plt.xticks(range(NHOURS))
plt.legend(['Residential', 'Business'])
plt.show()

plt.figure()
plt.gca().set(title='customer arrivals distribution', xlabel='time', ylabel='customer')
plt.plot(arrivedCustomers_RES, marker="X")
plt.plot(arrivedCustomers_BUS, marker="h")
plt.grid()
plt.xticks(range(NHOURS))
plt.legend(['Residential', 'Business'])
plt.show()

plt.figure()
plt.gca().set(title='Work distributed among servers', xlabel='server', ylabel='s')
plt.plot(busyT_RES, marker="X")
plt.plot(busyT_BUS, marker="h")
plt.legend()
plt.grid()
plt.xticks(range(NSCS))
plt.legend(['Residential', 'Business'], loc=9)
plt.show()

plt.figure()
plt.gca().set(title='MWh taken from power grid per time band', xlabel='hour', ylabel='MWh')
plt.plot(powerGridUsage_RES, marker="X")
plt.plot(powerGridUsage_BUS, marker="h")
plt.legend()
plt.grid()
plt.xticks(range(NHOURS))
plt.legend(['Residential', 'Business'])
plt.show()
