#!/usr/bin/python3

import random
from queue import Queue, PriorityQueue
import matplotlib.pyplot as plt
from numpy import random as rnd
import numpy as np
import logging
import math

# ******************************************************************************
# Constants
# ******************************************************************************

EV_ARRIVAL = 180  # EV inter-arrival time
CUSTOMER_ARRIVAL = 180  # customer inter-arrival time

SCENARIO = 'RESIDENTIAL'
# SCENARIO = 'BUSINESS'

RND_ASSIGNMENT = True
EXP_SERVICE_RATE = True

SIM_TIME = 86400  # 24H in seconds
FULL_CHARGE_T = 7200  # 2H in seconds
READY_THRESHOLD = 80  # %
W_MAX = 300  # 5m in seconds
NSCS = 5
C = 20  # kWh
INITIAL_VEHICLES = 0

k = math.log(240, SIM_TIME)

vehicles = []
queue = []
serving_delays = []
waiting_delays = []
queuing_ev = np.array([0, 0])
plugged_ev = np.array([0, 0])
BusyServer = [False] * NSCS  # True: server is currently busy; False: server is currently idle
ev_arrivals = []
customer_arrivals = []


# ******************************************************************************
# To take the measurements
# ******************************************************************************
class Measure:
    def __init__(self,
                 arrived_vehicles=0,
                 dropped_vehicle=0,
                 waiting_vehicles=0,
                 waiting_time=0,
                 dropped_waiting=0,
                 arrived_customers=0,
                 dropped_customers=0,
                 utilization=0,
                 old_time_event=0,
                 servers_usage=[0] * NSCS,
                 power_consuption=0):
        self.arrivedVehicles = arrived_vehicles
        self.droppedVehicles = dropped_vehicle
        self.waitingVehicles = waiting_vehicles
        self.droppedAfterWaiting = dropped_waiting
        self.waitingTime = waiting_time
        self.arrivedCustomers = arrived_customers
        self.droppedCustomers = dropped_customers
        self.utilization = utilization
        self.oldT = old_time_event
        self.busyT = servers_usage
        self.powerConsumption = power_consuption


# ******************************************************************************
# EV
# ******************************************************************************
class EV:
    def __init__(self, arrival_time, battery_level, scs_number, ready_time):
        self.arrival_time = arrival_time
        self.battery_level = battery_level
        self.scs_number = scs_number
        self.ready_time = ready_time


# ******************************************************************************

# arrivals *********************************************************************
def ev_arrival(time, FES):
    global vehicles
    global BusyServer
    global queue

    # cumulate statistics
    data.arrivedVehicles += 1
    data.utilization += len(vehicles) * (time - data.oldT)
    data.oldT = time

    # sample the time until the next event
    if SCENARIO == 'RESIDENTIAL':
        inter_arrival = random.expovariate(1 / (300 - time ** k))
        ev_arrivals.append(time + inter_arrival)
    if SCENARIO == 'BUSINESS':
        inter_arrival = random.expovariate(1 / (60 + time ** k))
        ev_arrivals.append(time + inter_arrival)

    # schedule the next arrival
    FES.put((time + inter_arrival, "ev_arrival"))

    if len(vehicles) < NSCS:
        # If there is at least one free plug, elect the server that will take care of the user
        servedBy = None
        available_plugs = rnd.permutation(NSCS) if RND_ASSIGNMENT else range(NSCS)
        for x in available_plugs:
            if not BusyServer[x]:
                BusyServer[x] = True
                servedBy = x
                break

        # sample the actual battery charging level
        battery_level = random.randrange(1, 100)

        # create a record for the client
        time_to_charge = (READY_THRESHOLD - battery_level) / 100 * FULL_CHARGE_T if battery_level < READY_THRESHOLD else 0
        ev = EV(time, battery_level, servedBy, time + time_to_charge)

        # insert the record in the queue
        vehicles.append(ev)
        logging.info(str(time) + '|| new ev: ' + str(len(vehicles)) + '/5 dock stations used')

    else:
        # If there are no free plugs, eval if it is probable that one will be available soon
        plugs_expected_to_become_free = 0
        for ev in vehicles:
            # check if and eventually how long the vehicle will be ready in the nex few minutes
            if time + W_MAX >= ev.ready_time:
                opportunity_window = W_MAX - ev.ready_time if time < ev.ready_time else W_MAX
                if opportunity_window / CUSTOMER_ARRIVAL >= 1:
                    plugs_expected_to_become_free += 1

        if plugs_expected_to_become_free > len(queue):
            queue.append(EV(time, random.randrange(1, 100), None, None))

        else:
            data.droppedVehicles += 1


def customer_arrival(time, FES):
    global vehicles
    global BusyServer

    # cumulate statistics
    data.arrivedCustomers += 1

    # sample the time until the next event
    if SCENARIO == 'RESIDENTIAL':
        inter_arrival = random.expovariate(1 / (60 + time ** k))
        customer_arrivals.append(time + inter_arrival)
    if SCENARIO == 'BUSINESS':
        inter_arrival = random.expovariate(1 / (300 - time ** k))
        customer_arrivals.append(time + inter_arrival)

    # schedule the next arrival
    FES.put((time + inter_arrival, "customer_arrival"))

    available = False
    leaving_ev = None

    if vehicles:
        for ev in vehicles:
            if time >= ev.ready_time:
                # client succesfully served
                leaving_ev = ev
                available = True
                break

    if available:
        # checking the power level
        service_time = time - leaving_ev.arrival_time
        charging_level = leaving_ev.battery_level + 100 * service_time / FULL_CHARGE_T
        if charging_level > 100:
            charging_level = 100
        vehicles.pop(vehicles.index(leaving_ev))
        BusyServer[leaving_ev.scs_number] = False
        data.busyT[leaving_ev.scs_number] += service_time
        serving_delays.append(service_time)
        data.powerConsumption += charging_level - leaving_ev.battery_level
        logging.info(str(time) + '|| car hired: ' + str(len(vehicles)) + '/5 dock stations used')

        # check if there is someone in queue to take immediately this plug
        if len(queue) > 0:
            while len(queue) > 0 and queue[0].arrival_time + W_MAX < time:
                logging.info(str(time) + '|| left queue after having waited till:' + str(queue[0].arrival_time + W_MAX))
                queue.pop(0)
                data.droppedVehicles += 1
                data.droppedAfterWaiting += 1

            if len(queue) > 0:
                next_ev = queue.pop(0)

                data.waitingVehicles += 1
                data.waitingTime += time - next_ev.arrival_time
                waiting_delays.append(time - next_ev.arrival_time)

                next_ev.arrival_time = time
                next_ev.scs_number = leaving_ev.scs_number
                time_to_charge = (READY_THRESHOLD - next_ev.battery_level) / 100 * FULL_CHARGE_T \
                    if next_ev.battery_level < READY_THRESHOLD else 0
                next_ev.ready_time = time + time_to_charge

                vehicles.append(next_ev)
                logging.info(str(time) + '|| ev entered from queue: ' + str(len(vehicles)) + '/5 dock stations used')

    else:
        # client not served
        data.droppedCustomers += 1


# ******************************************************************************
# the "main" of the simulation
# ******************************************************************************
random.seed(42)
logging.basicConfig(filename='station.log', filemode='w', level=logging.DEBUG, format='%(message)s')
data = Measure()

# the simulation time
time = 0

# ev available at the beginning
for i in range(INITIAL_VEHICLES):
    vehicles.append(EV(0, 0, i, 0))

# the list of events in the form: (time, type)
FES = PriorityQueue(10)

# schedule the first customer arrival at t=0 and the first ev_arrival at t=0
FES.put((0, "customer_arrival"))
FES.put((0, "ev_arrival"))

# simulate until the simulated time reaches a constant
while time < SIM_TIME:
    # if time > 6340.292:
    #     print('hi')
    plugged_ev = np.vstack([plugged_ev, [time, len(vehicles)]])
    queuing_ev = np.vstack([queuing_ev, [time, len(queue)]])

    (time, event_type) = FES.get()

    if event_type == "customer_arrival":
        customer_arrival(time, FES)

    elif event_type == "ev_arrival":
        ev_arrival(time, FES)

# print output data
print("No. of Vehicles arrived :", data.arrivedVehicles, ' - of which served: ', data.arrivedVehicles - data.droppedVehicles)
print("No. of Customers arrived :", data.arrivedCustomers, ' - of which served: ', data.arrivedCustomers - data.droppedCustomers)
print("Average number of ev docked: ", data.utilization / time, ' - with ', len(vehicles), ' actually present')
print("Average permanence time: ", sum(data.busyT) / (data.arrivedVehicles - data.droppedVehicles), ' s')
print('No. of Vehicles that waited some minutes: ', data.droppedAfterWaiting + data.waitingVehicles,
      ' - of which served: ', data.waitingVehicles)
print('Total waiting time in queue: ', data.waitingTime)
if data.waitingVehicles != 0:
    print('Average waiting time in queue: ', data.waitingTime / data.waitingVehicles)
mean_queue_len = 0
last_time = 0
for row in queuing_ev:
    mean_queue_len += row[1] * (row[0] - last_time)
    last_time = row[0]
mean_queue_len = mean_queue_len / time
print('Average number of Vehicles waiting in queue: ', mean_queue_len, ' - with ', len(queue), 'actually present')
print("Total power consumption: ", data.powerConsumption * C, ' kWh')
print('Mean power consumption per vehicle: ', data.powerConsumption / (data.arrivedVehicles - data.droppedVehicles),
      ' kWh')

for i in range(NSCS):
    print('Utilization of station ', i, ': ', data.busyT[i], ' s')

plt.figure()
plt.title('ev plugged over time')
plt.plot(plugged_ev[:, 0], plugged_ev[:, 1])
plt.show()

plt.figure()
plt.title('serving time distribution')
plt.hist(serving_delays, bins=50)
plt.show()

plt.figure()
plt.title('ev in queue over time')
plt.plot(queuing_ev[:, 0], queuing_ev[:, 1])
plt.show()

plt.figure()
plt.title('waiting time distribution')
plt.hist(waiting_delays, bins=10)
plt.show()

plt.figure()
plt.title('ev arrivals distribution')
plt.hist(ev_arrivals, bins=50)
plt.show()

plt.figure()
plt.title('customer arrivals distribution')
plt.hist(customer_arrivals, bins=50)
plt.show()
