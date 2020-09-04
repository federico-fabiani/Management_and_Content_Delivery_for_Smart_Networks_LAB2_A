#!/usr/bin/python3

import random
from queue import Queue, PriorityQueue
import matplotlib.pyplot as plt
from numpy import random as rnd
import numpy as np
import pandas as pd
import logging

# ******************************************************************************
# Constants
# ******************************************************************************

SCENARIO = 'RESIDENTIAL'
# SCENARIO = 'BUSINESS'

RND_ASSIGNMENT = True

SIM_TIME = 3600 * 12
STARTING_TIME = 7
FULL_CHARGE_T = 3600 * 2  # 2H in seconds
READY_THRESHOLD = 80  # %
W_MAX = 300  # 5m in seconds
NSCS = 10
INITIAL_VEHICLES = 7
C = 20  # kWh

ARRIVAL_RATE = 600
MAX_ARRIVAL_RATE = 1000
MIN_ARRIVAL_RATE = 60
k = (MAX_ARRIVAL_RATE - MIN_ARRIVAL_RATE)/SIM_TIME

vehicles = []
queue = []
serving_delays = []
waiting_delays = []
queuing_ev = np.array([0, 0])
plugged_ev = np.array([0, 0])
BusyServer = [False] * NSCS  # True: server is currently busy; False: server is currently idle
ev_arrivals = []
customer_arrivals = []

season = 'SPRING'
cost_electricity = pd.read_csv('electricity_prices.csv', names=['1', 'HOUR', '2', 'SPRING', '3', 'SUMMER', '4', 'FALL', '5', 'WINTER']).get(season)


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
                 power_consuption=0,
                 electricity_cost=0,
                 power_grid_usage=[0] * int(SIM_TIME / 3600)):
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
        self.electricityCost = electricity_cost
        self.powerGridUsage = power_grid_usage


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
        inter_arrival = random.expovariate(1 / (MAX_ARRIVAL_RATE - time * k))
        ev_arrivals.append(time + inter_arrival)
    elif SCENARIO == 'BUSINESS':
        inter_arrival = random.expovariate(1 / (MIN_ARRIVAL_RATE + time * k))
        ev_arrivals.append(time + inter_arrival)
    else:
        inter_arrival = random.expovariate(1 / ARRIVAL_RATE)
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
        available_plugs = []
        for ev in vehicles:
            # check if and eventually how long the vehicle will be ready in the nex few minutes
            if time + W_MAX >= ev.ready_time:
                temp = time + W_MAX - ev.ready_time if time < ev.ready_time else W_MAX
                available_plugs.append(temp)

        if len(available_plugs) > len(queue):
            available_plugs.sort(reverse=True)
            for i in range(len(queue)):
                available_plugs.pop(0)

            opportunity_window = available_plugs[0]

            CUSTOMER_ARRIVAL = MIN_ARRIVAL_RATE + (time + W_MAX) * k if SCENARIO == 'RESIDENTIAL' else MAX_ARRIVAL_RATE - (time + W_MAX) * k
            plugs_expected_to_become_free = min(int(opportunity_window / CUSTOMER_ARRIVAL), len(available_plugs))

            if plugs_expected_to_become_free > len(queue):
                queue.append(EV(time, random.randrange(1, 100), None, None))

            else:
                data.droppedVehicles += 1
        else:
            data.droppedVehicles += 1


def customer_arrival(time, FES):
    global vehicles
    global BusyServer

    # cumulate statistics
    data.arrivedCustomers += 1

    # sample the time until the next event
    if SCENARIO == 'RESIDENTIAL':
        inter_arrival = random.expovariate(1 / (MIN_ARRIVAL_RATE + time * k))
        customer_arrivals.append(time + inter_arrival)
    elif SCENARIO == 'BUSINESS':
        inter_arrival = random.expovariate(1 / (MAX_ARRIVAL_RATE - time * k))
        customer_arrivals.append(time + inter_arrival)
    else:
        inter_arrival = random.expovariate(1 / ARRIVAL_RATE)
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
        data.powerConsumption += (charging_level - leaving_ev.battery_level)/100 * C
        data.electricityCost += eval_consumption_cost(time, leaving_ev, charging_level)
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
# useful function
# ******************************************************************************
def eval_consumption_cost(time, leaving_ev, charging_level):
    if time > 3600 * 11:
        print('')
    start_charge_time = leaving_ev.arrival_time
    cost = 0
    if charging_level != 100:
        end_charge_time = time
    else:
        end_charge_time = leaving_ev.arrival_time + (100 - leaving_ev.battery_level) / 100 * FULL_CHARGE_T
    initial_band = int(start_charge_time // 3600)
    final_band = int(end_charge_time // 3600)
    for band in range(initial_band, final_band+1):
        if band == initial_band and band != final_band:
            time_in_this_band = (band + 1) * 3600 - start_charge_time
        elif band == final_band and band != initial_band:
            time_in_this_band = end_charge_time - band * 3600
        elif band == initial_band and band == final_band:
            time_in_this_band = end_charge_time - start_charge_time
        else:
            time_in_this_band = 3600
        # charging percentage in this band times the total capacity of the battery is the kWh taken from grind.
        charging_in_this_band = time_in_this_band * 100 / FULL_CHARGE_T
        MWh = charging_in_this_band / 100 * C / 1000
        cost += MWh * cost_electricity[band + STARTING_TIME]
        data.powerGridUsage[band] += MWh
    return cost

# ******************************************************************************
# the "main" of the simulation
# ******************************************************************************
random.seed(42)
rnd.seed(42)
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
    plugged_ev = np.vstack([plugged_ev, [time, len(vehicles)]])
    queuing_ev = np.vstack([queuing_ev, [time, len(queue)]])

    (time, event_type) = FES.get()

    if event_type == "customer_arrival":
        customer_arrival(time, FES)

    elif event_type == "ev_arrival":
        ev_arrival(time, FES)

# add to the measurements the data regarding evs plugged at the end of the simulation
if len(vehicles) > 0:
    for ev in vehicles:
        service_time = time - ev.arrival_time
        charging_level = ev.battery_level + 100 * service_time / FULL_CHARGE_T
        if charging_level > 100:
            charging_level = 100
        data.busyT[ev.scs_number] += service_time
        data.powerConsumption += (charging_level - ev.battery_level)/100 * C

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
print("Total power consumption: ", data.powerConsumption, ' kWh')
print('Mean power consumption per vehicle: ', data.powerConsumption / (data.arrivedVehicles - data.droppedVehicles),
      ' kWh')

for i in range(NSCS):
    print('Utilization of station ', i, ': ', data.busyT[i], ' s')

print('Total energy cost: ', data.electricityCost)

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

plt.figure()
plt.gca().set(title='Work distributed among servers', xlabel='server', ylabel='%working time')
plt.plot(data.busyT, marker="X")
plt.legend()
plt.grid()
plt.xticks(range(NSCS))
plt.show()

plt.figure()
plt.gca().set(title='MWh taken from power grid per time band', xlabel='time slot', ylabel='MWh absorbed')
plt.plot(data.powerGridUsage, marker="X")
plt.legend()
plt.grid()
plt.xticks(range(12))
plt.show()
