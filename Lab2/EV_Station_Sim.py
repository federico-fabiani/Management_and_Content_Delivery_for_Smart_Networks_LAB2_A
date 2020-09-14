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
# SCENARIO = '' # select this one for standard arrival rate (task 1)

RND_ASSIGNMENT = True

PLOTS = False

NHOURS = 24
SIM_TIME = 3600 * NHOURS
FULL_CHARGE_T = 3600 * 2  # 2H in seconds
READY_THRESHOLD = 100  # %
W_MAX = 60 * 10  # X min in seconds
NSCS = 10
INITIAL_VEHICLES = 8 if SCENARIO == 'RESIDENTIAL' else 2 if SCENARIO == 'BUSINESS' else 0
C = 20  # kWh

STANDARD_EV_RATE = 600
STANDARD_CUSTOMER_RATE = 600
CUSTOMER_ARRIVAL_RATES = [3600,3600,3600,3600,3600,1800,1800,60,300,600,700,900,3600,900,3600,3600,3600,3600,3600,3600,2700,1800,3600,3600]
EV_ARRIVAL_RATES = [3600,900,3600,3600,3600,3600,3600,3600,2700,1800,3600,3600,3600,3600,3600,3600,3600,1800,1800,60,300,600,700,900]

serving_delays = []
waiting_delays = []
queuing_ev = np.array([0, 0])
plugged_ev = np.array([0, 0])
BusyServer = [False] * NSCS  # True: server is currently busy; False: server is currently idle


season = 'FALL'
cost_electricity = pd.read_csv('electricity_prices.csv',
                               names=['1', 'HOUR', '2', 'SPRING', '3', 'SUMMER', '4', 'FALL', '5', 'WINTER']).get(
    season)

# TASK 3
adaptable_TMAX_f = True
f = int(50 / 100 * NSCS)
T_MAX = 2700
NPOSTPONED = 0
postponed_per_hour = [0] * NHOURS


# ******************************************************************************
# To take the measurements
# ******************************************************************************
class Measure:
    def __init__(self,
                 arrived_vehicles=[0]*NHOURS,
                 dropped_vehicle=0,
                 waiting_vehicles=0,
                 waiting_time=0,
                 dropped_waiting=0,
                 arrived_customers=[0]*NHOURS,
                 dropped_customers=0,
                 utilization=0,
                 old_time_event=0,
                 servers_usage=[0] * NSCS,
                 power_consuption=0,
                 electricity_cost=0,
                 power_grid_usage=[0] * NHOURS,
                 total_postponed=0):
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
        self.totalPostponed = total_postponed


# ******************************************************************************
# EV
# ******************************************************************************
class EV:
    def __init__(self, arrival_time, battery_level, scs_number, start_charging, ready_time):
        self.arrival_time = arrival_time
        self.battery_level = battery_level
        self.scs_number = scs_number
        self.start_charging = start_charging
        self.ready_time = ready_time


# ******************************************************************************

# arrivals *********************************************************************
def ev_arrival(time, FES, hour):
    global served_vehicles
    global BusyServer
    global queue
    global NPOSTPONED

    # cumulate statistics
    data.arrivedVehicles[hour] += 1
    data.utilization += len(served_vehicles) * (time - data.oldT)
    data.oldT = time

    # sample the time until the next event
    rate = eval_inter_arrival_rate('EV', time, hour)
    inter_arrival = random.expovariate(1/rate)#random.normalvariate(rate, rate/3)

    # schedule the next arrival
    FES.put((time + inter_arrival, "ev_arrival"))

    if len(served_vehicles) < NSCS:
        # If there is at least one free plug, elect the server that will take care of the user
        servedBy = None
        available_plugs = rnd.permutation(NSCS) if RND_ASSIGNMENT else range(NSCS)
        for x in available_plugs:
            if not BusyServer[x]:
                BusyServer[x] = True
                servedBy = x
                break

        # TASK 3: decide whether to start now the charging or to postpone it
        sec_to_next_hour = 3600 - time % 3600
        if adaptable_TMAX_f:
            expected_customers = sec_to_next_hour / eval_inter_arrival_rate('CUSTOMER', time, hour)
            maxDelay = sec_to_next_hour
            nDelay = NSCS - expected_customers
        else:
            maxDelay = T_MAX
            nDelay = f
        if sec_to_next_hour <= maxDelay and NPOSTPONED < nDelay and cost_electricity[hour] > cost_electricity[(hour+1) % NHOURS]:
            # postpone charging till the next time band
            start_charging = (hour+1) % NHOURS * 3600
            NPOSTPONED += 1
            postponed_per_hour[hour] += 1
        else:
            # don't postpone
            start_charging = time

        # sample the actual battery charging level
        battery_level = random.randrange(1, 100)

        # create a record for the client
        time_to_charge = (READY_THRESHOLD-battery_level)/100 * FULL_CHARGE_T if battery_level < READY_THRESHOLD else 0
        new_ev = EV(time, battery_level, servedBy, start_charging, start_charging + time_to_charge)

        # insert the record in the queue
        served_vehicles.append(new_ev)
        logging.info(str(time) + '|| new ev: ' + str(len(served_vehicles)) + '/5 dock stations used')

    else:
        # If there are no free plugs, eval if it is probable that one will be available soon
        available_plugs = []
        for ev in served_vehicles:
            # check if and eventually how long the vehicle will be ready in the nex few minutes
            if time + W_MAX >= ev.ready_time:
                temp = time + W_MAX - ev.ready_time if time < ev.ready_time else W_MAX
                available_plugs.append(temp)

        if len(available_plugs) > len(queue):
            available_plugs.sort(reverse=True)
            for i in range(len(queue)):
                available_plugs.pop(0)

            opportunity_window = available_plugs[0]

            plugs_expected_to_become_free = min(int(opportunity_window/eval_inter_arrival_rate('CUSTOMER', time, hour)),
                                                len(available_plugs))

            if plugs_expected_to_become_free != 0:
                queue.append(EV(time, random.randrange(1, 100), None, None, None))

            else:
                data.droppedVehicles += 1
        else:
            data.droppedVehicles += 1


def customer_arrival(time, FES, hour):
    global served_vehicles
    global BusyServer
    global NPOSTPONED

    # cumulate statistics
    data.arrivedCustomers[hour] += 1

    # sample the time until the next event
    rate = eval_inter_arrival_rate('CUSTOMER', time, hour)
    inter_arrival = random.expovariate(1/rate)#random.normalvariate(rate, rate/3)

    # schedule the next arrival
    FES.put((time + inter_arrival, "customer_arrival"))

    available = False
    leaving_ev = None

    if served_vehicles:
        for ev in served_vehicles:
            if time >= ev.ready_time:
                # client succesfully served
                leaving_ev = ev
                available = True
                break

    if available:
        # checking the power level
        service_time = time - leaving_ev.start_charging
        charging_level = leaving_ev.battery_level + 100 * service_time / FULL_CHARGE_T
        if charging_level > 100:
            charging_level = 100
        served_vehicles.pop(served_vehicles.index(leaving_ev))
        BusyServer[leaving_ev.scs_number] = False
        data.busyT[leaving_ev.scs_number] += service_time
        serving_delays.append(service_time)
        data.powerConsumption += (charging_level - leaving_ev.battery_level) / 100 * C
        data.electricityCost += eval_consumption_cost(time, leaving_ev, charging_level)
        logging.info(str(time) + '|| car hired: ' + str(len(served_vehicles)) + '/5 dock stations used')

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
                BusyServer[next_ev.scs_number] = True

                # TASK 3: decide whether to start now the charging or to postpone it
                sec_to_next_hour = 3600 - time % 3600
                if adaptable_TMAX_f:
                    expected_customers = sec_to_next_hour / eval_inter_arrival_rate('CUSTOMER', time, hour)
                    maxDelay = sec_to_next_hour
                    nDelay = NSCS - expected_customers
                else:
                    maxDelay = T_MAX
                    nDelay = f
                if sec_to_next_hour <= maxDelay and NPOSTPONED < nDelay and cost_electricity[hour] > cost_electricity[
                    (hour + 1) % NHOURS]:
                    # postpone charging till the next time band
                    start_charging = (hour + 1) % NHOURS * 3600
                    NPOSTPONED += 1
                    postponed_per_hour[hour] += 1
                else:
                    # don't postpone
                    start_charging = time

                next_ev.start_charging = start_charging

                time_to_charge = (READY_THRESHOLD-next_ev.battery_level)/100 * FULL_CHARGE_T \
                    if next_ev.battery_level < READY_THRESHOLD else 0
                next_ev.ready_time = start_charging + time_to_charge

                served_vehicles.append(next_ev)
                logging.info(str(time) + '|| ev entered from queue: ' + str(len(served_vehicles)) + '/5 dock stations used')

    else:
        # client not served
        data.droppedCustomers += 1


# ******************************************************************************
# useful function
# ******************************************************************************
def eval_inter_arrival_rate(event, time, hour):
    seconds = time % 3600
    rate = None
    if event == 'EV':
        if SCENARIO == 'RESIDENTIAL':
            rate = EV_ARRIVAL_RATES[hour] + EV_ARRIVAL_RATES[(hour + 1) % 24] / 3600 * seconds
        elif SCENARIO == 'BUSINESS':
            rate = CUSTOMER_ARRIVAL_RATES[hour] + CUSTOMER_ARRIVAL_RATES[(hour + 1) % 24] / 3600 * seconds
        else:
            rate = STANDARD_EV_RATE

    if event == 'CUSTOMER':
        if SCENARIO == 'RESIDENTIAL':
            rate = CUSTOMER_ARRIVAL_RATES[hour] + CUSTOMER_ARRIVAL_RATES[(hour + 1) % 24] / 3600 * seconds
        elif SCENARIO == 'BUSINESS':
            rate = EV_ARRIVAL_RATES[hour] + EV_ARRIVAL_RATES[(hour + 1) % 24] / 3600 * seconds
        else:
            rate = STANDARD_CUSTOMER_RATE

    return rate


def eval_consumption_cost(time, leaving_ev, charging_level):
    cost = 0
    if charging_level != 100:
        end_charge_time = time
    else:
        end_charge_time = leaving_ev.start_charging + (100 - leaving_ev.battery_level) / 100 * FULL_CHARGE_T
    initial_band = int(leaving_ev.start_charging // 3600)
    final_band = int(end_charge_time // 3600)
    for band in range(initial_band, final_band + 1):
        if band == initial_band and band != final_band:
            time_in_this_band = (band + 1) * 3600 - leaving_ev.start_charging
        elif band == final_band and band != initial_band:
            time_in_this_band = end_charge_time - band * 3600
        elif band == initial_band and band == final_band:
            time_in_this_band = end_charge_time - leaving_ev.start_charging
        else:
            time_in_this_band = 3600
        # charging percentage in this band times the total capacity of the battery is the kWh taken from grind.
        charging_in_this_band = time_in_this_band * 100 / FULL_CHARGE_T
        MWh = charging_in_this_band / 100 * C / 1000
        cost += MWh * cost_electricity[band]
        data.powerGridUsage[band] += MWh
    return cost


# ******************************************************************************
# the "main" of the simulation
# ******************************************************************************
random.seed(16)
rnd.seed(42)
logging.basicConfig(filename='station.log', filemode='w', level=logging.DEBUG, format='%(message)s')
data = Measure()
served_vehicles = []
queue = []

# the simulation time
time = 0

# ev available at the beginning
for i in range(INITIAL_VEHICLES):
    served_vehicles.append(EV(0, 100, i, 0, 0))
    BusyServer[i] = True

# the list of events in the form: (time, type)
FES = PriorityQueue(10)

# schedule the first customer arrival at t=0 and the first ev_arrival at t=0
FES.put((0, "customer_arrival"))
FES.put((0, "ev_arrival"))

# simulate until the simulated time reaches a constant
current_hour = 0

while time < SIM_TIME:
    plugged_ev = np.vstack([plugged_ev, [time, len(served_vehicles)]])
    queuing_ev = np.vstack([queuing_ev, [time, len(queue)]])

    (time, event_type) = FES.get()

    # Check if we are in a new time slot and in case reset the number of ev whose charging can be postponed
    if time // 3600 % NHOURS > current_hour:
        data.totalPostponed += NPOSTPONED
        NPOSTPONED = 0
        current_hour = int(time // 3600 % NHOURS)

    if event_type == "customer_arrival":
        customer_arrival(time, FES, current_hour)

    elif event_type == "ev_arrival":
        ev_arrival(time, FES, current_hour)

# add to the measurements the data regarding evs plugged at the end of the simulation
if len(served_vehicles) > 0:
    for ev in served_vehicles:
        service_time = time - ev.start_charging
        charging_level = ev.battery_level + 100 * service_time / FULL_CHARGE_T
        if charging_level > 100:
            charging_level = 100
        data.busyT[ev.scs_number] += service_time
        data.powerConsumption += (charging_level - ev.battery_level) / 100 * C

# print output data
print("No. of Vehicles arrived :", sum(data.arrivedVehicles), ' - of which served: ',
      sum(data.arrivedVehicles) - data.droppedVehicles)
print("No. of Customers arrived :", sum(data.arrivedCustomers), ' - of which served: ',
      sum(data.arrivedCustomers) - data.droppedCustomers)
print("Average number of ev docked: ", data.utilization / time, ' - with ', len(served_vehicles), ' actually present')
print("Average permanence time: ", sum(data.busyT) / (sum(data.arrivedVehicles) - data.droppedVehicles), ' s')
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
print('Mean power consumption per vehicle: ', data.powerConsumption / (sum(data.arrivedVehicles) - data.droppedVehicles),
      ' kWh')

for i in range(NSCS):
    print('Utilization of station ', i, ': ', data.busyT[i], ' s')

print('ev loss rate: ', data.droppedVehicles/sum(data.arrivedVehicles), ' - cust loss rate: ', data.droppedCustomers/sum(data.arrivedCustomers))

print('Total energy cost: ', data.electricityCost, ' - Average cost per vehicle served: ', data.electricityCost / (sum(data.arrivedVehicles) - data.droppedVehicles))

print('Number of charge postponed: ', data.totalPostponed, ' - equal to ', data.totalPostponed/sum(data.arrivedVehicles), '% of the arrived ev')

if PLOTS:
    plt.figure()
    plt.gca().set(title='ev plugged over time', xlabel='hour', ylabel='N_ev')
    plt.plot(plugged_ev[:, 0], plugged_ev[:, 1])
    plt.xticks(range(0, int(time), 3600), labels=[str(int(x/3600)) for x in range(0, int(time), 3600)])
    plt.show()

    plt.figure()
    plt.gca().set(title='serving time distribution', xlabel='s', ylabel='frequency')
    plt.hist(serving_delays, bins=50)
    plt.show()

    plt.figure()
    plt.gca().set(title='ev in queue over time', xlabel='hour', ylabel='N_ev')
    plt.plot(queuing_ev[:, 0], queuing_ev[:, 1])
    plt.xticks(range(0, int(time), 3600), labels=[str(int(x/3600)) for x in range(0, int(time), 3600)])
    plt.show()

    plt.figure()
    plt.gca().set(title='waiting time distribution', xlabel='s', ylabel='frequency')
    plt.hist(waiting_delays, bins=10)
    plt.show()

    plt.figure()
    plt.gca().set(title='ev arrivals distribution', xlabel='time', ylabel='ev')
    plt.plot(data.arrivedVehicles, marker="X")
    plt.grid()
    plt.xticks(range(NHOURS))
    plt.show()

    plt.figure()
    plt.gca().set(title='customer arrivals distribution', xlabel='time', ylabel='customer')
    plt.plot(data.arrivedCustomers, marker="X")
    plt.grid()
    plt.xticks(range(NHOURS))
    plt.show()

    plt.figure()
    plt.gca().set(title='Work distributed among servers', xlabel='server', ylabel='%working time')
    plt.plot(data.busyT, marker="X")
    plt.legend()
    plt.grid()
    plt.xticks(range(NSCS))
    plt.show()

    plt.figure()
    plt.gca().set(title='MWh taken from power grid per time band', xlabel='hour', ylabel='MWh')
    plt.plot(data.powerGridUsage, marker="X")
    plt.legend()
    plt.grid()
    plt.xticks(range(NHOURS))
    plt.show()

    plt.figure()
    plt.gca().set(title='Number of ev whose charging was postponed per time band', xlabel='time slot', ylabel='N')
    plt.plot(postponed_per_hour, marker="X")
    plt.legend()
    plt.grid()
    plt.xticks(range(NHOURS))
    plt.show()

# np.save('plugged_ev' + '_' + SCENARIO, plugged_ev)
# np.save('serving_delays' + '_' + SCENARIO, serving_delays)
# np.save('arrived_vehicles' + '_' + SCENARIO, data.arrivedVehicles)
# np.save('arrived_customers' + '_' + SCENARIO, data.arrivedCustomers)
# np.save('busyT' + '_' + SCENARIO, data.busyT)
# np.save('powerGridUsage' + '_' + SCENARIO, data.powerGridUsage)

