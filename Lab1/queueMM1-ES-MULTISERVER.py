#!/usr/bin/python3

import random
from queue import Queue, PriorityQueue
from numpy import random as rnd
import numpy as np

# ******************************************************************************
# Constants
# ******************************************************************************
LOAD = 0.85
SERVICE = 10  # av service time
ARRIVAL = SERVICE / LOAD  # av inter-arrival time
RND_ASSIGNMENT = False
EXP_SERVICE_RATE = True

SIM_TIME = 500000
QUEUE_LENGTH = 500
NSERVERS = 1

served_vehicles = 0
BusyServer = [False] * NSERVERS  # True: server is currently busy; False: server is currently idle


MM1 = []
delays = []
users = 0


# ******************************************************************************
# To take the measurements
# ******************************************************************************
class Measure:
    def __init__(self,
                 Narr = 0,
                 Ndep = 0,
                 NDirectlyServed = 0,
                 NAveraegUser = 0,
                 OldTimeEvent = 0,
                 AverageWaitingDelay = 0,
                 AverageQueuingDelay = 0,
                 NDrops = 0,
                 ServersUsage = [0] * NSERVERS):
        self.arr = Narr
        self.dep = Ndep
        self.direct = NDirectlyServed
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.waitingDelay = AverageWaitingDelay
        self.queuingDelay = AverageQueuingDelay
        self.drops = NDrops
        self.busyT = ServersUsage


# ******************************************************************************
# Client
# ******************************************************************************
class Client:
    def __init__(self, arrival_time, server):
        self.arrival_time = arrival_time
        self.server = server


# ******************************************************************************
# Server
# ******************************************************************************
class Server(object):

    # constructor
    def __init__(self):
        # whether the server is idle or not
        self.idle = True


# ******************************************************************************

# arrivals *********************************************************************
def arrival(time, FES, queue):
    global served_vehicles
    global BusyServer
    global users
        # print("Arrival no. ",data.arr+1," at time ",time," with ",users," users" )

    # cumulate statistics
    data.arr += 1
    data.ut += users * (time - data.oldT)
    data.oldT = time

    # sample the time until the next event
    inter_arrival = random.expovariate(lambd=1.0 / ARRIVAL)

    # schedule the next arrival
    FES.put((time + inter_arrival, "arrival"))

    if users < QUEUE_LENGTH + NSERVERS:      # For limited queues a customer is discarded if the line is full
        users += 1

        servedBy = None
        # if the server is idle start the service
        if users <= NSERVERS:
            # Elect the server that will server that will take care of the user (Round robin)
            available_servers = range(NSERVERS) if not RND_ASSIGNMENT else rnd.permutation(NSERVERS)
            for x in available_servers:
                if not BusyServer[x]:
                    BusyServer[x] = True
                    servedBy = x
                    break
            data.direct += 1

            # sample the service time
            service_time = random.expovariate(1.0 / SERVICE) if EXP_SERVICE_RATE else 1.0 / SERVICE
            data.busyT[x] += service_time

            # schedule when the client will finish the server
            FES.put((time + service_time, "departure"))

        # create a record for the client
        client = Client(time, servedBy)

        # insert the record in the queue
        queue.append(client)

    else:
        data.drops += 1


# ******************************************************************************

# departures *******************************************************************
def departure(time, FES, queue):
    global served_vehicles
    global BusyServer
    global users


    # cumulate statistics
    data.dep += 1
    data.ut += users * (time - data.oldT)
    data.oldT = time

    # get the first element from the queue
    client = queue.pop(0)

    # do whatever we need to do when clients go away

    data.waitingDelay += (time - client.arrival_time)
    users -= 1
    BusyServer[client.server] = False

    stillBusyServers = 0
    for x in range(NSERVERS):
        if BusyServer[x]:
            stillBusyServers += 1

    # see whether there are more clients to serve in the line
    if users - stillBusyServers > 0:
        data.queuingDelay += (time - queue[0 + stillBusyServers].arrival_time)
        delays.append(time - queue[0 + stillBusyServers].arrival_time)
        available_servers = range(NSERVERS) if not RND_ASSIGNMENT else rnd.permutation(NSERVERS)
        for x in available_servers:
            if not BusyServer[x]:
                BusyServer[x] = True
                queue[0 + stillBusyServers].server = x
                break

        # sample the service time
        service_time = random.expovariate(1.0 / SERVICE) if EXP_SERVICE_RATE else 1.0 / SERVICE
        data.busyT[x] += service_time

        # schedule when the client will finish the server
        FES.put((time + service_time, "departure"))



# ******************************************************************************
# the "main" of the simulation
# ******************************************************************************

random.seed(42)

data = Measure()

# the simulation time 
time = 0

# the list of events in the form: (time, type)
FES = PriorityQueue(10)

# schedule the first arrival at t=0
FES.put((0, "arrival"))

counter = 0
# simulate until the simulated time reaches a constant
while time < SIM_TIME:
    counter += 1
    (time, event_type) = FES.get()

    if event_type == "arrival":
        arrival(time, FES, MM1)

    elif event_type == "departure":
        departure(time, FES, MM1)

# print output data
print("No. of arrivals =", data.arr, "- No. of departures =", data.dep)

print("Load: ", SERVICE / ARRIVAL)
print("\nArrival rate: ", data.arr / time, " - Departure rate: ", data.dep / time)

print("\nAverage number of users: ", data.ut / time)

print("Average total waiting delay: ", data.waitingDelay / data.dep)
print("Average partial waiting delay: ", data.waitingDelay / (data.dep - data.direct + 0.00001))

print("Average queuing delay: ", data.queuingDelay / data.dep)
print("Partial queuing delay: ", data.queuingDelay / (data.dep - data.direct + 0.00001))

if len(MM1) - NSERVERS > 0:
    print("Actual queue size: ", len(MM1)-NSERVERS)
else:
    print("Queue actually empty")

if len(MM1) > 0:
    print("Arrival time of the last element in the queue:", MM1[- 1].arrival_time)

print("No. of drops: ", data.drops, "- Drops rate: ", data.drops / data.arr)

for server in range(NSERVERS):
    print("Time the server ", server, " was busy: ", data.busyT[server])

# x = np.array(delays)
# np.save(str(LOAD) + '_MM1_delays', x)


