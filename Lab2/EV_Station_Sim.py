#!/usr/bin/python3

import random
from queue import Queue, PriorityQueue
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random as rnd
import numpy as np

# ******************************************************************************
# Constants
# ******************************************************************************
LOAD = 0.95
SERVICE = 10  # av service time
ARRIVAL = SERVICE / LOAD  # av inter-arrival time
RND_ASSIGNMENT = False
EXP_SERVICE_RATE = True

SIM_TIME = 500000
QUEUE_LENGTH = 500
NSERVERS = 1

users = 0
BusyServer = [False] * NSERVERS  # True: server is currently busy; False: server is currently idle


# ******************************************************************************
# EV
# ******************************************************************************
class EV:
    def __init__(self, arrival_time, battery_percentage, assigned_scs):
        self.arrival_time = arrival_time
        self.battery_level = battery_percentage
        self.assigned_scs = assigned_scs


# ******************************************************************************

# arrivals *********************************************************************
def customer_arrival(time, FES, queue):
    global users
    global BusyServer

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
# the "main" of the simulation
# ******************************************************************************

random.seed(42)

# the simulation time
time = 0

# the list of events in the form: (time, type)
FES = PriorityQueue(10)

# schedule the first customer arrival at t=0 and the first ev_arrival at t=0
FES.put((0, "customer_arrival"))
FES.put((0, "ev_arrival"))

# simulate until the simulated time reaches a constant
while time < SIM_TIME:
    (time, event_type) = FES.get()

    if event_type == "customer_arrival":
        customer_arrival(time, FES, MM1)

    elif event_type == "ev_arrival":
        ev_arrival(time, FES, MM1)
