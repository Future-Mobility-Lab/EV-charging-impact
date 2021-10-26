#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import copy
import queue
import pandas as pd

####################################################################
# This is a EV charging station queue simulation program
# It reads file "Northern_Sydney_EV_charger_list.csv"
# And produces file "q2080_2016_seq.csv"
#
# Relies on multiprocessing package to perform parallel simulation.
# To perform calculations for specific OD traffic flow (2016, OD15, OD30)
# change line: DICT['StationFlow'] = float(dt[dt.Name==N]['2016 volume'])
# at the "Setup" section.
####################################################################

########################################################################
# EV class: every EV records time when it entered and exited queue, 
#        entered and exited service
########################################################################
class EV:
    def __init__(self):
        self.enteredQueueTime=-1
        self.exitedQueueTime=-1
        self.enteredServiceTime=-1
        self.exitedServiceTime=-1
        self.serviceTime=-1
        self.queueWaitingTime = -1
    
    def setNecessaryServiceTime(self,time):
        self.serviceTime=time
    
    def setQueueExitTime(self,time):
        self.exitedQueueTime=time
        
        self.queueWaitingTime = self.exitedQueueTime - self.enteredQueueTime
        
    def setQueueEnterTime(self,time):
        self.enteredQueueTime=time
        
    def setServiceExitTime(self,time):
        self.exitedServiceTime=time
        
        self.serviceWaitingTime = self.exitedServiceTime - self.enteredServiceTime
        
    def setServiceEnterTime(self,time):
        self.enteredServiceTime=time


########################################################################
# Service Plug class:
# Service plug can have state "busy" or "not busy".
# It relies on input service time value, which will be decremented on 
# 	every time step since EV entered service.
# Class provides statistics on total work time, % of utilization (worktime/totaltime)
########################################################################

class ServicePlug:
    
    def __init__(self):
        self.busy=False
        self.car=None
        self.resttime=0
        self.totaltime=0
#         self.worktime=0
        self.timeforservice=-1
        
    
    def addEV(self,car,time,timeforservice):
        self.busy=True
        self.car=car
        self.car.setQueueExitTime(time)
        self.car.setServiceEnterTime(time)
        self.timeforservice=timeforservice
        self.ready=False
    
    def tick(self):
        if self.busy:
            if self.timeforservice>0:
                self.timeforservice-=1
        
        if not self.busy:
            self.resttime+=1
#         else:
#             self.worktime+=1
            
        self.totaltime+=1
    
    def isFinished(self):
        return self.timeforservice==0
    
    def removeEV(self,time):
        
        self.busy=False
#         print(self.car)
        self.car.setServiceExitTime(time)
        
        tmpcar = copy.deepcopy(self.car)
        self.timeforservice=-1
        self.car=None
        return tmpcar
    def isbusy(self):
        return self.busy
    
    def utilization(self): # in % of time
        return np.round(1 - (self.resttime/self.totaltime), 2)
    
    
    def worktime(self):
        return np.round(self.totaltime - self.resttime,2)
    
    

########################################################################
# queue class has max size, current size. It is possible to "put" EV 
# into queue and "get" EV from queue
########################################################################

class Queue:
    def __init__(self,maxsize=20):
        self.maxsize=maxsize
        
        import queue
        self.QUEUE = queue.Queue(maxsize=self.maxsize)
        pass
    def size(self):
        return self.QUEUE.qsize()
    
    def put(self, EV, time):
        EV.setQueueEnterTime(time)
        self.QUEUE.put(EV)
    
    
    def putIfPossible(self, EV, time):
        if self.size()<self.maxsize:
            EV.setQueueEnterTime(time)
            self.QUEUE.put(EV)
    
    def get(self, time):
        EV = self.QUEUE.get()
        EV.setQueueExitTime(time)
        return EV   
    


########################################################################
# Model input parameters (second precision modeling):
# 1. Duration of modeling (day, week, month)
# 2. Number of plugs on EV station
# 3. Distribution of time intervals between arrivals
# 4. Distribution of charging time: 80% of charge is a top, let 20% is a min
    # microSDK will be used to estimate distribution of depleted charges 
	# near the station
# 5. Max queue size
# 6. Power usage: KW/h
# [7. Optional: title for simulation]
# Feed callbacks to generate distributions (even represented as one value). 
#     Histograms will be produced automatically.
#
# ######################################################################
#
# Model output parameters:
# 1. Plot of queue size by absolute time
# 2. Total time spent in queue diagram
# 3. Total time spent at charging station
# 4. EV station plugs utilization (usage time / total time of simulation)
# 5. Power consumption at specific EV station plug
########################################################################

class QueueModel:
    def __init__(self):
        self.duration = -1
        self.intervalsBetweenArrivals = None
        self.title = None
        self.serviceTime = None
        self.powerConsumption = None
        self.plugs = None
        self.queue = None
        self.processedEV = []
        self.log=False
        self.EVpercentage = -1
        self.EVneedsCharge = None
        pass
    
    
    def averageQueueWaitingTime(self):
        try:
            return np.array([ev.queueWaitingTime for ev in self.processedEV]).mean()/60
        except:
            return 0.0

    def maxQueueWaitingTime(self):
        try:
            return np.array([ev.queueWaitingTime for ev in self.processedEV]).max()/60
        except:
            return 0.0
    
    def maxTotalTime(self):
        try:
            return np.array([(ev.exitedServiceTime - ev.enteredQueueTime) for ev in self.processedEV]).max()/60
        except:
            return 0.0
    

    
    
    
    def haveFreePlug(self):
        return False in list(map(lambda x: x.isbusy(),self.plugs))

    def haveFinishedPlug(self):
        return True in list(map(lambda x: x.isFinished(),self.plugs))

    def getFreePlugIdx(self):
        for i in range(len(self.plugs)):
            if not self.plugs[i].isbusy():
                return i

    def finishedPlugIdx(self):
        for i in range(len(self.plugs)):
            if self.plugs[i].isFinished():
                return i
    
    def setPlugs(self, n=1):
        self.plugs = [ServicePlug() for i in range(n)]
    
    def setQueueMaxSize(self,maxsize):
        self.queue = Queue(maxsize=maxsize)
    
    def setModelingDuration(self, seconds=None, days=None):
        if seconds and days:
            raise Exception('Define modeling duration using seconds OR days')
        if seconds:
            self.duration = seconds
        if days:
            self.duration = days * 24*60 * 60
    
    def setIntervalBetweenArrivals(self, func):
        self.intervalsBetweenArrivals = list(func(self.duration).astype(np.int))
        
        if self.log:
            plt.figure(figsize=(8,5))
            if self.title:
                plt.title(self.title)
            plt.hist(np.array(self.intervalsBetweenArrivals)/60, bins=100)
            plt.xlabel('Time interval between two consequent arrivals [min]')
    #         plt.ylabel('Number of cases [n]')
            plt.yticks([], [])
            plt.show()
    
    def popInterval(self):
        return self.intervalsBetweenArrivals.pop()
    
    
    
    def setEVneedsCharge(self,perc=0.2):
        self.EVneedsCharge=perc
    
    def getHourInterval(self,vehiclesPerHour, EVpercentage):
        HOUR = 60*60
        vehiclesPerSec = vehiclesPerHour/HOUR
        EVvehiclesPerSec = vehiclesPerSec * EVpercentage/100

#         EVvehiclesPerSec*=self.EVneedsCharge

        return 1/EVvehiclesPerSec
    
    #############################################################################
    # Station inter-arrival interval calculation based on traffic flow from csv #
    #############################################################################
    def setIAcsv(self,csvfile,EVpercentage=0.1,weekend=False):
        self.EVpercentage = EVpercentage
        import pandas as pd
        dt = pd.read_csv(csvfile)
        dt = dt[dt.weekend==int(weekend)].median(axis=0)
        
        self.HOUR_RATES={}
        
        for h in range(24):
            self.HOUR_RATES[ h ] = dt['hour_'+str(h).zfill(2)]
        
        for k in self.HOUR_RATES.keys():
            v = self.HOUR_RATES[k]
            INTERVAL = self.getHourInterval(v, EVpercentage)
            self.HOUR_RATES[k] = list((np.random.normal(0,0.1,size=60*60)*INTERVAL + INTERVAL).astype(int))
    
    
    
    
    #inter-arrival interval calculation based on meadian normalized traffic flow
    def setIAline(self,locset):
        
        NORMS = np.array([0.07437889, 0.04894465, 0.04116507, 0.04268647, 0.08691086,
           0.30521161, 0.69427751, 0.8147551 , 0.77253353, 0.83118941,
           0.85683716, 0.82445112, 0.80252387, 0.78526077, 0.77686075,
           0.86578944, 0.82205588, 0.80057102, 0.6391149 , 0.42236786,
           0.30894179, 0.26085386, 0.2019333 , 0.13184672])
           
        # scaling normalized traffic flow profile by amount of flow through station
        locNORMS = locset['StationFlow'] * (NORMS/NORMS[7-1:8-1 +1].sum()) 
        self.EVpercentage = locset['Percentage of vehicles passing station that will charge their vehicle']
        import pandas as pd
        
        self.HOUR_RATES={}
        
        for h in range(24):
            self.HOUR_RATES[ h ] = locNORMS[h]
        
        
        for k in self.HOUR_RATES.keys():
            v = self.HOUR_RATES[k]
            INTERVAL = self.getHourInterval(v, self.EVpercentage)
            self.HOUR_RATES[k] = list((np.random.normal(0,0.1,size=60*60)*INTERVAL + INTERVAL).astype(int))
            

    # pop inter-arrival interval from calculated set of intervals for every hour
    def popIA(self,time):
        HOUR = int(time/(3600))
        return self.HOUR_RATES[HOUR].pop()
    
    def setIAduration(self):
        self.duration = len(self.HOUR_RATES.keys())*3600
    
    def setServiceTime(self, func):
        self.serviceTime = list(func(self.duration).astype(np.int))
        #print(max(self.serviceTime))
        if self.log:
            plt.figure(figsize=(8,5))
            if self.title:
                plt.title(self.title)
            plt.hist(np.array(self.serviceTime)/60, bins=100)
            plt.xlabel('Service time [min]')
    #         plt.ylabel('Number of cases [n]')
            plt.yticks([], [])
            plt.show()


            plt.figure(figsize=(8,5))
            if self.title:
                plt.title(self.title)
            plt.hist( self.powerConsumption * np.array(self.serviceTime) , bins=100)
            plt.xlabel('Energy consumption [kW]')
    #         plt.ylabel('Number of cases [n]')
            plt.yticks([], [])
            plt.show()

    def setTitle(self,title):
        self.title = title
    def setPowerConsumption(self,kwh):
        self.powerConsumption = kwh # * (1/3600) # getting power: kW*h / h = kW
        

    def consumedPower(self): #in kWh

        return np.round(self.powerConsumption * (np.array([ev.serviceWaitingTime for ev in self.processedEV]).sum()/3600.0),3)

    
        
    def simulate(self):
        self.queue_axis_time = []
        self.queue_axis_size = []
        self.worktime_axis = []
        self.active_plugs_axis = []
        
        time=0
        EVwaiting=False
        nextafter=0
        
        while time < self.duration:
            
            if not EVwaiting:
                if self.HOUR_RATES:
                    nextafter = self.popIA(time)
                else:
                    nextafter = self.popInterval()
                EVwaiting=True
                
            if EVwaiting:
                
                if nextafter<=0:
                    EVwaiting=False
                    
                    if self.log:
                        print('entered queue at ',np.round(time/3600,2),'hours')
                
                    self.queue.putIfPossible(EV(),time)
#                     print(self.queue.size())
            
            waitingForService = self.queue.size()>0
            
            if waitingForService:
                
                if self.haveFreePlug():

                    timeforservice = self.serviceTime.pop()
                    
                    self.plugs[self.getFreePlugIdx()].addEV( self.queue.get(time), time, timeforservice)
            
            [plug.tick() for plug in self.plugs]
            
            import copy
            
            while self.haveFinishedPlug():
        
                self.processedEV.append( copy.deepcopy( self.plugs[self.finishedPlugIdx()].removeEV(time) ))
            
            self.queue_axis_time.append(time)
            self.queue_axis_size.append(self.queue.size())
            self.worktime_axis.append(sum([plug.worktime() for plug in self.plugs]))
            self.active_plugs_axis.append(sum([int(plug.busy==True) for plug in self.plugs]))
            
            nextafter = nextafter - 1
            time+=1
        
        #finishing simulation for vehicles left in queue
        while self.queue.size()>0:
            if self.haveFreePlug():

                timeforservice = self.serviceTime.pop()
                self.plugs[self.getFreePlugIdx()].addEV( self.queue.get(time), time, timeforservice)
            
            [plug.tick() for plug in self.plugs]
            
            while self.haveFinishedPlug():
                self.processedEV.append( copy.deepcopy( self.plugs[self.finishedPlugIdx()].removeEV(time) ))
            
            nextafter = nextafter - 1
            time+=1
        
        
    def plotQueueSizeByTime(self):
        
        plt.rcParams.update({'font.size': 12})
        plt.rc('axes', labelsize=14)
        
        plt.figure(figsize=(17,5))
        plt.title('Daily queue length')
        plt.plot(np.array(self.queue_axis_time)/3600, self.queue_axis_size)
        plt.xlim(0,23)
        plt.ylabel('Queue size [n]')
        plt.xlabel('Absolute time [hour]')
        plt.xticks( np.arange( min(np.array(self.queue_axis_time)/3600), max(np.array(self.queue_axis_time)/3600))  )
        plt.yticks( np.arange(min(self.queue_axis_size), max(self.queue_axis_size) ))
        plt.savefig('Q_queue_size_time.eps',format='eps')
#         plt.show()
    
    def getQueueSizeByTime(self):
        return np.array(self.queue_axis_time), self.queue_axis_size
    
    def getSuppliedPowerByTime(self): #over time
        return (self.powerConsumption/3600) * np.array(self.worktime_axis)
        
    
    def getActivePlugsByTime(self):
        return self.active_plugs_axis
    
    def plugStats(self):
        for i in range(len(self.plugs)):
            print('Plug ',i)
            print('\t','Utilization:',self.plugs[i].utilization()*100,'% ', np.round(self.plugs[i].worktime()/3600,1),'hours ', np.round(170*self.plugs[i].worktime()/3600000,3),'MWh')
            print('\t','Total consumed power (station): ', np.round(self.consumedPower()/1000.0,3),' MWh ')
            print()
            if self.HOUR_RATES:
                print('Modeling duration: ',len(self.HOUR_RATES.keys()),'hrs')

    def plotWaitingTime(self):
        plt.rcParams.update({'font.size': 12})
        plt.rc('axes', labelsize=14)
        
        plt.hist(list(map(lambda v: v.queueWaitingTime/60, self.processedEV)), bins=20)
        print()
        plt.xlabel('Time [minutes]')
        plt.ylabel('Vehicles [n]')
        plt.title('EV queue waiting time distribution')
        plt.savefig('Q_waiting_time.eps',format='eps')



#######################################################################
# Setup
#######################################################################

SETUP=[]
testnumber = 0

dt = pd.read_csv('EV_Northern_Sydney_Lists_7May2020_OD.csv')

for N in dt.Name:
    # ~ for percneedscharge in [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]:
    for percneedscharge in np.concatenate([np.linspace(0.01,0.3,12), [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.75, 1, 1.5, 2, 2.5, 5]]):
        for plugcapacity in [22]:#6.6, 11, 22, 50, 150]:
            DICT={}
            DICT['Test_Number'] = testnumber
            DICT['Weekend'] = 0
            DICT['Station'] = N
            DICT['StationFlow'] = float(dt[dt.Name==N]['2016 volume'])
            DICT['Direction'] = 0
            
            # EV penetration: https://www.statista.com/statistics/789981/australia-electric-vehicle-sales/
            # https://www.abs.gov.au/AUSSTATS/abs@.nsf/Lookup/9309.0Main+Features131%20Jan%202017?OpenDocument
            
            DICT['Percentage of vehicles passing station that will charge their vehicle'] = percneedscharge
            DICT['Number of plugs'] = int(dt[dt.Name==N]['NumPorts']) #plugs
            DICT['Plug capacity'] = plugcapacity
            # ~ DICT['Plug capacity'] = float(dt[dt.Name==N]['portsPower'])
            DICT['Battery size'] = 82 #kWh Tesla Model 3
            SETUP.append(DICT)
            testnumber+=1
print(len(SETUP))


# In[5]:


SETUP

varr=0

def PROCESS(SET):
    print('Test Number: ',SET['Test_Number'])

    model = QueueModel()
    model.setEVneedsCharge(1.0)
    model.log=False
    model.setIAline(SET)

    model.setIAduration()
    model.setTitle('Station '+SET['Station'])

    model.setPowerConsumption(kwh=SET['Plug capacity']) #in kW
    
    SERVICEMAX = 3600 * ((SET['Battery size'])/SET['Plug capacity']) #battery size=40
    
    #generation of normal distribution, where 95% of SoC values within 20..80% interval
    
    # ~ def getST(seconds):
        # ~ NORM = np.random.normal(0,0.147,size=seconds)*1 + 0.5
        # ~ NORM[NORM<=0] = 0.0
        # ~ NORM[NORM>=1] = 1.0
        # ~ return NORM
        
    # ~ def get2040(seconds):
        # ~ NORM = np.random.normal(0,0.05,size=seconds)*1+0.7
        # ~ NORM[NORM<=0] = 0.0
        # ~ NORM[NORM>=1.0] = 1.0
        # ~ return NORM
    
    # ~ def get040(seconds):
        # ~ NORM = (np.random.normal(0,0.147,size=seconds)*1+0.5)
        # ~ NORM[NORM<=0] = 0.0
        # ~ NORM[NORM>=1.0] = 1.0
        # ~ return NORM*0.4
        
    
    def get2080(seconds):
        NORM = (np.random.normal(0,0.146,size=seconds)*1+0.5)
        NORM[NORM<=0] = 0.0
        NORM[NORM>=1.0] = 1.0
        return NORM*0.6+0.2

    model.setServiceTime(lambda seconds: (0.8-get2080(seconds))*SERVICEMAX )
    model.setQueueMaxSize(100)
    model.setPlugs(SET['Number of plugs'])
    
    
    model.simulate()


    ####################################################################
    # Collecting and calculating statistics
    ####################################################################

    Time,Size = model.getQueueSizeByTime()
    Power = model.getSuppliedPowerByTime()

    avgwait = model.averageQueueWaitingTime() #in minutes
    maxwait = model.maxQueueWaitingTime() #in minutes
    active = model.getActivePlugsByTime()
    maxtotal = model.maxTotalTime()
    
    hour=0
    HOURQUEUE = {}
    for P in np.round(np.array(np.split(np.array(Size),24)).mean(axis=1),1):
        HOURQUEUE[hour] = P
        hour+=1
    
    # calculating mean power consumption within hour
    HOURPOWER = dict(zip(np.arange(0,24), np.round((np.array(np.split(np.array(active), 24)).mean(axis=1))*model.powerConsumption,1)))
    # calculating max power consumption within hour
    HOURINSTA = dict(zip(np.arange(0,24), np.round((np.array(np.split(np.array(active), 24)).max(axis=1))*model.powerConsumption,1)))

    print(HOURPOWER)
    HOURWAIT = {}
    for i in range(24):
        HOURWAIT[i] = []

    for i in range(len(model.processedEV)):
        HOURWAIT[model.processedEV[i].enteredQueueTime//3600].append(model.processedEV[i].queueWaitingTime/60)

    HOURSERVICE=[[] for i in range(24)]
    
    for i in range(len(model.processedEV)):
        HOURSERVICE[model.processedEV[i].enteredQueueTime//3600].append(model.processedEV[i].exitedServiceTime - model.processedEV[i].enteredServiceTime)
    
    SERVICEMEAN=[0 for i in range(24)]
    for i in range(24):
        SERVICEMEAN[i] = np.nan_to_num(np.round(np.array(HOURSERVICE[i]).mean(),1),0)/60
    
    MEANST = np.mean([model.processedEV[i].exitedServiceTime - model.processedEV[i].enteredServiceTime for i in range(len(model.processedEV))])/60

    TOTAL=[[] for i in range(24)]
    
    for i in range(len(model.processedEV)):
        TOTAL[model.processedEV[i].enteredQueueTime//3600].append((model.processedEV[i].exitedServiceTime - model.processedEV[i].enteredQueueTime)/60)
    
    for k in range(len(TOTAL)):
        TOTAL[k] = np.nan_to_num(np.round(np.array(TOTAL[k]).mean(),1),0)
        
        
    HOURSUM = {}
    for k in HOURWAIT.keys():
        HOURSUM[k] = np.nan_to_num(np.round(np.array(HOURWAIT[k]).sum(),1),0)
        
    for k in HOURWAIT.keys():
        HOURWAIT[k] = np.nan_to_num(np.round(np.array(HOURWAIT[k]).mean(),1),0)
    
    
    # collecting statistics for a range of 24 hours
    ONERESULT = []
    for i in range(24):
        #to use later
        NEWSET = copy.deepcopy(SET)
        NEWSET['Hour of the day'] = i
        NEWSET['(O1) Mean queue length of an EV station [n]'] = HOURQUEUE[i] #(O1) Mean queue length of an EV stationSiper hour
        NEWSET['(O2) Mean waiting time in queue at an EV station [hours]'] = np.round(HOURWAIT[i]/60,2) #(O2) Mean waiting time in queue at an EV stationS
        NEWSET['(O3) Mean service time to charge at an EV station [hours]'] = np.round(SERVICEMEAN[i]/60,2) #(O3) Mean service time to charge at an EV stationSi
        NEWSET['(O4) Total time spent overall at an EV station [hours]'] = np.round(TOTAL[i]/60,2) #(O4) Total time spent overall at an EV stationS
        NEWSET['(O5) Total energy consumption of an EV station [kWh]'] = sum(HOURPOWER.values()) #np.sum(Power)#(O5) Total energy consumption of an EV stationSi
        NEWSET['(O6) Maximum recorded queue length of an EV station [n]'] = np.max(Size) #(O6) Maximum recorded queue length of an EV stationSi
        NEWSET['(O7) Maximum waiting time in queue at an EV station [hours]'] = np.round(maxwait/60,2) #(O7) Maximum waiting time in queue at an EV stationSi
        NEWSET['(O8) Maximum time spent overall at an EV station [hours]'] = np.round(maxtotal/60,2) #(O8) Maximum time spent overall at an EV station
        NEWSET['(O9) Maximal energy consumption of an EV station [kW]'] = HOURINSTA[i] #O9
        NEWSET['Consumed electricity by hour [kWh]'] = HOURPOWER[i]
        NEWSET['Total waiting time (minutes) by hour'] = np.round(HOURSUM[i]/60,2)
        NEWSET['Overall Mean Service time/day'] = np.round(MEANST/60,2)
        
        
        ONERESULT.append(NEWSET)
    return ONERESULT


from multiprocessing import Pool, Process, Lock, freeze_support
if __name__ == '__main__':
    freeze_support()
    
    results=[]
    
    with Pool(processes=8) as pool:
        results = pool.map(PROCESS, SETUP)
        
    rrr=[]
    for e in results:
        rrr.extend(e)
    dt = pd.DataFrame(rrr)
    
    ####################################################################
    # Output of the queue simulation results
    ####################################################################
    dt.to_csv('q2080_2016_seq.csv',index=False)


