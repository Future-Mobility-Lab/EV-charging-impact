# EV-charging-impact

This repository contains the code that has been used for the Queue modelling for the paper "How will electric vehicles affect traffic congestion and energy
consumption: an integrated modelling approach" by Artur Grigorev, Tuo Mao, Adam Berry, Joachim Tan, Loki Purushothaman, Adriana-Simona Mihaita. The paper has been published and presented during the IEEE ITSC 2021 conference. The preprint is available: https://arxiv.org/abs/2110.14064 .

You can find a working queue model in "queue_model.py" file.

This EV charging station queue simulation program reads file "Northern_Sydney_EV_charger_list.csv" and outputs queue simulation results into file "q2080_2016_seq.csv". It relies on multiprocessing package to perform parallel simulation. 

# Input parameters of the model:
1. Duration of modeling (day, week, month)
2. Number of plugs on EV stations
3. Distribution of time intervals between arrivals
4. Distribution of charging time: normaly distributed between 20% and 80%.
5. Max queue size
6. Power supply at EV charger: KW/h

# Model output:
* (O1) Mean queue length of an EV station [n]'] = HOURQUEUE[i]
* (O2) Mean waiting time in queue at an EV station [hours]
* (O3) Mean service time to charge at an EV station [hours]
* (O4) Total time spent overall at an EV station [hours]
* (O5) Total energy consumption of an EV station [kWh]
* (O6) Maximum recorded queue length of an EV station [n]
* (O7) Maximum waiting time in queue at an EV station [hours]
* (O8) Maximum time spent overall at an EV station [hours]
* (O9) Maximal energy consumption of an EV station [kW]
* Consumed electricity by hour [kWh]
* Total waiting time (minutes) by hour
* Overall Mean Service time/day'

![queue model](https://github.com/Future-Mobility-Lab/EV-charging-impact/blob/main/queue-model.PNG "Title")



To perform calculations for specific OD traffic flow (2016, OD15, OD30) change the line: DICT['StationFlow'] = float(dt[dt.Name==N]['2016 volume']) at the "Setup" section (to 2016, 15 or 30).

The structure of the framework:
![framework](https://github.com/Future-Mobility-Lab/EV-charging-impact/blob/main/framework.PNG "Title")

The code to produce lineplots is in "lineplots.ipynb":

![lineplot](https://github.com/Future-Mobility-Lab/EV-charging-impact/blob/main/lineplot.PNG "Title")

![lineplot2](https://github.com/Future-Mobility-Lab/EV-charging-impact/blob/main/lineplot2.PNG "Title")

The code to produce supplementary animation is in "anim.ipynb":
![anim](https://github.com/Future-Mobility-Lab/EV-charging-impact/blob/main/EV-029.png "Title")
