# EV-charging-impact

This repository contains the code for the paper "How will electric vehicles affect traffic congestion and energy
consumption: an integrated modelling approach" by Artur Grigorev, Tuo Mao, Adam Berry, Joachim Tan, Loki Purushothaman, Adriana-Simona Mihaita.

You can find a working queue model in "queue_model.py" file.

![queue model](https://github.com/Future-Mobility-Lab/EV-charging-impact/blob/main/queue-model.PNG "Title")

This EV charging station queue simulation program reads file "Northern_Sydney_EV_charger_list.csv" and outputs queue simulation results into file "q2080_2016_seq.csv". It relies on multiprocessing package to perform parallel simulation. 

To perform calculations for specific OD traffic flow (2016, OD15, OD30) change the line: DICT['StationFlow'] = float(dt[dt.Name==N]['2016 volume']) at the "Setup" section (to 15 or 30).
