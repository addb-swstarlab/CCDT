### 2023_KCC Model

This is the code of the **Korea Computer Congress 2023** (KCC 2023) paper 

**'A Study about Search Space of Knob Range Reduction for Database Tuning'.**

This study proposes a method to reduce the search space as an optimization method that can improve the performance of database parameters (knobs).

---
#### - MySQL ver. 5.7

#### - Num of Parameters = 139

#### - Num of Config = 200

#### - Workload : TPCC , Twitter
---

Firstly, we randomly generate 200 samples via **Latin Hypercube Sampling** (LHS). 

Secondly, we select **10  knobs** that have a **significant impact** on database performance by a knob ranking algorithm. 

Thirdly, 10  configurations within the generated samples are  selected based on their measured database performance, where we calculated **score *(throughput/latency)*** to compare multiple configurations. 

Then, we find the used **value range of  each selected knob from the selected configurations.**

With these newly defined knob ranges, the optimization  algorithm can search knob values within a narrower range than its default range.

## Paper
Below is link of OANet paper\
[Paper link](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11487958)
