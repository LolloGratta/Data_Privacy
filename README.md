# Data_Privacy
Data privacy applied to Time Series data, using Shou et al. article of 2013 "Supporting Pattern-Preserving Anonymization for Time-Series Data"

# Structure
Files needed in order to run the program:

folder/ <br>
├── Dataset/ <br>
│'''''''''''├──Anonymized <br>
│'''''''''''├── exoStars.csv <br>
│'''''''''''└── Sales_Transaction_Final_Weekly.csv <br>
├── kp-anonymity.py <br>
├── graphs.py <br>
├── Kapra/ <br>
│'''''''''''├── kapra.py <br>
│'''''''''''└── ... <br>
└── Naive/ <br>
''''''''''''├── naive.py <br>
''''''''''''└── ... <br>

# Usage
[*] python kp-anonymity.py algorithm k_value p_value paa_value max_level dataset_path verbosity_level number_rows  <br>
<br>
Where: <br>

**algorithm** can be {kapra.py,naive.py,both} <br>
**k_value** is the k in (k,P). <br>
**p_value** is the P in (k,P). <br>
**max_level** is the max cardinality of the alphabet used to represent the patterns. <br>
**dataset_path** path to dataset.csv (better put it in the Dataset folder. <br>
**verbosity_level** can be {INFO,DEBUG,TRACE}, where INFO is the least verbose and TRACE the most. <br>
**number_rows** defines ho many rows of the dataset will be read, if None all dataset will be used. <br>

Example: python kp-anonymity.py both 15 5 5 4 ./Dataset/exoStars.csv INFO 

