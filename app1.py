import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import time
import pyomo.environ as pyomo
from pyomo.environ import *

# import matplotlib.pyplot as plt
import csv
import shutil
import sys
import os.path


st.title("Hui Code Gui v2")

st.sidebar.title("Please configure parameters:")

file = st.sidebar.file_uploader("Select a job rules file")

file2 = st.sidebar.file_uploader("Please select a Resource Count file")

date = st.sidebar.date_input("Select a Date")

timee = st.sidebar.time_input(
    "Select time to start:",
)

resource = st.sidebar.selectbox(
    "Which Resource to process?",
    options=[
        "Agente_II_OPT",
        "SAP_Agente_II",
        "Auditor_I",
        "Auditor_II",
        "Connecting_Driver",
        "ECO",
        "PTY_Driver",
    ],
)

user_input = st.sidebar.text_input("Productivity Factor to use", "0.657534246575342")

if file:
    filename = file.name
else:
    filename = ""

if file2:
    filename2 = file2.name
else:
    filename2 = ""
st.text(f"The file selected for rules:{filename}")

st.text(f"The Resource Count is:{filename2}")

st.text(f"We will start processing the file at :{date} {timee}")

toggle = st.button("Run!")


def probar():
    my_bar = st.progress(1)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    st.download_button("Download Results", "")


if toggle:
    probar()


def time_seq(stat, endt):
    statt = file_df.index[file_df["rangos"] == stat].tolist()[0]
    endd = file_df.index[file_df["rangos"] == endt].tolist()[0]
    seque = file_df.loc[statt:endd]["rangos"].dt.strftime("%m/%d/%Y %-H:%M").tolist()
    return seque


# load data

# Input 1: data file to be read
data_file = filename  #'201203 MGA_1st_half2021_Bgt_Jun_Resource_Count.csv' #'Hui LIM TentativoCM 210725- RC.csv' #'Hui LIM Todo 210815- RC.csv' #'Hui LIM TentativoCM 210815- RC.csv' #'Hui LIM Tentativo3rd 210815- RC.csv' # 'Hui LIM Tentativo3rd 210725- RC.csv'
file_df = pd.read_csv(data_file)

# Input 2: How many days to scheduel after the specified starting slot?
start_time = "07/25/2021 0:00"  #'07/25/2021 0:00'  # '08/15/2021 0:00'  # Start scheduling from which date at what time
nday = 1  # Schedule for how many days since the start_time

# Input 3: Airport productivity percentage:
productivity = 0.657534246575342  # 0.673972602739726 #MIA #0.673972602739726 #JFK #0.632876712328767 #HAV # 0.589041095890411 #CCS # 0.657534246575342 #LIM # 0.754794520547945 #MGA  #0.652054794520548 #SCL  # FTE = sum(int(round(sum(workforce_model.n[k,t] for t in day_time)/productivity, 0))*work_salary[k] for k in work_type)

file_df["rangos"] = pd.to_datetime(file_df["rangos"])
file_dfc = pd.DataFrame()

n_ts = 96  # time slots per day (fixed usually)

start_slot = file_df.index[file_df["rangos"] == start_time].tolist()[0]

file_dfc = file_df.loc[start_slot : (start_slot + nday * n_ts - 1), :].copy()

day_time = file_dfc["rangos"].dt.strftime("%m/%d/%Y %-H:%M").tolist()
file_dfc["rangos"] = file_dfc["rangos"].dt.strftime("%m/%d/%Y %-H:%M")

file_dfc[["Date", "Time"]] = file_dfc.rangos.str.split(
    " ",
    expand=True,
)
time_slots = file_dfc["Time"].loc[start_slot : (start_slot + n_ts - 1)].tolist()

# Extract days in a list
days = file_dfc["Date"].unique().tolist()

# Input 4: working type information
worktype_info = {
    "Full Time Day": {
        "early_release": 0,
        "meal_in_shift": 1,
        "Lunch_Begin_Shift_Start_Offset_Plus": 1,
        "Lunch_Begin_Shift_Start_Offset_Minus": 1,
        "meal_mins": 45,
        "work_range": [
            "6:00",
            "6:15",
            "6:30",
            "6:45",
            "7:00",
            "7:15",
            "7:30",
            "7:45",
            "8:00",
            "8:15",
            "8:30",
            "8:45",
            "9:00",
            "9:15",
            "9:30",
            "9:45",
            "10:00",
        ],
        "shift_hours": 8,
        "salary": 8 / 8,
    },
    "Full Time Mix AM": {
        "early_release": 0,
        "meal_in_shift": 1,
        "Lunch_Begin_Shift_Start_Offset_Plus": 1,
        "Lunch_Begin_Shift_Start_Offset_Minus": 1,
        "meal_mins": 30,
        "work_range": [
            "3:00",
            "3:15",
            "3:30",
            "3:45",
            "4:00",
            "4:15",
            "4:30",
            "4:45",
            "5:00",
            "5:15",
            "5:30",
            "5:45",
        ],
        "shift_hours": 7.5,
        "salary": 7.5 / 8,
    },
    "Full Time Mix PM": {
        "early_release": 0,
        "meal_in_shift": 1,
        "Lunch_Begin_Shift_Start_Offset_Plus": 1,
        "Lunch_Begin_Shift_Start_Offset_Minus": 1,
        "meal_mins": 30,
        "work_range": [
            "10:15",
            "10:30",
            "10:45",
            "11:00",
            "11:15",
            "11:30",
            "11:45",
            "12:00",
            "12:15",
            "12:30",
            "12:45",
            "13:00",
            "13:15",
            "13:30",
        ],
        "shift_hours": 7.5,
        "salary": 7.5 / 8,
    },
    "Full Time Night AM": {
        "early_release": 0,
        "meal_in_shift": 1,
        "Lunch_Begin_Shift_Start_Offset_Plus": 1,
        "Lunch_Begin_Shift_Start_Offset_Minus": 1,
        "meal_mins": 30,
        "work_range": [
            "0:00",
            "0:15",
            "0:30",
            "0:45",
            "1:00",
            "1:15",
            "1:30",
            "1:45",
            "2:00",
            "2:15",
            "2:30",
            "2:45",
        ],
        "shift_hours": 7,
        "salary": 7 / 8,
    },
    "Full Time Night PM": {
        "early_release": 0,
        "meal_in_shift": 1,
        "Lunch_Begin_Shift_Start_Offset_Plus": 1,
        "Lunch_Begin_Shift_Start_Offset_Minus": 1,
        "meal_mins": 30,
        "work_range": [
            "13:45",
            "14:00",
            "14:15",
            "14:30",
            "14:45",
            "15:00",
            "15:15",
            "15:30",
            "15:45",
            "16:00",
            "16:15",
            "16:30",
            "16:45",
            "17:00",
            "17:15",
            "17:30",
            "17:45",
            "18:00",
            "18:15",
            "18:30",
            "18:45",
            "19:00",
            "19:15",
            "19:30",
            "19:45",
            "20:00",
            "20:15",
            "20:30",
            "20:45",
            "21:00",
            "21:15",
            "21:30",
            "21:45",
            "22:00",
            "22:15",
            "22:30",
            "22:45",
            "23:00",
            "23:15",
            "23:30",
            "23:45",
        ],
        "shift_hours": 7,
        "salary": 7 / 8,
    },
    "Part Time 6": {
        "early_release": 0,
        "meal_in_shift": 1,
        "Lunch_Begin_Shift_Start_Offset_Plus": 1,
        "Lunch_Begin_Shift_Start_Offset_Minus": 1,
        "meal_mins": 30,
        "work_range": time_slots,
        "shift_hours": 6,
        "salary": 6 / 8,
    },
    "Part Time 5": {
        "early_release": 0,
        "meal_in_shift": 0,
        "Lunch_Begin_Shift_Start_Offset_Plus": 0,
        "Lunch_Begin_Shift_Start_Offset_Minus": 0,
        "meal_mins": 0,
        "work_range": time_slots,
        "shift_hours": 5,
        "salary": 5 / 8,
    },
    "Part Time 4": {
        "early_release": 0,
        "meal_in_shift": 0,
        "Lunch_Begin_Shift_Start_Offset_Plus": 0,
        "Lunch_Begin_Shift_Start_Offset_Minus": 0,
        "meal_mins": 0,
        "work_range": time_slots,
        "shift_hours": 4,
        "salary": 4 / 8,
    },
}
resource_type = "Total_Agente_SAP"  # What type of resources will be scheduled, agent, supervisor etc.

# Input 5: Tuning parameters
FTE_opt = 2  # weight for boosting the role FTE played in optimization
NH_opt = 1  # weight for boosting the role NH played in optimization
width_hrs = 1  # Between-activation interval (hrs)

cost_overstaff = 0  # Usually zero. Cost of overstaffing. It is equvalent to minimize staff salaries. Its non-zero may help generate alternative solutions
switch_blocktime_briefing = 0  # Usually zero. Decide whether or not to activate a heuristics of briefing time blockage. For example, there is a demand increase at 5:00. We will block any workers from being hired at 5:00.
active_cost = 0  # Cost per activation of workers. Can help reduce activation. It can help supplement the role for Between-activation interval.

circular = 1  # If we need to make circular scheduling periodically 1- Yes, 0- No.  Circular means ...=>Day N 23:45==> Day 1 0:00==>0:15==>...Day N 23:45==>Day 1 0:00==>Day 1 0:15==> ... Noncircular may mean 0:00 is the start of the scheduling and 23:45 is the end. 23:45 does not preceed 0:00.
# worker_actfreq_UP ={'D1': []}  #Activation frequency upper bound for each day. It is disabled for now since between-activation interval and activation cost can replace it.

# Lower bound for a certain type of workers per day. In some airports, there are a minimum number of a certain type workers available who must be employed.
work_LB = {"Full Time": [], "Part Time 6": [], "Part Time 5": [], "Part Time 4": []}
# Full Time Workers include ['Full Time Day', 'Full Time Mix AM', 'Full Time Mix PM', 'Full Time Night AM', 'Full Time Night PM']

# Specify the upper bound for each type of workers in total (e.g., no part time workers allowed. Full time only in HAV)
work_UB = {
    "Full Time": [],
    "Part Time 6": [],
    "Part Time 5": [],
    "Part Time 4": [],  # Replace [] by entering the upper bound for each worker type per day
}

# When meal time requirement cannot be satisfied, we can run one of the following heuristic to adjust worker number or overstaffing:
# Strategy 1: Specify the lower bound adjustment of a certain type of workers to cover adjustment requirement (e.g. meals)
work_adj_LB = {
    "Full Time": [],
    "Part Time 6": [],
    "Part Time 5": [],
    "Part Time 4": [],  # Fill in the value inside bracket e.g., [5] or [3, 4,3,6]. It should match the size in work_t_adj
}

work_t_adj = {
    "Full Time": [],
    "Part Time 6": [],
    "Part Time 5": [],
    "Part Time 4": [],  # [[time_seq('07/25/2021 1:30', '07/25/2021 23:45')+ time_seq('07/25/2021 0:00', '07/25/2021 1:15')]]
}
for i in work_t_adj.keys():
    work_t_adj[i] = [item for sublist in work_t_adj[i] for item in sublist]

# Strategy 2: Specify the lower bound of overstaffing for each type of worker hired at different times. It can help generate overstaffing to ensure meal time. #16 is the minimum LB for LIM Tentativo3rd 210725- RC.csv. 3 is the minimu for TentativoCM 210725- RC.csv
Overstaff_LB = []  # [4]#[16] # Fill in the value inside bracket
Overstaff_time_ctrl = day_time  # [time_seq('07/25/2021 23:00', '07/25/2021 23:45')+ time_seq('07/25/2021 0:00', '07/25/2021 1:15')] #day_time #[item for sublist in meal_ctrl_range for item in sublist]  #['07/25/2021 0:15', '07/25/2021 0:30', '07/25/2021 23:30', '07/25/2021 23:45']


# If any, manually specify at which time no workers are assigned, e.g., the moment after no-demand period ends
prac_constr11 = []  # For example, ['06/01/2021 7:45','06/02/2021 17:45']

# If any, manusally specify at which time at least some workers must be assigned
prac_constr12 = (
    []
)  # ['07/25/2021 0:30'] #['07/25/2021 13:00'] #For example, ['06/01/2021 12:15'] or ['06/01/2021 8:00'] etc.


# Main code for running scheduling problem.
# This problem determines how many different types of staffs should be assigned at each time (output)
# The input the problem is the demand over time, type of workers, worker type cost, overstaffing cost
# The optimization is to minimize the overall overstaffing cost plus salaries
# The constraints are demand coverage and practical heuristics (if any, e.g., specify no workers allocated at some times)

# A function that can allow for circular demand
def getRangeList(mainList, startIndexInMainList):
    s = mainList[startIndexInMainList::]
    b = len(mainList) - len(s)
    shifted_time_slots = s + mainList[0:b]
    return shifted_time_slots


# A simple function of adding days to time slots
def appenddays(days, time_slots):
    day_time = []
    time_slotsd = {}
    if time_slots != []:
        for d in days:
            for t in time_slots:
                day_time.append(d + " " + t)
    return day_time


# A dictionary of 'Date': 'Date Time'
time_slotd = {}
for i in file_dfc["Date"].unique():
    time_slotd[i] = [
        file_dfc["rangos"][j] for j in file_dfc[file_dfc["Date"] == i].index
    ]

# List of all demands over time in its original sequence
demand = file_dfc[resource_type].tolist()

# create a dictionary 'Date Time': Demand
workforce_demand_daily = dict(zip(file_dfc["rangos"], file_dfc[resource_type]))

work_type = list(
    worktype_info.keys()
)  # List of worker types, Full Time, Part time etc.
work_gentype = list(work_LB.keys())
FullTimeWk = work_type[0:5]

work_gentype_val = [FullTimeWk]

for j in range(5, len(work_type)):
    work_gentype_val.append([work_type[j]])

work_gentype_dict = dict(zip(work_gentype, work_gentype_val))

shift_hours = [
    worktype_info[j]["shift_hours"] for j in work_type
]  # Extract hours per shift for each type of worker in a list
shift_early_release = [
    worktype_info[j]["early_release"] for j in work_type
]  # Extract early release for each type of worker in a list
actual_shift_hours = [ai - bi / 60 for ai, bi in zip(shift_hours, shift_early_release)]
shift = [
    int(element * 4) for element in actual_shift_hours
]  # Unit: Time slots. Currently 1/4 hr per time slot. So, 16 means 4 hrs per shift, 8 means 2 hrs per shift etc.
type_shift = dict(
    zip(work_type, shift)
)  # A dictionary for worker's type vs. their shift (time slots)

meal_beginplus = [
    worktype_info[j]["Lunch_Begin_Shift_Start_Offset_Plus"] * 4 for j in work_type
]  # Meal time begin plus/minus, for estimating meal time window
work_shift_meal_beginplus = dict(zip(work_type, meal_beginplus))
meal_beginminus = [
    worktype_info[j]["Lunch_Begin_Shift_Start_Offset_Minus"] * 4 for j in work_type
]
work_shift_meal_beginminus = dict(zip(work_type, meal_beginminus))
meal_time = [
    worktype_info[j]["meal_mins"] / 15 for j in work_type
]  # Extract meal time and save the information in a list
work_meal_time = dict(
    zip(work_type, meal_time)
)  # A dictionary for worker's type vs. meal time
if width_hrs != []:
    w = int(width_hrs * 4)
else:
    w = 0

# cost_salary = [worktype_info[j]['salary'] for j in work_type]
cost_salary = [
    worktype_info[j]["salary"]
    + (1 - worktype_info[j]["meal_in_shift"]) * worktype_info[j]["meal_mins"] / 60
    for j in work_type
]
work_salary = dict(
    zip(work_type, cost_salary)
)  # Worker type's salary vs. worker type. Used for cost optimization

c = (
    {}
)  # c  is used to calculate FTE and NH. FTE does not differentiate Full time Day, Full time night/morning etc. though their working hours per shift are different
for k in work_type:
    if work_salary[k] >= 7 / 8:
        c[k] = 1
    else:
        c[k] = work_salary[k]

workforce_indx = {day_time[i]: i for i in range(0, len(day_time))}
# Pratical constraints or heuristics:

# Heuristic 1: Do not assign any workers at a time if zero demand at that moment except (15 min before a non-zero demand happens).
prac_constr2 = [t for t in day_time if (workforce_demand_daily[t] == 0)]
# Make sure 15 min before the non-zero demand can be allowed to activate workers
del_key = []
for t in day_time:
    if workforce_indx[t] + 1 < len(workforce_indx):
        if demand[workforce_indx[t]] == 0 and demand[workforce_indx[t] + 1] != 0:
            del_key.append(t)
    else:
        if demand[len(workforce_indx) - 1] == 0 and demand[0] != 0:
            del_key.append(t)

for i in range(0, len(del_key)):
    prac_constr2.remove(del_key[i])  #  Except 15 mins before a non-zero demand

# Heuristic 2: Determine working block hours for different types of workers. For example, full time workers are available only for a few time slots in a day
time_block = [[e for e in day_time]]
for i in range(0, len(work_type) - 1):
    time_block.append([e for e in day_time])
del_range = [worktype_info[j]["work_range"] for j in work_type]

for j in range(0, len(del_range)):
    del_range[j] = appenddays(days, del_range[j])
    for i in range(0, len(del_range[j])):
        if del_range[j][i] in time_block[j]:
            time_block[j].remove(del_range[j][i])

workblock = dict(
    zip(work_type, time_block)
)  # Dictionary: Worker type vs. time slots at which they are not activated

# Heuristic 3: The block time specified to allow for briefing time. The block time is when demand increases. No workers will be activated at this moment.
t_bfb = []
for t in day_time:
    if workforce_indx[t] - 1 >= 0:
        if demand[workforce_indx[t]] > demand[workforce_indx[t] - 1]:
            t_bfb.append(t)
    else:
        if demand[0] > demand[len(workforce_indx) - 1]:
            t_bfb.append(t)

# Start constructing the optimization model to cover demand
workforce_model = ConcreteModel()

workforce_model.n = Var(
    work_type, day_time, domain=NonNegativeIntegers
)  # How many workers of each type at each time to be activated
workforce_model.y = Var(
    day_time, domain=Binary
)  # Indicator variable y: when some workers are activated==> count activation frequency
workforce_model.ya = Var(
    day_time, domain=Binary
)  # Indicator variable ya for some If then constraints
workforce_model.yk = Var(
    work_type, day_time, domain=Binary
)  # Indicator variable yk for some If then constraints

M = 100000  # A big M notation to linearize If then constraint
workforce_model.dmd = ConstraintList()

Overstaff = 0
CumuStaff_t = dict(zip(day_time, []))
Overstaff_t = dict(zip(day_time, []))

for t in day_time:
    # By default, we consider circular shift. E.g., Workers activated at 22:00 may impact the workforce demand at 0:15 next day. There is no starting/end time.
    CumuStaff = 0
    CumuStaffb = 0

    for k in work_type:
        if circular == 1:  # if we consider circular demand over periodical times
            # Estimate the index of the time slots whose workers still have their shift active
            if (
                workforce_indx[t] - type_shift[k] + 1 < 0
            ):  # workforce_indx[t]-type_shift[k]+1 ensures the starting shift can cover the time t
                sft = (workforce_indx[t] - type_shift[k] + 1) % len(workforce_indx)
                shifted_time = getRangeList(
                    day_time, sft
                )  # starting shift time for the current t is at the beginning of shifted_time. t=0:00, k=part time 4, shifted_time = [20:15 ... 0:00]
                ts = shifted_time[0 : type_shift[k]]
                tsb = shifted_time[0 : type_shift[k] - 1]
                # Not correct- tsb.insert(0,shifted_time[-1])  #The first element should include starting shift time for t-1 that is at the end of shifted_time. (20:00, ... 23:45)
                # if t == '06/01/2021 0:00' and k == 'Part Time 4':
                #  print(shifted_time)
            else:
                ts = day_time[
                    (workforce_indx[t] - type_shift[k] + 1) : (workforce_indx[t] + 1)
                ]
                tsb = day_time[
                    (workforce_indx[t] - type_shift[k] + 1) : workforce_indx[t]
                ]
        else:
            # If the shift has a starting time, then the cumulative activated staffs are estimated from
            ts = day_time[
                max(workforce_indx[t] - type_shift[k] + 1, 0) : workforce_indx[t] + 1
            ]
            if (
                workforce_indx[t] == 0
            ):  # if non-circular, at the beginning, we have to ensure people are hired to cover the non-zero demand. No briefing for them.
                tsb = day_time[
                    max(workforce_indx[t] - type_shift[k] + 1, 0) : workforce_indx[t]
                    + 1
                ]
                # print(t,tsb)
            else:
                tsb = day_time[
                    max(workforce_indx[t] - type_shift[k] + 1, 0) : workforce_indx[t]
                ]

        # Example: t =10:00, then ts = 6:15, 6:30... 10:00 (cumulative activated staffs at t).
        #                         tsb = 6:15, 6:30... 9:45 (cumulative activated staffs at 10:00 excluding new hire at 10:00 should exceed the demand at 10:00)
        #   if t == '06/01/2021 8:00' and k == 'Full Time Day':
        #     print(k, t, ts)
        #     print(k, t, tsb)

        CumuStaff += sum(workforce_model.n[k, t] for t in ts)
        CumuStaffb += sum(workforce_model.n[k, t] for t in tsb)
    CumuStaff_t[t] = CumuStaff

    Overstaff_t[t] = CumuStaff_t[t] - workforce_demand_daily[t]
    Overstaff += CumuStaff - workforce_demand_daily[t]
    # print(Overstaff_m)

    workforce_model.dmd.add(
        CumuStaffb >= workforce_demand_daily[t]
    )  # Cumulative activated workers at time slot t excluding the new activation at t > demand(t), ensuring new hires have briefing time.
# time_slotd[days[0]]
# Constraints of lower bound and upper bound of each type of workers
for d in days:
    for k1 in work_gentype:
        if work_UB[k1] != []:
            workforce_model.dmd.add(
                sum(
                    workforce_model.n[k, t]
                    for t in time_slotd[d]
                    for k in work_gentype_dict[k1]
                )
                <= work_UB[k1]
            )  # or t in day_time
        if work_LB[k1] != []:
            workforce_model.dmd.add(
                sum(
                    workforce_model.n[k, t]
                    for t in time_slotd[d]
                    for k in work_gentype_dict[k1]
                )
                >= work_LB[k1]
            )
    # Verify if upper/lower bound calculation is correct:
    # print(sum(workforce_model.n[k,t] for t in day_time for k in work_gentype_dict[k1]))
    # Verify if the output results are making sense
    # print(t,workforce_demand_daily[t], CumuStaff)

for k1 in work_gentype:
    for adj in range(len(work_adj_LB[k1])):
        if work_adj_LB[k1][adj] != []:
            workforce_model.dmd.add(
                sum(
                    workforce_model.n[k, t]
                    for t in work_t_adj[k1][adj]
                    for k in work_gentype_dict[k1]
                )
                >= work_adj_LB[k1][adj]
            )

        # print(sum(workforce_model.n[k,t] for t in work_t_adj[k1][adj] for k in work_gentype_dict[k1]),work_adj_LB[k1][adj])

# Apply practical constraints below or any heuristics:

# Indicator variable y[t], equals 1 if any workers are assigned at time t, otherwise 0
for t in day_time:
    workforce_model.dmd.add(
        sum(workforce_model.n[k, t] for k in work_type)
        >= 1 - M * (1 - workforce_model.y[t])
    )
    workforce_model.dmd.add(
        sum(workforce_model.n[k, t] for k in work_type) <= M * workforce_model.y[t]
    )

if (
    prac_constr11 != []
):  # Constraints from heuristic 1.1: Manually specify some time slots when no workers will be assigned
    for t in prac_constr11:
        workforce_model.dmd.add(workforce_model.y[t] == 0)

if (
    prac_constr12 != []
):  # Constraints from heuristic 1.2: Manually specify some time slots when some workers must be assigned
    for t in prac_constr12:
        workforce_model.dmd.add(workforce_model.y[t] >= 1)

if prac_constr2 != []:
    for (
        t
    ) in (
        prac_constr2
    ):  # Constraints from heuristic 2: No workers are assigned when no demand exist at that time except briefing time slots
        workforce_model.dmd.add(workforce_model.y[t] == 0)

for k in work_type:
    if (
        workblock[k] != []
    ):  # Constraints from Heuristics 3: the working hour ranges of worker types
        for t in workblock[k]:
            workforce_model.dmd.add(workforce_model.n[k, t] == 0)

# Constraints from heuristic 4: The block time specified to allow for briefing time
if switch_blocktime_briefing == 1:
    if t_bfb != []:
        for t in t_bfb:
            workforce_model.dmd.add(workforce_model.y[t] == 0)
# Set an upper bound for the activation frequency each day
# for d in days:
#  if worker_actfreq_UP[d]!=[]:
#    workforce_model.dmd.add(sum(workforce_model.y[t] for t in time_slotsd[d]) <= worker_actfreq_UP[d])

# Heuristic 4: Set a lower bound for the activation interval

for t in day_time:
    if circular == 1:
        if workforce_indx[t] - w < 0:
            sftal = (workforce_indx[t] - w) % len(workforce_indx)
            shifted_timeal = getRangeList(day_time, sftal)
            tsa_L = shifted_timeal[0:w]
        else:
            tsa_L = day_time[(workforce_indx[t] - w) : (workforce_indx[t])]

        if workforce_indx[t] + w + 1 > len(workforce_indx):
            sftau = (workforce_indx[t] + w + 1) % len(workforce_indx)
            shifted_timeau = getRangeList(day_time, sftau)
            shifted_timeau = shifted_timeau[::-1]
            tsa_U = shifted_timeau[0:w]

            tsa_U = tsa_U[::-1]
        else:
            tsa_U = day_time[workforce_indx[t] + 1 : (workforce_indx[t] + w + 1)]
    else:
        tsa_L = day_time[max(workforce_indx[t] - w, 0) : (workforce_indx[t])]
        tsa_U = day_time[
            workforce_indx[t] + 1 : min(workforce_indx[t] + w + 1, len(workforce_indx))
        ]
    tsa = tsa_L + tsa_U
    # print(t, tsa)

    if tsa != []:
        # if workforce_model.y[t]>=1 then {workforce_model.dmd.add(workforce_model.y[tsa] == 0 ), for tsa in [t-w, t+w]\t] in day_time
        workforce_model.dmd.add(workforce_model.y[t] <= 0 + M * workforce_model.ya[t])
        for ta in tsa:
            workforce_model.dmd.add(
                workforce_model.y[ta] <= 0 + M * (1 - workforce_model.ya[t])
            )

# Record meal time range for different workers. Only for individual scheduling. Not used for demand coverage optimization.
tmeal_kt = {}
for k in work_type:
    if work_meal_time[k] != 0 and worktype_info[k]["meal_in_shift"] != 0:
        tmeal_kt[k] = {}
        for t in day_time:
            tmeal_kt[k][t] = []

# Hueristic 5: Set a lower bound for overstaffing during the meal time for those activated workers; Also record the meal time window at the same time.
tmeal = []

for t in day_time:
    # Circular checking time_slots after future shift hour exceeds len(day_time)
    for k in work_type:
        if work_meal_time[k] != 0 and worktype_info[k]["meal_in_shift"] != 0:
            if circular == 1:
                if workforce_indx[t] + type_shift[k] - work_shift_meal_beginminus[
                    k
                ] > len(workforce_indx):
                    sftm = (
                        workforce_indx[t]
                        + type_shift[k]
                        - work_shift_meal_beginminus[k]
                    ) % len(workforce_indx)
                    shifted_timem = getRangeList(day_time, sftm)
                    shifted_timem = shifted_timem[::-1]
                    meal_length = (
                        type_shift[k]
                        - work_shift_meal_beginminus[k]
                        - work_shift_meal_beginplus[k]
                    )
                    tmeal = shifted_timem[0:meal_length]
                    tmeal = tmeal[::-1]
                else:
                    tmeal = day_time[
                        (workforce_indx[t] + work_shift_meal_beginplus[k]) : (
                            workforce_indx[t]
                            + type_shift[k]
                            - work_shift_meal_beginminus[k]
                        )
                    ]
            else:
                tmeal = day_time[
                    (workforce_indx[t] + work_shift_meal_beginplus[k]) : min(
                        workforce_indx[t]
                        + type_shift[k]
                        - work_shift_meal_beginminus[k],
                        len(workforce_indx),
                    )
                ]
            # print(k, t, tmeal)   # Verify the meal time window for the eligible worker types

            # if workforce_model.n[k,t]>= 1, then sum(CumuStaff_t[tss] - workforce_demand_daily[tss] for tss in tmeal) >=Overstaff_LB
            # Equivalently to say, it means workforce_model.n[k,t]<=0 or sum(CumuStaff_t[tss] - workforce_demand_daily[tss] for tss in tmeal) >=Overstaff_LB or both

            for tj in range(
                len(Overstaff_LB)
            ):  # tj - an index for each time range where overstaffing needes to be controlled
                if (
                    Overstaff_LB[tj] > 0
                    and tmeal != []
                    and t in Overstaff_time_ctrl[tj]
                ):
                    workforce_model.dmd.add(
                        workforce_model.n[k, t] <= 0 + M * workforce_model.yk[k, t]
                    )
                    workforce_model.dmd.add(
                        sum(
                            CumuStaff_t[tss] - workforce_demand_daily[tss]
                            for tss in tmeal
                        )
                        >= Overstaff_LB[tj] - M * (1 - workforce_model.yk[k, t])
                    )

            # Record tmeal_kt (worker type, day_time) only for meal time check in inidividual scheduling in the future. NOT involved in the optimization here.
            tmeal_kt[k][t].append(tmeal)
            tmeal_kt[k][t] = tmeal_kt[k][t][0]

switch_salary = 1  # minimization of working salaries and overstaffing can be equivalent. Sometimes, we may just minimize overstaffing by supressing total salaries
workforce_model.TotalCost = Objective(
    expr=sum(
        work_salary[k] * workforce_model.n[k, t] for k in work_type for t in day_time
    )
    * switch_salary
    + cost_overstaff * Overstaff
    + active_cost * sum(workforce_model.y[t] for t in day_time)
    + FTE_opt
    * sum(
        sum(workforce_model.n[k, t] for t in day_time) / productivity * c[k]
        for k in work_type
    )
    + NH_opt
    * sum(
        sum(workforce_model.n[k, t] for t in day_time) / productivity for k in work_type
    ),
    sense=minimize,
)


results = SolverFactory("cbc").solve(workforce_model)
results.write()

for t in day_time:
    # print(workforce_model.y[t], "=", workforce_model.y[t]())
    for k in work_type:
        if workforce_model.n[k, t]() != 0.0:
            print(workforce_model.n[k, t], "=", workforce_model.n[k, t]())

print("Objective value is: ", workforce_model.TotalCost())

print(
    "Total Workers' Salaries are:  ",
    sum(
        work_salary[k] * workforce_model.n[k, t]() for k in work_type for t in day_time
    ),
)
print(
    "Total Number of Workers is: ",
    sum(workforce_model.n[k, t] for k in work_type for t in day_time)(),
)

NH = sum(
    int(round(sum(workforce_model.n[k, t]() for t in day_time) / productivity, 0))
    for k in work_type
)
print("NH (Normal Heads) is  ", NH)

FTE = sum(
    int(round(sum(workforce_model.n[k, t]() for t in day_time) / productivity, 0))
    * c[k]
    for k in work_type
)
print("FTE value is  ", FTE)

print(
    "Total Number of Activation (Frequency) is: ",
    sum(workforce_model.y[t]() for t in day_time),
)

# name of csv file
filename = "workforce_scheduling_" + data_file
fields = [
    "Date and Time",
    "Worker Type",
    "Staffs types and number to be activated at each time",
    "Total Workers",
    "NH",
    "FTE",
    "Activation",
]
rows = []

for t in day_time:
    for k in work_type:
        if workforce_model.n[k, t]() != 0.0:
            rows.append([t, k, workforce_model.n[k, t]()])

rows[0].append(sum(workforce_model.n[k, t] for k in work_type for t in day_time)())
rows[0].append(NH)
rows[0].append(FTE)
rows[0].append(sum(workforce_model.y[t]() for t in day_time))

# writing to csv file
with open(filename, "w") as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # writing the fields
    csvwriter.writerow(fields)
    # writing the data rows
    csvwriter.writerows(rows)
