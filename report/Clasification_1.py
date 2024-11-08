import pandas as pd
import os

# Get the current working directory
current_folder = os.getcwd()
relative_path = '../RawData/2000_to_2024.xlsx'
full_path = os.path.join(current_folder, relative_path)
# get raw data from excel file
df = pd.read_excel(full_path)
## Preprocessing
# 1. Drop signals with is_closed = False
df = df[df['is_closed'] == True]
# 2. Drop signals with same signal_time
df = df.drop_duplicates(subset=['signal_time'])
# 3. Normalize rsi_value due to trend_type : if trend_type = -1 then rsi_value = 100 - rsi_value
df['rsi_value'] = df.apply(lambda x: 100 - x['rsi_value'] if x['trend_type'] == -1 else x['rsi_value'], axis=1)
# 4. Drop columns that are not needed including 'is_closed', 'trend_type', 'signal_time', 'signal_price',
# 'tp_1_hit', 'tp_2_hit', 'tp_3_hit', 'tp_4_hit', 'tp_5_hit', 'tp_6_hit', 'tp_7_hit', 'tp_8_hit', 
# last_stop_tp_1_number, last_stop_tp_2_number, last_stop_tp_3_number, last_stop_tp_4_number, last_stop_tp_5_number,
# last_stop_tp_6_number, last_stop_tp_7_number, last_stop_tp_8_number, rd_filter_passed , hd_filter_passed
df = df.drop(columns=['max_price_move','pair_name','max_seen_value','min_seen_value','sl_2_hit','sl_1_hit', 'is_closed', 'trend_type', 'signal_time', 'signal_price', 'tp_1_hit', 'tp_2_hit', 'tp_3_hit', 'tp_4_hit', 'tp_5_hit', 'tp_6_hit', 'tp_7_hit', 'tp_8_hit', 'last_stop_tp_1_number', 'last_stop_tp_2_number', 'last_stop_tp_3_number', 'last_stop_tp_4_number', 'last_stop_tp_5_number', 'last_stop_tp_6_number', 'last_stop_tp_7_number', 'last_stop_tp_8_number', 'rd_filter_passed', 'hd_filter_passed'])
