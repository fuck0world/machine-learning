import pandas as pd
import numpy as np
data_path = 'E:\\code\\python\\qoe_model\\raw_data\\3w_data.csv'

def choose_data(data_path, score_name, choose_num):
    
    final_DataFram = pd.DataFrame()
    for i in range(1,6):
        data = pd.read_csv(data_path)
        data = data.loc[data[score_name] == i]
        data = data.reset_index(drop = True)
        row = data.iloc[:,0].size - 1
        data = data.loc[np.random.randint(0, row, choose_num).tolist()]
        final_DataFram = pd.concat([final_DataFram, data])
    final_DataFram = final_DataFram.reset_index(drop = True)
        
    return final_DataFram

USER_SCORE = choose_data(data_path, 'USER_SCORE', 400)
print(USER_SCORE)