import pandas as pd
import numpy as np
import ast
import os


def sigmoid(x):
    return 1/(1 + np.exp(-(x.astype(float))))

def string_to_nplist(x):
    if pd.isnull(x):
        return []
    else:
        return np.array(ast.literal_eval(x))

def read_data(file_name, col_names, col_list):
    data = pd.read_csv(file_name, names=col_names, delimiter="|")
    for col in col_list:
        data[col] = data[col].apply(lambda x: string_to_nplist(x))
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data.set_index(["DateTime"], inplace=True)
    return data

def clean_lob(data, cols_need, cols_check, weight_mid_price=0.5, num_level=10):
    lst_valid_samples = []
    mid_prices = []
    for ind, row in data.iterrows():
        if len(row[cols_check[0]]) and len(row[cols_check[1]]):
            if (row[cols_check[0]].shape[0] == num_level) and (row[cols_check[1]].shape[0] == num_level):
                lst_valid_samples.append(ind)
                mid_p = weight_mid_price * row[cols_check[0]][0] + (1 - weight_mid_price) * row[cols_check[1]][0]
                mid_prices.append(mid_p)
    ret_data = pd.DataFrame(index=lst_valid_samples, data=data.loc[lst_valid_samples, cols_need])
    ret_data["Midprice"] = mid_prices
    return ret_data

def my_func(a):
    all_items = np.concatenate(a.values)
    return np.mean(all_items), np.std(all_items)

def zscore_normalization(data, cols_need, freq="5H", min_periods=4*12*60):
    z_score_cols, stat_data = [], []
    for col in cols_need:
        rolling_col = data[col].rolling(window=freq, min_periods=min_periods)
        col_lst_mean_std = [my_func(a) for a in rolling_col]
        mu_col = "Mu" + col
        std_col = "Std" + col
        z_score_col = "Zscore" + col
        tmp_data = pd.DataFrame(data=col_lst_mean_std, columns=[mu_col, std_col], index=data.index)
        tmp_data.index = tmp_data.index.shift(1, freq="H")
        idx_intersect = list(set(tmp_data.index).intersection(set(data.index)))
        tmp_data[col] = np.nan
        tmp_data.loc[idx_intersect, col] = data.loc[idx_intersect, col]
        tmp_data = tmp_data.dropna()
        tmp_data[z_score_col] = (tmp_data[col] - tmp_data[mu_col]) / tmp_data[std_col]
        tmp_data = tmp_data[[z_score_col]]
        stat_data.append(tmp_data)
        z_score_cols.append(z_score_col)
    ret_data = pd.concat(stat_data, axis=1)
    return z_score_cols, ret_data


if __name__ == "__main__":
    cols_LOB = ["DateTime","Open","High","Low","Last","Volume","NumTrades","BidVolume","AskVolume","SumBid","SumAsk","BidPrices","BidVolumes","AskPrices","AskVolumes"]
    col_list_LOB = ["BidPrices","BidVolumes","AskPrices","AskVolumes"]

    lob_path = ".\DeepLOB"
    file_name = os.path.join(lob_path, "LOB_NQU22-CME_2_1_10_10level.lob")

    data = read_data(file_name, cols_LOB, col_list_LOB)
    print(data.shape)
    data = data.loc[data.index[:5000]]
    print(data.shape)

    z_score_cols, ret_data = zscore_normalization(data, col_list_LOB)

    cols_check = [z_score_cols[0], z_score_cols[2]]
    data_cleaned = clean_lob(ret_data, z_score_cols, cols_check)
    print(data_cleaned.head())

    data_cleaned["ConcatLOB"] = data_cleaned[z_score_cols].apply(lambda x: np.concatenate(x.values), axis=1)
    print(data_cleaned["ConcatLOB"].tail(1).values[0].shape)
    print(data_cleaned["ConcatLOB"].tail(1).values[0])

    