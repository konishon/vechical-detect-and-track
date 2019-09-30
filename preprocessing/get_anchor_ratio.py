import argparse
import glob
import math
import pandas as pd


def get_aspect_ratio(w, h):
    return w / h


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def map_to_ratio(row):
    return get_aspect_ratio(row['w'], row['h']).as_integer_ratio()


def load_as_df(files):
    li = []
    for filename in files:
        df = pd.read_csv(filename)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True, type=str, help="path to input csv f")
    args = vars(ap.parse_args())
    path = args['folder']

    all_files = glob.glob(path + "/*.csv")
    print(all_files)
    df = load_as_df(all_files)

    df['ratio'] = df.apply(map_to_ratio, axis=1)
    grouped = df.groupby('ratio').size()
    print("Found {} groups".format(len(grouped)))
    top_ten = grouped.nlargest(10)
    print("Top ten {} groups".format(top_ten))
