import sys
import gnss
import gzip
import tarfile
import argparse
import progressbar
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from io import BytesIO, StringIO
from functools import reduce

sns.set_style('darkgrid')


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--stations")
    p.add_argument("--date", type=np.datetime64, default=None)
    p.add_argument("--output", default='gsod.csv')
    return p


def read_stations(station_file):
    stations = pd.read_csv(station_file)

    good = reduce(np.logical_and, [stations['LAT'] != 0.,
                                   stations['LON'] != 0.,
                                   stations['ELEV(M)'] > -999.,
                                   stations['USAF'] != '999999',
                                   stations['END'] >= 20180101,
                                   stations['CTRY'] == 'US',
                                   stations['LAT'] >= 25.,
                                   stations['LAT'] <= 50.,
                                   stations['LON'] <= -60.,
                                   stations['LON'] >= -125.,
                                   ~stations['STATE'].isnull(),
                                   ~stations['STATION NAME'].str.contains('BUOY').astype('bool'),
                                   ])
    stations = stations[good]

    stations = stations[['USAF', 'WBAN', 'LAT', 'LON', 'ELEV(M)']]
    ecef = gnss.ecef_from_llh([stations['LAT'].values,
                               stations['LON'].values,
                               stations['ELEV(M)'].values])
    for k, v in zip(['X', 'Y', 'Z'], ecef):
        stations[k] = v

    stations.drop_duplicates(subset='USAF', inplace=True)
    return stations.set_index('USAF', drop=False)


def extract_date(df, date):
    if date is None:
        return df
    else:
        return df[df['YEARMODA'] == int(pd.to_datetime(date).strftime('%Y%m%d'))]

def get_station_from_member(member):
    return member.name.split('-')[0].strip('./')


def add_station_info(df, stations):
    stations.index = stations.index.astype('S')
    merged = stations.merge(df, how='inner', left_index=True, right_on='STATION')
    return merged.set_index('STATION', drop=False)


def iter_data(data_file, station_ids):
    pbar = progressbar.ProgressBar()
    with tarfile.open(data_file) as tf:
        print("Extracting data for required stations")
        for member in pbar(tf.getmembers()):
            if get_station_from_member(member) in station_ids:
                fobj = BytesIO(tf.extractfile(member.name).read())
                gzf = gzip.GzipFile(fileobj=fobj)
                data_string = BytesIO(gzf.read())
                df = pd.read_fwf(data_string)
                df = df.rename(columns={'STN---': 'STATION'})
                df['STATION'] = df['STATION'].astype('S')
                dates = pd.to_datetime(df['YEARMODA'], format='%Y%m%d')
                df['DAY_OF_YEAR'] = dates.dt.dayofyear
                yield df


if __name__ == "__main__":

    p = create_parser()
    args = p.parse_args()

    stations = read_stations(args.stations)
    observations = [extract_date(df, args.date)
                    for df in iter_data(args.input, stations['USAF'].values)]
    obs = pd.concat(observations)
    obs = add_station_info(obs, stations)
    
    obs = obs[['LAT', 'LON', 'ELEV(M)', 'X', 'Y', 'Z', 'TEMP', 'DAY_OF_YEAR']]
    obs.index = obs.index.astype('int64')
    obs.to_csv(args.output)
