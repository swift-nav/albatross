"""
Loads elevation data and creates a CSV which can be used to
feed into temperature_example.cc in order to make predictions
at a grid over the earth surface for CONUS.
"""
import os
import sys
import gnss
import urllib2
import argparse
import numpy as np
import pandas as pd
import xarray as xra

from cStringIO import StringIO

_elevation_data_url = "http://research.jisao.washington.edu/data/elevation/elev.0.5-deg.nc"
        

def create_parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", default=os.path.basename(_elevation_data_url))
    p.add_argument("--output", default='prediction_locations.csv')
    return p

if __name__ == "__main__":

    p = create_parser()
    args = p.parse_args()

    if not os.path.exists(args.input):
        print("Downloading elevation data to: %s" % args.input)
        with open(args.input, 'w') as fout:
            web_data = urllib2.urlopen(_elevation_data_url)
            fout.write(web_data.read())
            web_data.close()

    print("Loading elevation data from: %s" % args.input)
    elev = xra.open_dataset(args.input).isel(time=0)['data']
    # Longitudes in the file are in 0, 360 but we want -180, 180
    elev['lon'] = np.mod(elev['lon'] + 180., 360.) - 180.

    # Select only CONUS
    lat_inds = np.logical_and(elev['lat'] >= 25., elev['lat'] <= 50.)
    lon_inds = np.logical_and(elev['lon'] >= -125., elev['lon'] <= -60.)
    conus = elev.isel(lat=lat_inds, lon=lon_inds)

    # Setup the columns names to match what the c++ code expects.
    locations = conus.to_dataframe().reset_index()
    locations.rename(columns={'lat': 'LAT', 'lon': 'LON', 'data': 'ELEV(M)'},
                     inplace=True)
    locations.drop('time', axis=1, inplace=True)
    ecef = gnss.ecef_from_llh([locations['LAT'].values,
                               locations['LON'].values,
                               locations['ELEV(M)'].values])
    for k, v in zip(['X', 'Y', 'Z'], ecef):
        locations[k] = v

    locations['TEMP'] = np.nan
    locations['STATION'] = np.arange(locations.shape[0])
    locations.set_index('STATION', inplace=True, drop=False)
    locations = locations[['LAT', 'LON', 'ELEV(M)', 'X', 'Y', 'Z', 'TEMP']]

    print("Writing output to: %s" % args.output)
    locations.to_csv(args.output)
