import sys
import gnss
import pyproj
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits import basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set_style('darkgrid')

def geographic_distance(lon0, lat0, lon1, lat1, ellps="sphere"):
    """
    Computes the distance (in meters) between two points
    assuming a particular earth ellipse model.
    """
    geod = pyproj.Geod(ellps=ellps)
    return geod.inv(lon0, lat0, lon1, lat1)[-1]


def bounding_box(lons, lats, pad=0.1, lon_pad=None, lat_pad=None):

    lon_diffs = lons[:, None] - lons
    lat_diffs = lats[:, None] - lats

    western_most_ind = np.nonzero(np.all(lon_diffs >= 0., axis=0))[0]
    western_most = np.unique(lons[western_most_ind]).item()
    eastern_most_ind = np.nonzero(np.all(lon_diffs <= 0., axis=0))[0]
    eastern_most = np.unique(lons[eastern_most_ind]).item()

    northern_most_ind = np.nonzero(np.all(lat_diffs <= 0., axis=0))[0]
    northern_most = np.unique(lats[northern_most_ind]).item()
    southern_most_ind = np.nonzero(np.all(lat_diffs >= 0., axis=0))[0]
    southern_most = np.unique(lats[southern_most_ind]).item()

    # count the number of lons greater than and less than each lon
    # and take the difference.  The longitude (or pair of lons) that
    # minimize this help us determine the median.  This allows different
    # definitions of longitude.
    lon_rel_loc = np.abs(np.sum(lon_diffs >= 0., axis=0) -
                         np.sum(lon_diffs <= 0., axis=0))
    central_lons = lons[lon_rel_loc == np.min(lon_rel_loc)]
    # make sure the central two aren't too far apart.
    assert np.max(central_lons) - np.min(central_lons) < 90
    median_lon = np.median(central_lons)

    lat_rel_loc = np.abs(np.sum(lat_diffs >= 0., axis=0) -
                         np.sum(lat_diffs <= 0., axis=0))
    central_lats = lats[lat_rel_loc == np.min(lat_rel_loc)]
    median_lat = np.median(central_lats)

    width = geographic_distance(western_most, median_lat,
                                       eastern_most, median_lat)
    height = geographic_distance(median_lon, northern_most,
                                        median_lon, southern_most)
    if lon_pad is None:
        lon_pad = pad * np.abs(eastern_most - western_most)
    if lat_pad is None:
        lat_pad = pad * np.abs(northern_most - southern_most)

    return {'llcrnrlon': western_most - lon_pad,
            'urcrnrlon': eastern_most + lon_pad,
            'urcrnrlat': northern_most + lat_pad,
            'llcrnrlat': southern_most - lat_pad,
            'lon_0': median_lon,
            'lat_0': median_lat}


def get_basemap(lons, lats, pad=0.1, lat_pad=None, lon_pad=None, **kwdargs):
    kwdargs['projection'] = kwdargs.get('projection', 'cyl')
    kwdargs['resolution'] = kwdargs.get('resolution', 'i')
    bm_args = bounding_box(lons, lats, pad=pad, lat_pad=lat_pad, lon_pad=lon_pad)
    bm_args.update(kwdargs)
    # explicitly specify axis, even if its just the gca.
    bm_args['ax'] = bm_args.get('ax', None)
    m = basemap.Basemap(**bm_args)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    return m

def add_lat_lon_height(df):
    llh = [gnss.llh_from_ecef(xyz)
           for xyz in preds[['ecef_x', 'ecef_y', 'ecef_z']].values]
    lat, lon, h = zip(*llh)
    df['lat'] = np.round(lat, 8)
    df['lon'] = np.round(lon, 8)
    df['height'] = np.round(h, 8)
    return df

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("train")
    p.add_argument("predictions")
    return p

if __name__ == "__main__":

    p = create_parser()
    args = p.parse_args()

    # read in the training and prediction data
    preds = pd.read_csv(args.predictions)
    preds = add_lat_lon_height(preds)
    pred_lats = preds['lat'].values
    pred_lons = preds['lon'].values

    df = pd.read_csv(args.train)
    lons = df['LON'].values
    lats = df['LAT'].values

    norm = plt.Normalize(vmin=df['TEMP'].values.min(), vmax=df['TEMP'].values.max())

    ncols = np.nonzero(np.diff(preds['lat'].values) != 0)[0][0] + 1
    assert ncols > 1
    nrows = float(preds.shape[0]) / ncols
    if not (nrows - int(nrows) == 0.):
        raise ValueError("Couldn't infer data shape")
    nrows = int(nrows)

    def plot_variable(variable_name, units, **kwdargs):

        fig, axes = plt.subplots(1, 1, figsize=(16, 6))
        axes = np.array(axes).reshape(-1)

        bm = get_basemap(pred_lons, pred_lats, pad=0.05, ax=axes[0])

        # Here we plot the variable as a transparent mesh.
        pred_x, pred_y = bm(pred_lons, pred_lats)
        pred_values = preds[variable_name].values
        land_values = basemap.maskoceans(pred_lons.reshape(nrows, ncols),
                                         pred_lats.reshape(nrows, ncols),
                                         pred_values.reshape(nrows, ncols), inlands=False)

        pm = bm.pcolormesh(pred_x.reshape(nrows, ncols),
                      pred_y.reshape(nrows, ncols),
                      land_values.reshape(nrows, ncols),
                      alpha=0.7,
                      norm=norm,
                      cmap=plt.get_cmap('coolwarm'),
                      linewidths=0, zorder=10000, **kwdargs)

        # Then overlay the points where we have observations.
        x, y = bm(lons, lats)
        sc = bm.scatter(x, y,
                        c='k',
                        edgecolor='k',
                        alpha=0.5,
                        s=10, cmap=plt.get_cmap('coolwarm'),
                        norm=norm, linewidths=1, zorder=10000)

        fig.tight_layout()

        divider = make_axes_locatable(axes[0])

        cax = fig.add_axes([0.92, 0.05, 0.02, 0.7])

        plt.colorbar(pm, ax=axes[0], cax=cax, label=units)

    plot_variable('prediction', "Degrees F")
    plt.savefig("mean_temperature.png", norm=norm)

    plot_variable('prediction_variance', "Standard Deviation (F)", vmax=3.0)
    plt.savefig("sd_temperature.png")
