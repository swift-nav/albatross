#!/bin/bash -x

if [ ! -f ./gsod_2018.tar ]; then
    wget ftp://ftp.ncdc.noaa.gov/pub/data/gsod/2018/gsod_2018.tar
fi

if [ ! -f ./ish-history.csv ]; then
    wget ftp://ftp.ncdc.noaa.gov/pub/data/gsod/ish-history.csv
fi

if [ ! -f ./elev.0.5-deg.nc ]; then
    wget http://research.jisao.washington.edu/data/elevation/elev.0.5-deg.nc
fi

python make_gsod_csv.py --input ./gsod_2018.tar --stations ./isd-history.csv --output ./gsod.csv --date "2018-05-01"
python make_prediction_locations.py --input elev.0.5-deg.nc --output ./prediction_locations.csv

cd ../../
mkdir -p build
cd build
cmake ../
make temperature_example

./examples/temperature_example -input ../examples/temperature_example/gsod.csv -predict ../examples/temperature_example/prediction_locations.csv -output ./predictions.csv
python ../examples/temperature_example/plot_temperature_example.py ../examples/temperature_example/gsod.csv ./predictions.csv
