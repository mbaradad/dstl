#!/bin/bash
#kaggle cookie can be downloaded using a chrome extension and copy pasted to the root dir, search "wget kaggle"
mkdir -p ../input
for f in sample_submission.csv.zip grid_sizes.csv.zip sixteen_band.zip three_band.zip train_wkt_v2.csv.zip
do
    wget https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/download/$f -O ../input/$f --load-cookies ../kaggle-cookie
    unzip ../input/$f -d ../input
    rm ../input/$f
done
