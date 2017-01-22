#!/bin/bash
# kaggle cookie can be downloaded using a chrome extension and copy pasted to the root dir, search "wget kaggle"
# https://www.kaggle.com/c/belkin-energy-disaggregation-competition/forums/t/5118/downloading-data-via-wget?forumMessageId=49609
mkdir -p ../input
for f in sample_submission.csv.zip grid_sizes.csv.zip train_wkt_v4.csv.zip sixteen_band.zip three_band.zip
do
    wget https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/download/$f -O ../input/$f --load-cookies ../kaggle-cookie
    unzip -o ../input/$f -d ../input
    rm ../input/$f
done
