#!/bin/bash

RESOURCE_PATH=/proj/uppmax2024-2-5/shared/lecture3-inclass
FILENAMES=(IMG_4997.ppm  IMG_5008.ppm  IMG_5067.ppm  IMG_5385.ppm  \
    IMG_5400.ppm IMG_5002.ppm  IMG_5013.ppm  IMG_5260.ppm  \
    IMG_5387.ppm  IMG_5440.ppm)

for filename in ${FILENAMES[@]}
do
    ln -s $RESOURCE_PATH/$filename
done

echo "Done making links"
