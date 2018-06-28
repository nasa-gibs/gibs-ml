# gibs_ml
Machine learning for anomaly detection in Global Imagery Browse Services ([GIBS](https://earthdata.nasa.gov/about/science-system-description/eosdis-components/global-imagery-browse-services-gibs)) Earth satellite imagery.

# Dependencies
To install the [GDAL translator library](http://www.gdal.org/) run ```conda install gdal```

# Download Data
```download_data.py``` downloads a GIBS dataset where each layer for a day is a single image or tiled (see ```--tiled_world``` flag). The script uses ```gdal_translate``` with the [GIBS API](https://wiki.earthdata.nasa.gov/display/GIBS/GIBS+API+for+Developers#GIBSAPIforDevelopers-ServiceEndpointsandGetCapabilities). Images are saved in the ```data/``` directory by default. 

```
arguments:
  --time_begin          The date from which to begin searching back in format YYYY-MM-DD.  Default:  Today
  --time_stop           The date to stop searching in format YYYY-MM-DD.  Default:  Date of last check or Today-20
  
  --epsg                The numeric EPSG code of the map projection {4326:geographic, 3413:arctic, 3031:antarctic}.  Default:  4326 (geographic)
  
  --tiled_world         Flag to download the entire world as a series of tiled images.
  --tile_resolution     The zoom resolution of the tiles. Must be lower than the image resolution of layer.  Default:  8km
  
  --num_threads         Number of concurrent threads to launch to download images.  Default:  10
  --output_dir          Full path of output directory.  Default:  ./data
```

# Data Preprocessing
```augment_data.py``` augment the training set by rotations (90, 180, 270 degrees) and flips (horizontal and vertical).
