# gibs_ml
Machine learning for anomaly detection in Global Imagery Browse Services ([GIBS](https://earthdata.nasa.gov/about/science-system-description/eosdis-components/global-imagery-browse-services-gibs)) Earth satellite imagery.

# Dependencies
Run ```conda install -c conda-forge tqdm``` to install tqdm.
Run ```conda install future``` to install future.
Run ```conda install -c conda-forge gdal ``` to install the [GDAL translator library](http://www.gdal.org/). 

# Download Data
Run ```download_data.py``` to download a GIBS layer dataset. The script uses [```gdal_translate```](http://www.gdal.org/gdal_translate.html) to query the [GIBS API](https://wiki.earthdata.nasa.gov/display/GIBS/GIBS+API+for+Developers#GIBSAPIforDevelopers-ServiceEndpointsandGetCapabilities).

Images are saved in the ```data/``` directory by default. The folder structure will be ```data/{EPSG code}/{YYYY-MM-DD}/{Layer Name}.{Image Format}```. Image format (i.e. PNG, JPEG) is predefined by the ```layer_name```.

```
arguments:
  --layer_name          The layer name to download.  Default:  VIIRS_SNPP_CorrectedReflectance_TrueColor

  --start_date          The date from which to begin (inclusive) searching back in format YYYY-MM-DD.  Default:  None (uses layer start date)
  --end_date            The date to stop (non-inclusive) searching in format YYYY-MM-DD (or "Today").  Default:  Date of last check or Today

  --epsg                The numeric EPSG code of the map projection: 4326 (geographic), 3413 (arctic), 3031 (antarctic).  Default:  4326
  
  --tiled_world         Flag to download the entire world as a series of tiled images.

  --tile_resolution     The distance corresponding to a pixel in the image. Must more coarse than the native image resolution of the layer.  Default:  16km
  
  --num_threads         Number of concurrent threads to launch to download images.  Default:  10

  --output_dir          Name path of the output directory.  Default:  data
```

Set ```--tile_resolution``` to decide the output resolution (img_resolution) of the image according to the table below. 

Set the ```--tiled_world``` flag to download the layer for each day as a collection of 512x512 tiles.

| tile_resolution 	| tile_level 	|   img_resolution  	| tiled_resolution 	|
|:---------------:	|:----------:	|:-----------------:	|:----------------:	|
|       16km      	|      2     	|    (4096,2048)    	|       (8,4)      	|
|       8km       	|      3     	|    (8192,4096)    	|      (16,8)      	|
|       4km       	|      4     	|    (16384,8192)   	|      (32,16)     	|
|       2km       	|      5     	|   (32768,16384)   	|      (64,32)     	|
|       1km       	|      6     	|   (65536,32768)   	|     (128,64)     	|
|       500m      	|      7     	|   (131072,65536)  	|     (256,128)    	|
|       250m      	|      8     	|  (262144,131072)  	|     (512,256)    	|
|       125m      	|      9     	|  (524288,262144)  	|    (1024,512)    	|
|      62.5m      	|     10     	|  (1048576,524288) 	|    (2048,1024)   	|
|      31.25m     	|     11     	| (2097152,1048576) 	|    (4096,2048)   	|

For example, ```download_data.py``` downloads the `VIIRS_SNPP_CorrectedReflectance_TrueColor` layer since the instrument began collecting data (i.e. '2015-11-24'). Each date will have a single image of the globe stitched together by GDAL. 

# Split Data
```split_data.py``` generates a text file with dates for the labels. You still have to hand label the anomalies though!

# Data Preprocessing
Run ```augment_data.py``` to augment the training dataset by rotations (90, 180, 270 degrees) and flips (horizontal and vertical).

# Notebooks

### Unsupervised Approach (```missing_data_detection.ipynb```)

### Linear Classification (```linear_classifier.ipynb```)

### Neural Network (```neural_net.ipynb```)

### Vanilla CNN (```cnn.ipynb```)

### DenseNet-121 CNN (```pretrained-cnn.ipynb```)

# Other
```gibs_layer.py``` contains several predefined GIBS layers as well as the XML formats for [TMS and Tiled WMS](http://www.gdal.org/frmt_wms.html) services to request layers from the GIBS API using a gdal driver.

See examples using ```gdal_translate``` with TMS and Tiled WMS services [here](https://wiki.earthdata.nasa.gov/display/GIBS/Map+Library+Usage#expand-GDALBasics).