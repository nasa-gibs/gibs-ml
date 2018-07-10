# gibs_ml
Machine learning for anomaly detection in Global Imagery Browse Services ([GIBS](https://earthdata.nasa.gov/about/science-system-description/eosdis-components/global-imagery-browse-services-gibs)) Earth satellite imagery.

# Dependencies
Run ```conda install -c conda-forge tqdm``` to install tqdm.
Run ```conda install future``` to install future.
Run ```conda install -c conda-forge gdal ``` to install the [GDAL translator library](http://www.gdal.org/). 

# Download Data
Run ```download_data.py``` to download a GIBS layer dataset. The script uses [```gdal_translate```](http://www.gdal.org/gdal_translate.html) to query the [GIBS API](https://wiki.earthdata.nasa.gov/display/GIBS/GIBS+API+for+Developers#GIBSAPIforDevelopers-ServiceEndpointsandGetCapabilities).

Images are saved in the ```data/``` directory by default. The folder structure will be ```data/{EPSG code}/{YYYY-MM-DD}/{Layer Name}.{Image Format}```. Image format (i.e. PNG, JPEG) is predefined by the ```layer_name```. For our purposes, we work with satellite instruments that take a single image of the globe each day.

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

Set ```--tile_resolution``` to set the tile_resolution with the corresponding output resolution (img_resolution) of the image according to the table below. 

Set the ```--tiled_world``` flag to download the layer for each day as a collection of 512x512 tiles (see tiled_resolution for number of tiles returned).

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
```split_data.py``` generates a text file ```{Layer Name}.txt``` with a labels (train, val, test) for each date. You still have to hand label the anomalies though!

# Notebooks
### Unsupervised Approach

#### Missing Data Detection. [```missing_data_detection.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/missing_data_detection.ipynb). 
Uses image processing techniques to automatically detect missing data holes in an image. 

### Handcrafted Featurizatization Approach
Each image has computed a Histogram of Oriented Gradients (HOG) as well as a color histogram using the hue channel in HSV color space. Roughly speaking, HOG should capture the texture of the image while ignoring color information, and the color histogram represents the color of the input image while ignoring texture. The final feature vector for each image is formed by concatenating the HOG and color histogram feature vectors. See [```features.py```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/features.py) for implementation details.

#### Linear Classification. [```linear_classifier.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/linear_classifier.ipynb). 
Linear classifer with both SVM (hinge) and softmax loss functions. The softmax classifier outputs probabilities. 

#### Neural Network. [```neural_net.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/neural_net.ipynb). 
A 2-Layer fully connected neural network that uses softmax to output probabilities.

### Deep End-to-End Approaches 

#### Vanilla CNN. [```cnn.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/cnn.ipynb). 
The image is passed through 3 layers of ```conv > bn > max_pool > relu```, followed by flattening the image and then applying 2 fully connected layers. Implemented using PyTorch framework.

#### Transfer Learning. [```pretrained-cnn.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/pretrained-cnn.ipynb). 
121-layer DenseNet pretrained on the Imagenet dataset. The last fully connected layer is retrained on the our dataset. It is important to note the difference between the distribution of the ImageNet dataset and our own dataset. Implemented using PyTorch framework.

# Other
```gibs_layer.py``` contains several predefined GIBS layers as well as the XML formats for [TMS and Tiled WMS](http://www.gdal.org/frmt_wms.html) services to request layers from the GIBS API using a gdal driver.

```augment_data.py``` augments the training dataset by rotations (90, 180, 270 degrees) and flips (horizontal and vertical).

See examples using ```gdal_translate``` with TMS and Tiled WMS services [here](https://wiki.earthdata.nasa.gov/display/GIBS/Map+Library+Usage#expand-GDALBasics).