# gibs_ml
Machine learning for anomaly detection in Global Imagery Browse Services ([GIBS](https://earthdata.nasa.gov/about/science-system-description/eosdis-components/global-imagery-browse-services-gibs)) Earth satellite imagery.

## Dependencies
Run ```conda install -c conda-forge tqdm``` to install tqdm.

Run ```conda install future``` to install future.

Run ```conda install -c conda-forge gdal ``` to install the [GDAL translator library](http://www.gdal.org/). 

# Dataset Preparation

## Download Data
Run ```download_data.py``` to download a GIBS layer dataset. The script uses [```gdal_translate```](http://www.gdal.org/gdal_translate.html) to query the [GIBS API](https://wiki.earthdata.nasa.gov/display/GIBS/GIBS+API+for+Developers#GIBSAPIforDevelopers-ServiceEndpointsandGetCapabilities).

Images are saved in the ```data/``` directory by default. The folder structure will be ```data/{EPSG code}/{YYYY-MM-DD}/{Layer Name}.{Image Format}```. Image format (i.e. PNG, JPEG) is predefined by the ```layer_name```. For our purposes, we work with layers of the globe that are available each day.

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

Set ```--tile_resolution``` to set the tile_resolution with the corresponding output resolutions (see pixel_resolution) according to the table below. If the ```--tiled_world``` flag is set this single image resolution is split up into a grid of tiled_resolution. This will download the layer for each day as a collection of num_tiles tiles. Each tile is a 512x512 image. 

| tile_resolution 	| tile_level 	|  pixel_resolution 	| tiled_resolution 	| num_tiles 	|
|:---------------:	|:----------:	|:-----------------:	|:----------------:	|:---------:	|
|       16km      	|      2     	|    (4096,2048)    	|       (8,4)      	|     32    	|
|       8km       	|      3     	|    (8192,4096)    	|      (16,8)      	|    128    	|
|       4km       	|      4     	|    (16384,8192)   	|      (32,16)     	|    512    	|
|       2km       	|      5     	|   (32768,16384)   	|      (64,32)     	|    2048   	|
|       1km       	|      6     	|   (65536,32768)   	|     (128,64)     	|    8192   	|
|       500m      	|      7     	|   (131072,65536)  	|     (256,128)    	|   32768   	|
|       250m      	|      8     	|  (262144,131072)  	|     (512,256)    	|   131072  	|
|       125m      	|      9     	|  (524288,262144)  	|    (1024,512)    	|   524288  	|
|      62.5m      	|     10     	|  (1048576,524288) 	|    (2048,1024)   	|  2097152  	|
|      31.25m     	|     11     	| (2097152,1048576) 	|    (4096,2048)   	|  8388608  	|

For example, by default, ```download_data.py``` downloads the `VIIRS_SNPP_CorrectedReflectance_TrueColor` layer since the instrument began collecting data (i.e. '2015-11-24') up to today. Each date will have a single 4096x2048 image of the globe stitched together by GDAL. If the ```--tile_resolution``` flag were set for each date we would retrieve the 32 tiles in the (8,4) grid.

*NOTE: ```download_tiled_data.py``` was added to rapidly download tiled images for layers whose tiles were only accessible via URL.*  

## Split Data
```split_data.py``` generates a text file ```{Layer Name}.txt``` with a split (i.e. train, val, test) for each date. You still have to hand label the anomalies though!

# Experiments

## Unsupervised Labeling
We explore (automated) unsupervised techniques to label the images and the pixels. Currently we analyze MODIS and VIIRS layers, but this analysis can be extended to other datasets.

#### Image-Level Missing Data Detection. [```image_labeling_missing_data_viirs.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/image_labeling_missing_data_viirs.ipynb), [```image_labeling_missing_data_modis_terra.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/image_labeling_missing_data_modis_terra.ipynb), [```image_labeling_missing_data_modis_aqua.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/image_labeling_missing_data_modis_aqua.ipynb). 
Uses image processing techniques and morphological operations to automatically detect missing data holes in an image. 

#### Pixel-Level Anomaly Detection. [```pixel_labeling_miscoloration_viirs.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/pixel_labeling_miscoloration_viirs.ipynb), [```pixel_labeling_missing_data_viirs.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/pixel_labeling_missing_data_viirs.ipynb). 
Uses image processing techniques, basic statistical analysis, and density-based clustering methods to automatically detect anomalous pixel values in an image. 

## Handcrafted Featurizatization Approach
Each image has computed a Histogram of Oriented Gradients (HOG) as well as a color histogram using the hue channel in HSV color space. Roughly speaking, HOG should capture the texture of the image while ignoring color information, and the color histogram represents the color of the input image while ignoring texture. 

The final feature vector for each image is formed by concatenating the HOG and color histogram feature vectors. See [```features.py```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/features.py) for implementation details.

#### Linear Classification. [```linear_classifier.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/linear_classifier.ipynb). 
Linear classifer with both SVM (hinge) and Softmax loss functions. The Softmax classifier outputs probabilities. Note that the probabilities computed by the Softmax classifier are better thought of as confidences where, similar to the SVM, the ordering of the scores is interpretable, but the absolute numbers (or their differences) technically are not.

#### Random Forest Classifier. [```random_forest.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/random_forest.ipynb), [```random_forest_pixels.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/random_forest_pixels.ipynb). 
A Random Forest classifier with tree depth of 5. Notebooks for image-level detection and pixel-level detection. 

#### Neural Network. [```neural_net.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/neural_net.ipynb). 
A 2-Layer fully connected neural network that uses a Softmax classifer to output probabilities.

## End-to-End Approaches 

#### Vanilla CNN. [```cnn.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/cnn.ipynb), [```cnn_pixels.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/cnn_pixels.ipynb). 
Our simple CNN architecture is 3 layers of ```conv > bn > max_pool > relu```, followed by flattening the image and then applying 2 fully connected layers. Implemented using PyTorch framework.

#### Transfer Learning. [```pretrained_cnn.ipynb```](https://github.jpl.nasa.gov/xue/gibs_ml/blob/master/pretrained_cnn.ipynb). 
121-layer DenseNet pretrained on the Imagenet dataset. The last fully connected layer is retrained on our dataset. It is important to note the difference between the distribution of the ImageNet dataset and our own dataset. Implemented using PyTorch framework.

# Other

```gibs_layer.py``` contains a GIBS layer class with several predefined GIBS layers as well as the XML formats for [TMS and Tiled WMS](http://www.gdal.org/frmt_wms.html) services to request layers from the GIBS API using a gdal driver.

```utils.py``` contains various helper functions for PyTorch and date manipulation.

See examples using ```gdal_translate``` with TMS and Tiled WMS services [here](https://wiki.earthdata.nasa.gov/display/GIBS/Map+Library+Usage#expand-GDALBasics).