# Copyright 2018 California Institute of Technology.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from lxml import etree

###############################################################################
# XML file definitions
###############################################################################

# See XML documentation for WMS (http://www.gdal.org/frmt_wms.html)
gdal_wms = """<GDAL_WMS>
    <Service name="TMS">
        <ServerUrl></ServerUrl>
    </Service>
    <DataWindow>
        <UpperLeftX></UpperLeftX>
        <UpperLeftY></UpperLeftY>
        <LowerRightX></LowerRightX>
        <LowerRightY></LowerRightY>
        <TileLevel></TileLevel>
        <TileCountX></TileCountX>
        <TileCountY></TileCountY>
        <YOrigin>top</YOrigin>
    </DataWindow>
    <Projection></Projection>
    <BlockSizeX>512</BlockSizeX>
    <BlockSizeY>512</BlockSizeY>
    <BandsCount>3</BandsCount>
    <Timeout>3000</Timeout>
</GDAL_WMS>"""

gdal_twms = """<GDAL_WMS>
    <Service name="TiledWMS">
        <ServerUrl></ServerUrl>
        <TiledGroupName></TiledGroupName>
        <Change key="${time}"></Change>
    </Service>
</GDAL_WMS>"""

# See XML documentation for WTMS (http://www.gdal.org/frmt_wmts.html)
gdal_wmts = """<GDAL_WMTS>
  <GetCapabilitiesUrl></GetCapabilitiesUrl>
  <Layer></Layer>
  <Style></Style>
  <TileMatrixSet></TileMatrixSet>
  <DataWindow>
    <UpperLeftX></UpperLeftX>
    <UpperLeftY></UpperLeftY>
    <LowerRightX></LowerRightX>
    <LowerRightY></LowerRightY>
  </DataWindow>
  <BandsCount>4</BandsCount>
  <Cache/>
  <UnsafeSSL>true</UnsafeSSL>
  <ZeroBlockHttpCodes>404</ZeroBlockHttpCodes>
  <ZeroBlockOnServerException>true</ZeroBlockOnServerException>
</GDAL_WMTS>"""

class GIBSLayer:
    """GIBS Layer"""
    def __init__(self, title, layer_name, format, image_resolution, date_min):
        self.title = title # title to use with TiledGroupName
        self.layer_name = layer_name
        self.format = format
        self.image_resolution = image_resolution # predefined image resolution (e.g. 2km, 1km, 250m, etc.)
        self.date_min = date_min # datestring when layer started collecting data

        format_extensions = {"GTiff": "tiff", "JPEG": "jpg", "PNG": "png"}
        self.format_suffix = format_extensions[format]
        self.gibs_xml = "" 

        self.image_location = ""
        self.legend_location = ""

    def get_gibs_layer(layer_name):
        ###############################################################################
        # Fixed layer definitions (static method)
        # Most popular layers for MODIS Terra and VIIRS SNNP
        # Note: Most definitions found here (https://wiki.earthdata.nasa.gov/display/GIBS/GIBS+Available+Imagery+Products#expand-ReferenceLayers9Layers)
        ###############################################################################

        # MODIS \ Terra, Aqua
        MODIS_Terra_CorrectedReflectance_TrueColor = GIBSLayer(title="MODIS TERRA", layer_name="MODIS_Terra_CorrectedReflectance_TrueColor", format="JPEG", image_resolution="250m", date_min="2003-01-01")
        MODIS_Terra_CorrectedReflectance_Bands367 = GIBSLayer(title="MODIS TERRA, Bands 367", layer_name="MODIS_Terra_CorrectedReflectance_Bands367", format="JPEG", image_resolution="250m", date_min="2003-01-01")
        MODIS_Terra_CorrectedReflectance_Bands721 = GIBSLayer(title="MODIS TERRA, Bands 721", layer_name="MODIS_Terra_CorrectedReflectance_Bands721", format="JPEG", image_resolution="250m", date_min="2003-01-01")
        MODIS_Aqua_CorrectedReflectance_TrueColor = GIBSLayer(title="MODIS AQUA", layer_name="MODIS_Aqua_CorrectedReflectance_TrueColor", format="JPEG", image_resolution="250m", date_min="2003-01-01")
        MODIS_Aqua_CorrectedReflectance_Bands721 = GIBSLayer(title="MODIS AQUA, Bands 721", layer_name="MODIS_Aqua_CorrectedReflectance_Bands721", format="JPEG", image_resolution="250m", date_min="2003-01-01")
        
        # MODIS No Data Masks
        MODIS_Terra_Data_No_Data = GIBSLayer(title="MODIS TERRA Data No Data", layer_name="MODIS_Terra_Data_No_Data", format="PNG", image_resolution="250m", date_min="2003-01-01")
        MODIS_Aqua_Data_No_Data = GIBSLayer(title="MODIS AQUA Data No Data", layer_name="MODIS_Aqua_Data_No_Data", format="PNG", image_resolution="250m", date_min="2003-01-01")

        # VIIRS \ SNPP
        VIIRS_SNPP_CorrectedReflectance_TrueColor = GIBSLayer(title="VIIRS SNPP True Color", layer_name="VIIRS_SNPP_CorrectedReflectance_TrueColor", format="JPEG", image_resolution="250m", date_min="2015-11-24")
        VIIRS_SNPP_CorrectedReflectance_BandsM11_I2_I1 = GIBSLayer(title="VIIRS SNPP Bands M11-I2-I1", layer_name="VIIRS_SNPP_CorrectedReflectance_BandsM11-I2-I1", format="JPEG", image_resolution="250m", date_min="2015-11-24")
        VIIRS_SNPP_CorrectedReflectance_BandsM3_I3_M11 = GIBSLayer(title="VIIRS SNPP Bands M3-I3-M11", layer_name="VIIRS_SNPP_CorrectedReflectance_BandsM3-I3-M11", format="JPEG", image_resolution="250m", date_min="2015-11-24")
        VIIRS_SNPP_Brightness_Temp_BandI5_Day = GIBSLayer(title="VIIRS SNPP Brightness Temp BandI5 Night", layer_name="VIIRS_SNPP_Brightness_Temp_BandI5_Day", format="PNG", image_resolution="250m", date_min="2017-09-17")

        # VIIRS Fire (WMS only)
        VIIRS_SNPP_Fires_375m_Day = GIBSLayer(title=None, layer_name="VIIRS_SNPP_Fires_375m_Day", format="PNG", image_resolution="1km", date_min="2015-11-25")
        VIIRS_SNPP_Fires_375m_Night = GIBSLayer(title=None, layer_name="VIIRS_SNPP_Fires_375m_Night", format="PNG", image_resolution="1km", date_min="2015-11-25")

        # MODIS Fire (Day and Night) (WMS Only)
        MODIS_Fires_All = GIBSLayer(title=None, layer_name="MODIS_Fires_All", format="PNG", image_resolution="1km", date_min="2012-05-08")

        # MODIS Data Masks
        MODIS_Terra_Land_Surface_Temp_Day = GIBSLayer(title="MODIS TERRA Daytime Land Surface Temperature", layer_name="MODIS_Terra_Land_Surface_Temp_Day", format="PNG", image_resolution="1km", date_min="2003-01-01")
        MODIS_Terra_Chlorophyll_A = GIBSLayer(title="MODIS Terra Chlorophyll A", layer_name="MODIS_Terra_Chlorophyll_A", format="PNG", image_resolution="1km", date_min="2013-07-02")
        MODIS_Terra_NDVI_8Day = GIBSLayer(title="MODIS Terra NDVI 8Day", layer_name="MODIS_Terra_NDVI_8Day", format="PNG", image_resolution="250m", date_min="2016-07-30")

        # VIIRS Data Masks
        VIIRS_SNPP_DayNightBand_ENCC = GIBSLayer(title="VIIRS SNPP DayNightBand ENCC", layer_name="VIIRS_SNPP_DayNightBand_ENCC", format="PNG", image_resolution="500m", date_min="2016-11-30")

        # Organize within dictionaries
        basemap_layer_dict = {
            # "MODIS_Terra_CorrectedReflectance_TrueColor": MODIS_Terra_CorrectedReflectance_TrueColor,
            # "MODIS_Terra_CorrectedReflectance_Bands367": MODIS_Terra_CorrectedReflectance_Bands367,
            "VIIRS_SNPP_CorrectedReflectance_TrueColor": VIIRS_SNPP_CorrectedReflectance_TrueColor, 
        }

        modis_layer_mask_dict = {
            "MODIS_Terra_Chlorophyll_A": MODIS_Terra_Chlorophyll_A,
            "MODIS_Terra_Land_Surface_Temp_Day": MODIS_Terra_Land_Surface_Temp_Day,
            "MODIS_Terra_NDVI_8Day": MODIS_Terra_NDVI_8Day,
            "MODIS_Terra_Data_No_Data": MODIS_Terra_Data_No_Data,
        }

        viirs_layer_mask_dict = {
            "VIIRS_SNPP_DayNightBand_ENCC": VIIRS_SNPP_DayNightBand_ENCC, 
            "VIIRS_SNPP_Brightness_Temp_BandI5_Day": VIIRS_SNPP_Brightness_Temp_BandI5_Day, 
        }

        all_layer_dict = {
            "MODIS_Terra_CorrectedReflectance_TrueColor": MODIS_Terra_CorrectedReflectance_TrueColor,
            "MODIS_Terra_CorrectedReflectance_Bands367": MODIS_Terra_CorrectedReflectance_Bands367,
            "MODIS_Terra_CorrectedReflectance_Bands721": MODIS_Terra_CorrectedReflectance_Bands721,
            "MODIS_Aqua_CorrectedReflectance_TrueColor": MODIS_Aqua_CorrectedReflectance_TrueColor,
            "MODIS_Aqua_CorrectedReflectance_Bands721": MODIS_Aqua_CorrectedReflectance_Bands721,

            "MODIS_Terra_Data_No_Data": MODIS_Terra_Data_No_Data,
            "MODIS_Aqua_Data_No_Data": MODIS_Aqua_Data_No_Data,

            "MODIS_Fires_All": MODIS_Fires_All,

            "MODIS_Terra_Chlorophyll_A": MODIS_Terra_Chlorophyll_A,
            "MODIS_Terra_Land_Surface_Temp_Day": MODIS_Terra_Land_Surface_Temp_Day,
            "MODIS_Terra_NDVI_8Day": MODIS_Terra_NDVI_8Day,

            "VIIRS_SNPP_CorrectedReflectance_TrueColor": VIIRS_SNPP_CorrectedReflectance_TrueColor,
            "VIIRS_SNPP_CorrectedReflectance_BandsM11_I2_I1": VIIRS_SNPP_CorrectedReflectance_BandsM11_I2_I1,
            "VIIRS_SNPP_CorrectedReflectance_BandsM3_I3_M11": VIIRS_SNPP_CorrectedReflectance_BandsM3_I3_M11,
            "VIIRS_SNPP_Brightness_Temp_BandI5_Day": VIIRS_SNPP_Brightness_Temp_BandI5_Day,

            "VIIRS_SNPP_Fires_375m_Day": VIIRS_SNPP_Fires_375m_Day,
            "VIIRS_SNPP_Fires_375m_Night": VIIRS_SNPP_Fires_375m_Night,

            "VIIRS_SNPP_DayNightBand_ENCC": VIIRS_SNPP_DayNightBand_ENCC, 
        }

        
        if layer_name in all_layer_dict:
            return all_layer_dict[layer_name]
        else:
            return None

    def generate_xml(self, protocol, epsg, tile_resolution, datestring):
        """
        Populate the XML file with data fields
        Arguments: 
            protocol: "tms" or "twms"
            tile_resolution: <TileLevel> field for TMS service (i.e. zoom level)
            datestring: date in string format
        """

        # Use the tiled WMS service
        parser = etree.XMLParser(strip_cdata=False, remove_blank_text=True)
        if self.title is not None and protocol == "twms":
            xml = etree.fromstring(gdal_twms, parser)

            # Define the service URL (Tiled WMS)
            for Service in xml.findall('Service'):
                Service.find('ServerUrl').text = "https://gibs.earthdata.nasa.gov/twms/epsg"+epsg+"/best/twms.cgi?"
                Service.find('TiledGroupName').text = self.title + " tileset"  # TiledGroupName is the layer title concatenated with 'tileset'
                Service.find('Change').text = datestring
                
        # Use TMS service
        else:
            xml = etree.fromstring(gdal_wms, parser)

            # Number of bands/channels
            # 1 for grayscale data, 3 for RGB, 4 for RGBA. (optional, defaults to 4)
            if self.format.lower() == "png": 
                xml.find("BandsCount").text = "4"
            else:
                xml.find("BandsCount").text = "3"
            if self.layer_name == "VIIRS_SNPP_DayNightBand_ENCC":
                xml.find("BandsCount").text = "1"

            # Declared projection (defaults to value of the TileMatrixSet)
            xml.find("Projection").text = "EPSG:" + epsg

            # Define the service URL (TMS)
            for Service in xml.findall('Service'):
                Service.find('ServerUrl').text = "https://gibs.earthdata.nasa.gov/wmts/epsg"+epsg+"/best/"+self.layer_name+"/default/{Time}/"+self.image_resolution+"/${z}/${y}/${x}."+ self.format_suffix

            # Zoom Level (Projections & Resolution)
            if epsg == "3413" or epsg == "3031":
                # Polar Projection
                tile_level = {"2km": "2", "1km": "3", "500m": "4", "250m": "5"}
                for DataWindow in xml.findall('DataWindow'):
                    # Use -4194304, 4194304, 4194304, -4194304 for the Polar projections. 
                    # This is the bounding box of the topmost tile, which matches the bounding box of the actual imagery for Polar but not for Geographic.
                    DataWindow.find('UpperLeftX').text = "-4194304"
                    DataWindow.find('UpperLeftY').text = "4194304"
                    DataWindow.find('LowerRightX').text = "4194304"
                    DataWindow.find('LowerRightY').text = "-4194304"
                    DataWindow.find('TileLevel').text = tile_level[tile_resolution]
                    # Use 2, 2 for Polar projections
                    DataWindow.find('TileCountX').text = "2"
                    DataWindow.find('TileCountY').text = "2"
            else:
                # Geographic Projection
                tile_level = {"16km":"2", "8km": "3", "4km": "4", "2km": "5", "1km": "6", "500m": "7", "250m": "8", "31.25m": "11"}
                for DataWindow in xml.findall('DataWindow'):
                    # Use -180.0, 90, 396.0, -198 for Geographic projection
                    DataWindow.find('UpperLeftX').text = "-180.0"
                    DataWindow.find('UpperLeftY').text = "90"
                    DataWindow.find('LowerRightX').text = "396.0"
                    DataWindow.find('LowerRightY').text = "-198"
                    DataWindow.find('TileLevel').text = tile_level[tile_resolution]
                    # Use 2, 1 for Geographic projection
                    DataWindow.find('TileCountX').text = "2"
                    DataWindow.find('TileCountY').text = "1"       
                
        pretty_xml = etree.tostring(xml, pretty_print=False)
        self.gibs_xml = pretty_xml.decode()