# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from lxml import etree

###############################################################################
# XML file fefinitions
###############################################################################

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
</GDAL_WMS>
"""

gdal_twms = """<GDAL_WMS>
    <Service name="TiledWMS">
        <ServerUrl></ServerUrl>
        <TiledGroupName></TiledGroupName>
        <Change key="${time}"></Change>
    </Service>
</GDAL_WMS>
"""

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
</GDAL_WMTS>
"""


class GIBSLayer:
    """GIBS Layer"""
    def __init__(self, title, layer_name, epsg, format, image_resolution, tile_resolution, time):
        self.title = title
        self.layer_name = layer_name
        self.epsg = epsg
        self.format = format
        format_extensions = {"GTiff": "tiff", "JPEG": "jpg", "PNG": "png"}
        self.format_suffix = format_extensions[format]

        self.image_resolution = image_resolution # Image resolution (e.g. 2km, 1km, 250m, etc.)
        self.tile_resolution = tile_resolution # Tile resolution (must be lower than image resolution)

        self.time = time

        self.gibs_xml = ""

        self.image_location = ""
        self.legend_location = ""
    
    def generate_xml(self, protocol, datestring):
        """
        Populate the XML file with data fields
        Arguments: 
            protocol: "tms" or "twms"
            datestring: date in string format
        """
        # Use the tiled WMS service
        parser = etree.XMLParser(strip_cdata=False)
        if protocol == "twms":
            xml = etree.fromstring(gdal_twms, parser)

            # Define the service URL (Tiled WMS)
            for Service in xml.findall('Service'):
                Service.find('ServerUrl').text = "https://gibs.earthdata.nasa.gov/twms/epsg"+self.epsg+"/best/twms.cgi?"
                Service.find('TiledGroupName').text = self.title + " tileset" # TiledGroupName is the layer title concatenated with 'tileset'
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
            xml.find("Projection").text = "EPSG:" + self.epsg

            # Zoom Level (Projections & Resolution)
            if self.epsg == "3413" or self.epsg == "3031":
                # Polar Projection
                tile_level = {"2km": "2", "1km": "3", "500m": "4", "250m": "5"}

                # Define the service URL (TMS)
                for Service in xml.findall('Service'):
                    Service.find('ServerUrl').text = "https://gibs.earthdata.nasa.gov/wmts/epsg"+self.epsg+"/best/"+self.layer_name+"/default/{Time}/"+self.image_resolution+"/${z}/${y}/${x}."+ self.format_suffix
                for DataWindow in xml.findall('DataWindow'):
                    # Use -4194304, 4194304, 4194304, -4194304 for the Polar projections. 
                    # This is the bounding box of the topmost tile, which matches the bounding box of the actual imagery for Polar but not for Geographic.
                    DataWindow.find('UpperLeftX').text = "-4194304"
                    DataWindow.find('UpperLeftY').text = "4194304"
                    DataWindow.find('LowerRightX').text = "4194304"
                    DataWindow.find('LowerRightY').text = "-4194304"
                    DataWindow.find('TileLevel').text = tile_level[self.tile_resolution]
                    # Use 2, 2 for Polar projections
                    DataWindow.find('TileCountX').text = "2"
                    DataWindow.find('TileCountY').text = "2"
            else:
                # Geographic Projection
                tile_level = {"2km": "5", "1km": "6", "500m": "7", "250m": "8", "31.25m": "11"}

                # Define the service URL (TMS)
                for Service in xml.findall('Service'):
                    Service.find('ServerUrl').text = "https://gibs.earthdata.nasa.gov/wmts/epsg"+self.epsg+"/best/"+self.layer_name+"/default/{Time}/"+self.image_resolution+"/${z}/${y}/${x}."+ self.format_suffix
                for DataWindow in xml.findall('DataWindow'):
                    # Use -180.0, 90, 396.0, -198 for Geographic projection
                    DataWindow.find('UpperLeftX').text = "-180.0"
                    DataWindow.find('UpperLeftY').text = "90"
                    DataWindow.find('LowerRightX').text = "396.0"
                    DataWindow.find('LowerRightY').text = "-198"
                    DataWindow.find('TileLevel').text = tile_level[self.tile_resolution]
                    # Use 2, 1 for Geographic projection
                    DataWindow.find('TileCountX').text = "2"
                    DataWindow.find('TileCountY').text = "1"       
                
        pretty_xml = etree.tostring(xml, pretty_print=True)
        self.gibs_xml = pretty_xml.decode()