"""
|==================================== navlib/plot/geoplot.py ======================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/plot/geoplot.py                                                                |
|  @brief    Plot methods for map based figures.                                                   |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
"""

import io
import numpy as np
import matplotlib.pyplot as plt

import cartopy as ct
import cartopy.crs as ccrs
import cartopy.geodesic as cgeo
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.geoaxes import GeoAxes
from planar import BoundingBox
from shapely.geometry import LineString
from PIL import Image

from urllib.request import urlopen, Request

# import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import planar


def geoplot(lat, lon, tiles="satellite", fig=None, ax=None, figsize=None, plot_init_pos=False, **kwargs):
    match tiles:
        case "map":
            cimgt.OSM.get_image = __image_spoof
            img = cimgt.OSM()  # spoofed, downloaded street map

        case "satellite":
            cimgt.QuadtreeTiles.get_image = __image_spoof
            img = cimgt.QuadtreeTiles()  # spoofed, downloaded street map

        case _:
            print("invalid style")

    # if isinstance(plt.gca(), GeoAxes):
    #     ax = plt.gca()
    # else:
    #     plt.close()
    if (fig is None) and (ax is None):
        # fig, ax = plt.subplots(projection=img.crs)
        if figsize is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection=img.crs)

        # data_crs = ccrs.PlateCarree()
        extent, radius = __compute_multiple_coordinate_extent(lons=lon, lats=lat)

        # auto-calculate scale
        scale = int(120 / np.log(radius))
        scale = (scale < 20) and scale or 19

        ax.set_extent(extent)  # set extents
        ax.add_image(img, int(scale))  # add OSM with zoom specification

    # add site
    data_crs = ccrs.PlateCarree()
    ax.scatter(lon, lat, transform=data_crs, **kwargs)
    if plot_init_pos:
        ax.scatter(lon[0], lat[0], transform=data_crs, color="limegreen", label="Initial Position", s=200, marker="*")

    gl = ax.gridlines(draw_labels=True, crs=data_crs, color="#C5C9C7", lw=0.1, auto_update=True)

    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = ct.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = ct.mpl.gridliner.LATITUDE_FORMATTER

    # ax.legend()

    # if output_dir is not None:
    #     plt.tight_layout()
    #     save_figs(output_dir / "geoplot", types=TYPES)

    return fig, ax


def __compute_multiple_coordinate_extent(lons, lats):
    pairs = [(lon, lat) for lon, lat in zip(lons, lats)]
    bounding_box = BoundingBox(pairs)

    # buffer = 0.05 * bounding_box.height  # add 15% buffer
    buffer = 0.2 * bounding_box.height

    min_y = bounding_box.min_point.y - buffer
    max_y = bounding_box.max_point.y + buffer

    height = max_y - min_y
    geodetic_radius = height / 2
    width = height

    points = np.array(
        [
            [bounding_box.center.x, bounding_box.center.y],
            [bounding_box.center.x, bounding_box.center.y + geodetic_radius],
        ],
    )
    radius_geometry = LineString(points)
    radius = cgeo.Geodesic().geometry_length(geometry=radius_geometry)

    min_x = bounding_box.center.x - width
    max_x = bounding_box.center.x + width

    extent = np.round(
        [
            min_x,
            max_x,
            min_y,
            max_y,
        ],
        decimals=8,
    )

    return extent, radius


def __image_spoof(self, tile):
    """this function reformats web requests from OSM for cartopy
    Heavily based on code by Joshua Hrisko at:
        https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
    """

    url = self._image_url(tile)  # get the url of the street map API
    req = Request(url)  # start request
    req.add_header("User-agent", "Anaconda 3")  # add user agent to request
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read())  # get image
    fh.close()  # close url
    img = Image.open(im_data)  # open image with PIL
    img = img.convert(self.desired_tile_form)  # set image format

    return img, self.tileextent(tile), "lower"  # reformat for cartopy


class Geoplot:
    def __init__(self):
        self._colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        self._font = {"family": "Times New Roman", "size": 18}
        self._title = ""
        self._group_num = 0
        self._data = {}

    # === SET_TITLE ===
    # set the figure title
    #
    # INPUTS:
    #   title   str   desired title
    #
    def set_title(self, title: str):
        self._title = title

    # === SET_FONT ===
    # set the figure font
    #
    # INPUTS:
    #   font    dict    desired font
    #
    def set_font(self, font: dict):
        self._font = font

    # === PLOT ===
    # add plot to current figure window
    #
    # INPUTS:
    #   lat     Nx1   Latitude [deg]
    #   lon     Nx1   Longitude [deg]
    #   alt     Nx1   (optional) Altitude [m]
    #   time    Nx1   (optional) Time [s]
    #   kwargs
    #     color       str     desired plot color as hex code
    #     label       str     label for legend
    #     marker_size double  desired relative marker size
    #
    def plot(
        self, lat: np.ndarray = None, lon: np.ndarray = None, alt: np.ndarray = None, time: np.ndarray = None, **kwargs
    ):
        if not isinstance(lat, np.ndarray):
            lat = np.array([lat], dtype=np.double)
            lon = np.array([lon], dtype=np.double)
            alt = np.array([alt], dtype=np.double)
            time = np.array([time], dtype=np.double)

        # check if lat and lon are input and same size
        if lat is None or lon is None:
            print("Must input both " "lat" " and " "lon" " as degrees!")
            print("Failed to add item!")
            return
        elif lat.size != lon.size:
            print("Size of " "lat" " and " "lon" " inputs must be equal!")
            print("Failed to add item!")
            return

        # check if alt is input
        if alt is None or alt.size != lat.size:
            alt = np.zeros(lat.shape)

        # check if time is input
        if time is None or time.size != lat.size:
            time = np.zeros(lat.shape)

        # check keyword arguments
        if "color" in kwargs:
            self._colors.insert(self._group_num, kwargs["color"])
        if "label" in kwargs:
            label = kwargs["label"]
        else:
            label = f"group{self._group_num}"
        if "marker_size" in kwargs:
            marker_size = kwargs["marker_size"]
        else:
            marker_size = 2.0

        # add data
        self._data[f"lat{self._group_num}"] = lat
        self._data[f"lon{self._group_num}"] = lon
        self._data[f"alt{self._group_num}"] = alt
        self._data[f"time{self._group_num}"] = time
        self._data[f"label{self._group_num}"] = label
        self._data[f"marker_size{self._group_num}"] = marker_size
        self._group_num += 1

    # === SHOW ===
    # Display the plot figure
    #
    def show(self):
        self.__gen_dataframe()
        self.__gen_figure()
        self._fig.show()

    # === __GEN_DATAFRAME ===
    # generates pandas dataframe from the stored plot data dictionary
    def __gen_dataframe(self):
        # combine data
        for i in np.arange(self._group_num):
            if i == 0:
                LLAT = np.array(
                    [
                        self._data[f"lat{i}"],
                        self._data[f"lon{i}"],
                        self._data[f"alt{i}"],
                        self._data[f"time{i}"],
                        np.repeat([self._data[f"marker_size{i}"]], self._data[f"lat{i}"].shape[0]),
                    ]
                ).T
                self._sources = np.repeat([self._data[f"label{i}"]], self._data[f"lat{i}"].shape[0])
            else:
                temp = np.array(
                    [
                        self._data[f"lat{i}"],
                        self._data[f"lon{i}"],
                        self._data[f"alt{i}"],
                        self._data[f"time{i}"],
                        np.repeat([self._data[f"marker_size{i}"]], self._data[f"lat{i}"].shape[0]),
                    ]
                ).T
                LLAT = np.vstack((LLAT, temp))
                self._sources = np.append(
                    self._sources, np.repeat([self._data[f"label{i}"]], self._data[f"lat{i}"].shape[0])
                )

        # generate dataframe
        self._df = pd.DataFrame(LLAT, columns=["lat", "lon", "alt", "time", "size"])

    # === __GEN_FIGURE ===
    # creates the geoplot figure and zooms to optimal settings
    def __gen_figure(self):
        self._fig = px.scatter_mapbox(
            self._df,
            lat="lat",
            lon="lon",
            color_discrete_sequence=self._colors,
            color=self._sources,
            hover_data=["alt", "time"],
            labels={
                "lat": " Latitude [deg] ",
                "lon": "Longitude [deg] ",
                "alt": "   Altitude [m] ",
                "time": "       Time [s] ",
            },
            zoom=0.0,
            size="size",
            size_max=max(2 * self._df["size"]),
        )

        all_pairs = []
        for lon, lat in zip(self._df.lon, self._df.lat):
            all_pairs.append((lon, lat))
        b_box = planar.BoundingBox(all_pairs)
        if b_box.is_empty:
            return 0, (0, 0)
        area = b_box.height * b_box.width
        # zoom = np.interp(area,
        #                 [0, 5**-10, 4**-10, 3**-10, 2**-10, 1**-10, 1**-5],
        #                 [20, 17, 16, 15, 14, 7, 5])
        zoom = np.interp(area, [0, 5**-10, 4**-10, 3**-10, 2**-10, 1**-10, 1**-5], [22, 19, 17.5, 16.5, 15, 12, 10])
        center = b_box.center

        self._fig.update_layout(
            title=self._title,
            font=self._font,
            mapbox_style="white-bg",
            mapbox_layers=[
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "sourceattribution": "United States Geological Survey",
                    "source": [
                        # "https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer/tile/{z}/{y}/{x}"
                        # "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                        "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ],
                }
            ],
            mapbox=dict(center=go.layout.mapbox.Center(lat=center.y, lon=center.x), zoom=zoom),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )
