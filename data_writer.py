# expect dicationary as data
import csv
import pyproj
from functools import partial


def __projection__(coord):
    virtual_plane_proj = pyproj.Proj("+proj=utm +zone=10 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    sf_latlon_proj = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs", preserve_units=True)
    transformer = pyproj.Transformer.from_proj(virtual_plane_proj, sf_latlon_proj)
    return transformer.transform(*(coord['x'], coord['y']))


def __parse__(time, telemetry, attitude, coord):
    lon, lat = __projection__(coord)
    data = {
        'Time':         round(time, 8),
        'Longitude':    round(lon, 8),
        'Latitude':     round(lat, 8),
        'Altitude':     round(telemetry['H, m'], 8),
        'Roll':         round(attitude['aviahorizon_roll'], 8),
        'Pitch':        round(attitude['aviahorizon_pitch'], 8),
        'Yaw':          round(attitude['compass'] - 180, 8),
        'AOA':          round(telemetry['AoA, deg'], 8),
        'AOS':          round(telemetry['AoS, deg'], 8),
        'TAS':          round(telemetry['TAS, km/h'], 8)
    }
    return data


class DataWriter:
    def __init__(self, path):
        file = open(path, 'w')
        self.writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        self.web_mercator = pyproj.Proj('EPSG:3857')
        self.wgs84 = pyproj.Proj('EPSG:4326')
        self.data = 0

    # Take the coordinates of x, y in meters and project them to earth surface to get lat and lon using pyproj

    def write(self, time, telemetry, attitude, coord):
        # if self.data does not exist, create it
        if self.data == 0:
            # parse the data to proper format
            self.data = __parse__(time, telemetry, attitude, coord)
            self.writer.writerow(self.data.keys())
        else:
            # if self.data exists, check if it has the same keys as data
            if self.data.keys() != self.data.keys():
                raise ValueError('Data does not have the same keys as the previous data')
            else:
                # write the values to the file
                self.data = __parse__(time, telemetry, attitude, coord)
                self.writer.writerow(self.data.values())
