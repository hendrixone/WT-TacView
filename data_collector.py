import math
import os.path
import time

import numpy as np
import requests

from data_writer import DataWriter

state_url = 'http://localhost:8111/state'
map_object_url = 'http://localhost:8111/map_obj.json'
indicators_url = 'http://localhost:8111/indicators'


def get_telemetry():
    state_response = requests.get(state_url)
    if state_response.ok:
        state = state_response.json()
        return {key: float(value) for key, value in state.items() if key in
                ['H, m', 'TAS, km/h', 'AoA, deg', 'AoS, deg', 'Vy, m/s']}
    else:
        raise Exception("Bad Conn")


def get_map_objs():
    map_object_response = requests.get(map_object_url)
    if map_object_response.ok:
        map_objs = map_object_response.json()
        return map_objs
    else:
        raise Exception("Bad Conn")


def get_attitude():
    indicator_response = requests.get(indicators_url)
    if indicator_response.ok:
        indicators = indicator_response.json()
        if indicators['valid'] == 'false':
            return 0
        indicators = {key: -float(value) for key, value in indicators.items() if key in
                      ['aviahorizon_roll', 'aviahorizon_pitch', 'compass']}
        indicators['compass'] = -indicators['compass']
        return indicators
    else:
        raise Exception("Bad Conn")


# return the current data of the selected aircraft
def get_data(map_size):
    telemetry = get_telemetry()
    map_object = get_map_objs()
    attitude = get_attitude()
    coord = 0
    for obj in map_object:
        if obj['icon'] == 'Player':
            coord = {key: float(value) * map_size for key, value in obj.items() if key in
                     ['x', 'y', 'dx', 'dy']}
            coord['x'] = 1 - coord['x']
    earth_relative_airspeed = calculate_earth_relative_airspeed(telemetry, attitude, coord)
    return coord, telemetry, attitude, earth_relative_airspeed


"""
Calculate the earth relative TAS vector using:

Aos, AoA, heading, Roll, Pitch, TAS

---------------------------------------------

The calculation will be as followed:
1. Calculate the vector of TAS from fixed body reference using AoA, AoS
2. Rotate the vector using the rotation matrices Rx, Ry, Rz calculated using attitude of the aircraft

-----------------------------------------------------------------------------------------------------

WT doesn't save indicators data from other aircraft, thus the flight data have to be calculated in other way
"""


def calculate_earth_relative_airspeed(telemetry, attitude, coord):
    beta = -np.radians(telemetry['AoS, deg'])
    alpha = -np.radians(telemetry['AoA, deg'])
    if attitude == 0:
        # Alternative method to calculate altitude
        attitude = calculate_attitude(telemetry, coord)
    heading = np.radians(attitude['compass'] - 90)
    roll = np.radians(-attitude['aviahorizon_roll'])
    pitch = np.radians(-attitude['aviahorizon_pitch'])
    TAS = telemetry['TAS, km/h'] * 1000 / 3600

    # Calculate body-relative airspeed vector
    U = TAS * np.cos(alpha) * np.cos(beta)
    V = TAS * np.sin(beta)
    W = TAS * np.sin(alpha) * np.cos(beta)
    body_relative_airspeed = np.array([U, V, W])

    # Define rotation matrices
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_heading = np.array([[np.cos(heading), -np.sin(heading), 0],
                          [np.sin(heading), np.cos(heading), 0],
                          [0, 0, 1]])

    # Combine rotation matrices
    R = R_heading @ R_pitch @ R_roll

    # Calculate Earth-relative airspeed vector
    earth_relative_airspeed = R @ body_relative_airspeed

    return earth_relative_airspeed


"""
Alternative
"""


def calculate_attitude(telemetry, coord):
    pass


# Due to the precision of the map_objs data the x, y coordinate have limits
def calculate_accuracy(coord_1, coord, V_z, earth_relative_airspeed, map_size=1, dt=0.5):
    dx = (coord_1['x'] - coord['x']) * map_size
    dy = -(coord_1['y'] - coord['y']) * map_size
    x_loss = dx - (earth_relative_airspeed[0] * dt)
    y_loss = dy - (earth_relative_airspeed[1] * dt)
    z_loss = V_z - earth_relative_airspeed[2]
    # print result in one line
    print(f'x_loss: {x_loss}, y_loss: {y_loss}, z_loss: {z_loss}')


def start_listen(path, map_size, game_speed, update_interval):
    coord_1 = 0

    # check if file already exited
    if os.path.exists(path):
        raise Exception(path, "Already exits")

    state = "waiting"

    start_time = 0

    data_writer = DataWriter(path)

    counter = 0

    while True:

        loop_start = time.time()
        coord, telemetry, attitude, earth_relative_airspeed = get_data(map_size)
        request_time = time.time() - loop_start

        t = 0

        # try catch
        try:
            # By comparing the present coord and previous coord, we can determine if the game is running
            if state == 'waiting':
                if coord_1 == coord or coord_1 == 0:
                    state = 'waiting'
                    print('State: waiting', end='\r')
                    coord_1 = coord
                    continue
                else:
                    state = 'running'
                    start_time = time.time()
            if state == 'running':
                # calculate the intended time offset
                t = counter * update_interval * game_speed

                # calculate current time_offset
                time_offset = (time.time() - start_time) * game_speed
                if coord_1 == coord:
                    print('paused, terminating')
                    print('time taken:', time_offset)
                    exit()
                coord_1 = coord

                print('State: running ' + '.' * ((counter // 10) % 10)
                      + ' ' * (10 - (counter // 10) % 10), end='\r')
                data_writer.write(time_offset - request_time, telemetry, attitude, coord)
                counter += 1

                # calculate sleep time minimize the error and maintain the update rate
                sleep_time = update_interval - (time_offset - t)
                time.sleep(sleep_time)


        except Exception as e:
            print(e)
            continue