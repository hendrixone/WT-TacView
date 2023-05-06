# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from data_collector import *
from data_writer import *
from data_collector import *


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    update_interval = 0.1
    game_speed = 1
    map_size = 128000

    start_listen('data.csv', map_size, game_speed, update_interval)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# 89.41.936