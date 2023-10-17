import pandas as pd


def custom_metric(file, avg_rooms, avg_occupancy):
    """Return the rooms to occupany ratio.
    file represents the CSV file
    avg_rooms represents the average number of rooms
    avg_occupancy represents the average number of people per household
    """

    df = pd.read_csv(file)
    last_row_number = df.shape[0] - 1

    total_rooms = 0
    total_occupants = 0

    for i in range(last_row_number):
        rooms = df.loc[i, avg_rooms]
        occupancy = df.loc[i, avg_occupancy]

        total_rooms += rooms
        total_occupants += occupancy

    return total_rooms / total_occupants
