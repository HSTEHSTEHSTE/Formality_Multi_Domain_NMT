import csv
import os

def prepare_data(formal_data_file, informal_data_file, target_file):
    formal_file = open(formal_data_file, "r", newline='')
    informal_file = open(informal_data_file, "r", newline='')
    target = open(target_file, "w", newline='')
    target_writer = csv.writer(target, lineterminator='\n')
    for line in formal_file:
        target_writer.writerow([line[:-1]] + [1])
    for line in informal_file:
        target_writer.writerow([line[:-1]] + [0])

prepare_data(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/formal.txt"), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/informal.txt"), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/data.csv"))
