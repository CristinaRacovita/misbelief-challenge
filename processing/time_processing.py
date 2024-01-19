import datetime as DT
import numpy as np

def get_datetime(timestamp):
    today = DT.date.today()
    timestamp = timestamp.split(' ')[-1]
    if 'd' in timestamp:
        timestamp = process_timestamp(timestamp, 'd')
        return today - DT.timedelta(days=float(timestamp))
    if 'mo' in timestamp:
        timestamp = process_timestamp(timestamp, 'mo')
        return today - DT.timedelta(days=float(timestamp)*30)
    if 'y' in timestamp:
        timestamp = process_timestamp(timestamp, 'y')
        return today - DT.timedelta(days=float(timestamp)*365)
    if 'h' in timestamp:
        timestamp = process_timestamp(timestamp, 'h')
        return today - DT.timedelta(hours=float(timestamp))
    if 'm' in timestamp:
        timestamp = process_timestamp(timestamp, 'm')
        return today - DT.timedelta(minutes=float(timestamp))
    if 's' in timestamp:
        timestamp = process_timestamp(timestamp, 's')
        return today - DT.timedelta(seconds=float(timestamp))

def process_timestamp(timestamp, character):
    return timestamp.replace(character,'').strip()

# print(get_datetime('4mo'))