import json
import random
import time


def randdelay(a, b):
    time.sleep(random.uniform(a, b))


def load_json(filename):
    with open(filename, 'r') as f:
        js = json.load(f)
        return js
