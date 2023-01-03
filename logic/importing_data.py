import pandas as pd
import json


path = "amazonReviews.json"
with open(path, encoding='utf-8', errors='ignore') as j:
    contents = json.loads(j.read())
