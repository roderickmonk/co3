import time
import pymongo
import numpy as np

#insert credentials
client = pymongo.MongoClient( "" )

db = client["derived-history"]
collection = db['trimmed-orderbooks-V1']

t0 = time.time()
x = list(collection.find({}))
print( int( time.time() - t0))

x = pd.DataFrame.from_dict(x)
x.to_pickle( "./raw_data_pd.pkl" )
