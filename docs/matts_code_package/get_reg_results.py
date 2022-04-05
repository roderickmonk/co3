import os
import numpy as np
import pandas as pd

files = np.array( os.listdir())
files = files[ [ ("_regression_" in i) for i in files ]]

files_split = [ i.split("_s@") for i in files ]
experiments = np.unique( [ i[0] for i in files_split ])

summaries = []
for i in experiments:
    trials = files[[ ( (i + "_s@") in j) for j in files ]]

    last_rows = []
    for j in trials:
        trial = pd.read_csv( j, delimiter=",")
        last_rows.append( trial.iloc[[-1]] )

    data = pd.concat( last_rows )

    summary = [
        i,
	data.shape[0],
        data.tstmad.mean(),
        data.mintstmad.mean(),
        data.mintstmad.std(),
        data.tnmad.mean(),
        data.mintnmad.mean(),
        np.mean( data.mintstmad / data.mintnmad )]

    summaries.append( summary )

summaries = pd.DataFrame.from_records( summaries,
    columns = ["Experiment", "n", "tstmad", "mintstmad", "mintstmadSD", "tnmad", "mintnmad", "overfit"])

print("")
print(summaries)
print("")
