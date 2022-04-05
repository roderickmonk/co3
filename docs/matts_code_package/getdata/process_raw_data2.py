import pandas as pd
import numpy as np
import torch
import time

import gridob
import reward_stuff
#importlib.reload()



#########################################
in_filename  =  "../data/raw_data_pd.pkl"
out_filename =  "DL_torch_data.pt"
#########################################
allseed = 7
start, stop = -4, 2
grid_resolution = 0.1
test_set_size     = 20000
train_subset_size = 20000
#########################################
# Settings for labels/reward
reward_mid_price = 0.03    #This is only used for the reward function and has negligible effect if relative tick size is not small
rconf = reward_stuff.reward_config #loads PDF, also sets ql and tick size to defaults
#########################################
np.random.seed( allseed )
torch.manual_seed( allseed) #not actually used AFAIK
torch.set_num_threads(1) 



#############################
# Load data, convert to NN input structure, shuffle data

t0 =time.time()
obdata = pd.read_pickle( in_filename)

#mid price
mp = np.array([ (obdata.buy[i][0][0] + obdata.sell[i][0][0]) / 2 for i in np.arange(len(obdata))])

#this is done to remove any OB with a total size >= 100. There are only about 300 out of ~2 million of them.
#this is done because it causes problem given the current range of (-4, 2) (and note that 10^2 = 100) for the obstate bins.
#I don't want to change that range, because I want to keep the properties of the state vector the same as before.
#note the use of the number 80 in np.greater. This was done for superstitious reasons and can easily be 
#changed back to 100 for any other dataset.
i =  np.greater( 80, [ np.sum( np.array(x)[: ,0] * np.array(x)[: ,1]) for x in obdata.buy])
obdata = obdata.iloc[i]
obdata = obdata.reset_index(drop=True)
mp = mp[i]

obstates = [ gridob.get_grid_ob( -4, 2, grid_resolution, mid_rate = mp[i], OB = np.array( obdata.buy[i]) ) for i in np.arange(len(mp))]
obstates = np.unique( np.array( obstates), axis=0)

random_i = np.arange( len(obstates))
np.random.shuffle( random_i)
obstates = obstates[ random_i ]



#############################
# Get the data labels

bin_sizes = 10**np.arange( start, stop + grid_resolution, grid_resolution )
bin_sizes = (bin_sizes[:-1] + bin_sizes[1:]) / 2

states =    [ reward_stuff.get_standard_state( bin_sizes, i) for i in obstates ]
max_prof =  [ reward_stuff.get_reward( rconf, action=0.001, state=i, mid_price=reward_mid_price)[2] for i in states]


#--------
#there is a problem with negative profits that should not be occurring, but it is only 8 out of 2 million
max_prof = np.array(max_prof)
#print( np.equal(max_prof, 0).sum())
#print( np.less (max_prof, 0).sum())
i = np.greater( max_prof, 0)
max_prof = max_prof[i]
obstates = obstates[i]
#--------

max_prof =  np.log10(max_prof)



#############################
# Make the test set, convert to tensors, save dataset file

test_inputs  = obstates[ :test_set_size  ]
test_labels  = max_prof[ :test_set_size  ]
train_inputs = obstates[  test_set_size: ]
train_labels = max_prof[  test_set_size: ]

test_inputs  = [ torch.from_numpy( i ).unsqueeze(0).unsqueeze(0).float() for i in test_inputs ]
test_labels  = [ torch.from_numpy( np.array(i) ).unsqueeze(0).float() for i in test_labels ]
train_inputs = [ torch.from_numpy( i ).unsqueeze(0).unsqueeze(0).float() for i in train_inputs ]
train_labels = [ torch.from_numpy( np.array(i) ).unsqueeze(0).float() for i in train_labels ]
train_inputs_subset = train_inputs[ :train_subset_size ] #to estimate train set loss
train_labels_subset = train_labels[ :train_subset_size ]


test_inputs = torch.cat( test_inputs)
test_labels = torch.cat( test_labels)
train_inputs_subset = torch.cat( train_inputs_subset)
train_labels_subset = torch.cat( train_labels_subset)

out = { "train_inputs":train_inputs, "train_labels":train_labels, "test_inputs":test_inputs, "test_labels":test_labels, "train_inputs_subset":train_inputs_subset, "train_labels_subset":train_labels_subset }
torch.save(  out, out_filename )



#############################
print( "Time Taken")
print( int(time.time() - t0))