import numpy as np
import matplotlib.pyplot as plt
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def regression_process(config):

    torch.set_num_threads(1) 
    
    data_path = "../data/DL_torch_data.pt"

    allseed = config
    act_seed = 80     #print to console if the seed matches this. Useful for multiprocessing.
    condition_name = "layers4"
    start_time = str(int(time.time()))
    postfix = "_" + condition_name + "_s@" + "%i" % allseed + "_" + start_time
    csv_out_name  = "_regression_0.0.4" + postfix + ".csv"
    model_save_name = "model_0.0.4" + postfix + ".pt"
    save_net = False

    random.seed(allseed)
    np.random.seed( allseed)
    torch.manual_seed( allseed)
    log_delim = ","


    in_length = 61
    filter_size = 3
    layer1_filters = 12
    layer2_filters = 32 
    final_conv_out_len = int( in_length - (filter_size - 1) - (filter_size - 1) - (filter_size - 1) - (filter_size - 1))
    final_conv_out_len = int( final_conv_out_len)

    episodes = 15000
    batch_size = 48
    test_frequency_in_samples = 48000


    data_in = torch.load( data_path )
    train_inputs_raw = data_in[ "train_inputs"]
    train_labels_raw = data_in[ "train_labels"]
    test_inputs = data_in[ "test_inputs"]
    test_labels = data_in[ "test_labels"]
    train_inputs_subset = data_in[ "train_inputs_subset"]
    train_labels_subset = data_in[ "train_labels_subset"]




    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv1d(1, layer1_filters, filter_size)
            self.conv2 = nn.Conv1d( layer1_filters, layer2_filters, filter_size)
            self.conv3 = nn.Conv1d( layer2_filters, layer2_filters, filter_size)
            self.conv4 = nn.Conv1d( layer2_filters, layer2_filters, filter_size)
            self.fc1 = nn.Linear(   layer2_filters * final_conv_out_len, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = x.view(-1, layer2_filters * final_conv_out_len)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()


    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.001 )



    def test(test_inputs, test_labels, net):

        with torch.no_grad():

            out = net( test_inputs )
            error = torch.mean(torch.abs( torch.transpose(out.data, 0, 1) - test_labels))

        return error 


    csv_header = [ "epoch", "tstmad", "mintstmad", "tnmad", "mintnmad", "totalt", "t"]
    csv_header = ",".join(csv_header) + "\n"
    with open( csv_out_name,'a') as fd:
        fd.write( csv_header )

    t0 =time.time()
    tprev = t0
    samples_count = 0
    min_test_MAD = 1000
    min_train_MAD = 1000
    batch_amount = int( len(train_labels_raw) / batch_size)
    #shuffle_i = np.arange( len(train_labels))
    if True:
        for epoch in range( episodes ):  # loop over the dataset multiple times
            
            #shuffle the training data
            train_combined = list( zip( train_inputs_raw, train_labels_raw))
            random.shuffle(train_combined)
            train_inputs, train_labels = zip(*train_combined)
            
            train_inputs = train_inputs[ 0:(batch_amount * batch_size)]
            train_labels = train_labels[ 0:(batch_amount * batch_size)]
            train_inputs = [ torch.cat( train_inputs[ i:(i + batch_size) ]) for i in (np.arange(batch_amount) * batch_size) ]
            train_labels = [ torch.cat( train_labels[ i:(i + batch_size) ]) for i in (np.arange(batch_amount) * batch_size) ]

            for i in np.arange(len(train_labels)):

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(train_inputs[i])
                loss = criterion(outputs.squeeze(-1), train_labels[i])
                loss.backward()
                optimizer.step()
                
                
                samples_count += batch_size
                if samples_count  >=  test_frequency_in_samples:
                    samples_count -=  test_frequency_in_samples

                    test_MAD = test(test_inputs, test_labels, net)
                    if test_MAD < min_test_MAD:
                        min_test_MAD = test_MAD
                        if (allseed == act_seed) & save_net :
                            torch.save( net.state_dict(), model_save_name )
                    
                    train_MAD = test(train_inputs_subset, train_labels_subset, net)
                    if train_MAD < min_train_MAD:
                        min_train_MAD = train_MAD
                        
                        
                    tnow = time.time()
                    
                    log_out = [ "%i" % epoch, "%.5f" % test_MAD, "%.5f" % min_test_MAD, "%.5f" % train_MAD,
                        "%.5f" % min_train_MAD, "%i" % (tnow - t0), "%.3f" % (tnow - tprev) ]
                        
                    if allseed == act_seed:                        
                        print( "\t".join(log_out) )
                    
                    log_out = ",".join(log_out) + "\n"
                    with open( csv_out_name,'a') as fd:
                        fd.write( log_out )
                    
                    tprev = tnow
                              

        print('Finished Training')


