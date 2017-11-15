import pickle
import caffe 
import numpy as np
import os

'''
This scipt iterates the caffe files, initializes them with the weights after
the final training iteration and saves the weights and biases to pickle files. 

'''
rootdir = '/mnt/raid/ni/dnn/invariance/Claudia/Network_defs'


caffe.set_mode_cpu()

DEPLOY_CAFFE_FILE = '/MLP_net_deploy.prototxt' # to use the constant caffe 
# networks they have to be changed to 'deploy' format 
# (see also 'deploy_caffe_file.txt' in folder 'notes')
CAFFE_MODEL = '/_iter_390000.caffemodel'


Subdirs = []
Dirs = []
Files = []

for subdir, dirs, files  in os.walk(rootdir): 
    Subdirs.append(subdir)
    Dirs.append(dirs)
    Files.append(files)
    
#-------------------------------------------------------------------------------------------#
# iterate all files in rootdir and ask for every file if it should be convertet. If answer
# is yes, extract the networkname form the filename, load the respective deploy-file with the 
# final caffemodel. Create lists W and B containing the weights and biases for each layer and 
# also save the parameter dimensions to a file to check if the correct model was selected.
#-------------------------------------------------------------------------------------------#
for subdir in Subdirs: 
    print(subdir)
    print('convert? (y/n)')
    x =raw_input()
    if x == 'y': 
        path = subdir + '/'
        location1 = -1 
        while True: # extract network name from filename
            location2 = location1 
            location1 = subdir.find("/", location1 + 1) 
            if location1 == -1: break      # Break if not found.
        
        NETWORK_NAME = subdir[location2+1:]                     
        print(NETWORK_NAME)
        
        # load the deployed caffe file and initialize with correct caffemodel
        net = caffe.Net(path + DEPLOY_CAFFE_FILE, path + CAFFE_MODEL , caffe.TEST)
        
        # create a file to save the dimensions of weights and biases. 
        f = open(path +'sanity_check.txt', 'w')
        f.write(NETWORK_NAME + '\n')
        f.write('blobs:' + format(net.blobs.keys()) + '\n')
        f.write('params:' + format(net.params.keys()) + '\n')
       
        # read in weights and biases
        W = []
        B = []
        for i, key in enumerate(net.params.keys()):
                W.append(net.params[key][0].data)
                B.append(net.params[key][1].data)
                f.write(key + '\t' + str(net.params[key][0].data.shape) + '\t' + str(net.params[key][1].data.shape) + '\n')        
        f.close()


	    # write all params to dictionary and save as pickle
        params = {}
        params['weights'] = W
        params['biases'] = B
        pickle.dump(params, open(rootdir+ '/weights/' + NETWORK_NAME + '_params.p', 'wb' ))
        
    else: 
        print('miao')


