import numpy as np
import argparse
import glob

M = 7+1+64+1+64+3+64+2
DEBUG = False

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Data Maker",
        epilog = "Use this program to randomly split the training data for p4 into random sub-samples.",
        add_help = "How to use",
        prog = "python trainer.py -i <path_to_list> " )
    
    parser.add_argument("-i", "--input",
        help = "The path to find the list of directories with the results in them")

    args = vars(parser.parse_args())
    inp  = args['input']
    inpf = open(inp)
    
    directories = [x.strip() for x in inpf.readlines()]
    
    for directory in directories:
        print("Stitching data for {}...".format(directory))
        iterrator = glob.iglob("Results/{}/*".format(directory))
        maxX = 0
        maxY = 0
        results = {}
        i=0
        for fil in iterrator:
            realfil = fil[fil.find("\'")+1:fil.find("_")]
            result = {}
            m = fil.find('_')
            n = fil.rfind("\'.npy")
            coors = [int(x) for x in fil[m+1:n].split('_')]
            if DEBUG :
                print("realfil: {}".format(realfil))
                print("n: {}".format(n))
                print("m: {}".format(n))
                print("fil[m:n]: {}".format(fil[m:n]))
                print("coors: {}".format(coors))
            xx  = coors[1]
            x = coors[0]
            yy  = coors[3]
            y = coors[2]
            if xx > maxX : maxX = xx
            if yy > maxY : maxY = yy
            result['x']=x
            result['y']=y
            result['xx']=xx
            result['yy']=yy
            result['data'] = np.load(fil)
            results[i]=result
            i+=1
        if DEBUG :print("maxX: {},maxY: {}".format(maxX,maxY))
        full = np.zeros((maxX,maxY))
        hits = np.zeros((maxX,maxY))
        for i in results.keys():
            res = results[i]
            if DEBUG :
                print("res['x']: {}".format(res['x']))
                print("res['y']: {}".format(res['y']))
                print("res['xx']: {}".format(res['xx']))
                print("res['yy']: {}".format(res['yy']))                
            for r in range(128):
                for s in range(128):
                    if DEBUG :
                        print("R: {}, S: {}".format(r,s))
                        print("hits.shape : {}, full.shape: {}, res['data'].shape : {}".format(hits.shape , full.shape, res['data'].shape))
                    hits[r+res['x'],s+res['y']] +=1
                    full[r+res['x'],s+res['y']] += res['data'][r,s]
        prediction = full / hits
        np.save("Predictions/{}.npy".format(realfil),prediction)
