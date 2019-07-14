import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class BoxMaker():
    '''
    To make expert data set for solving 3D tetris
    '''
    def __init__(self,ldc_ht=45,ldc_wid=45,ldc_len=80):
        self.ldc_ht  = ldc_ht
        self.ldc_wid = ldc_wid
        self.ldc_len = ldc_len

    def get_coords(self,ldc_len,min_len,_range):
        nhs = []
        h = 0 
        while h<=ldc_len:
            nh = np.random.randint(_range[0],_range[1])
            if h+nh<=ldc_len-min_len:
                h+=nh
    #             print(h)
                nhs.append(h)
            if ldc_len-h<=_range[0]+min_len:
                break
        return nhs

    def get_boxes(self):
        len_cuts = np.array(self.get_coords(self.ldc_len,10,[10,50]))
        len_cuts_new = np.copy(len_cuts)
        len_cuts = np.sort(np.append(len_cuts,0))
        len_cuts_new = np.append(len_cuts_new,self.ldc_len)

        wid_cuts = np.array(self.get_coords(self.ldc_wid,10,[10,25]))
        wid_cuts_new = np.copy(wid_cuts)
        wid_cuts = np.sort(np.append(wid_cuts,0))
        wid_cuts_new = np.append(wid_cuts_new,self.ldc_wid)

        ht_cuts = np.array(self.get_coords(self.ldc_ht,10,[10,25]))
        ht_cuts_new = np.copy(ht_cuts)
        ht_cuts = np.sort(np.append(ht_cuts,0))
        ht_cuts_new = np.append(ht_cuts_new,self.ldc_ht)

        lens = len_cuts_new - len_cuts
        wids = wid_cuts_new - wid_cuts
        hts  = ht_cuts_new  - ht_cuts

        floor_building_breadth = 0
        floor_building_length  = 1
        wall_building_length   = 2
        wall_building_breadth  = 3

        building_choice = np.random.randint(0,4,1)[0]

        build_dict = {0:'Floor Building Breadth',
                      1:'Floor Building Length',
                      2:'Wall Building Length',
                      3:'Wall Building Breadth',
                      }
        print('Building Choice is: ',build_dict[building_choice])

        boxes=[]
        if building_choice == floor_building_breadth:
            for i in range(len(ht_cuts)):
                for k in range(len(len_cuts)):
                    for j in range(len(wid_cuts)):
                        boxes.append([lens[k],wids[j],hts[i],wid_cuts[j],len_cuts[k],ht_cuts[i]])

        elif building_choice == floor_building_length:
            for i in range(len(ht_cuts)):
                for j in range(len(wid_cuts)):
                    for k in range(len(len_cuts)):
                        boxes.append([lens[k],wids[j],hts[i],wid_cuts[j],len_cuts[k],ht_cuts[i]])

        elif building_choice == wall_building_length:
            for j in range(len(wid_cuts)):
                for k in range(len(len_cuts)):
                    for i in range(len(ht_cuts)):
                        boxes.append([lens[k],wids[j],hts[i],wid_cuts[j],len_cuts[k],ht_cuts[i]])

        elif building_choice == wall_building_breadth:
            for k in range(len(len_cuts)):
                for j in range(len(wid_cuts)):
                    for i in range(len(ht_cuts)):
                        boxes.append([lens[k],wids[j],hts[i],wid_cuts[j],len_cuts[k],ht_cuts[i]])
        
        return boxes
    
    def get_data_dict(self):
        ldc = np.zeros((self.ldc_wid,self.ldc_len))
        boxes = self.get_boxes()
        data = []
        for m in range(len(boxes)):
            l = boxes[m][0]
            b = boxes[m][1]
            h = boxes[m][2]
            i = boxes[m][3]
            j = boxes[m][4]
            k = boxes[m][5]
            ldc[i:i+b,j:j+l] += h
            ldc_flatten = np.flatten(ldc)
            data.append([[ldc_flatten,],[i,j,k]])

if __name__ == "__main__":
    import shutil
    if os.path.exists('./Box_data'):
        shutil.rmtree('./Box_data')

    if not os.path.exists('./Box_data'):
        os.makedirs('./Box_data')

    data_maker = BoxMaker()
    boxes = data_maker.get_boxes()
    ldc = np.zeros((45,80))
    ldc_ht = 45
    for m in range(len(boxes)):
        l = boxes[m][0]
        b = boxes[m][1]
        h = boxes[m][2]
        i = boxes[m][3]
        j = boxes[m][4]
        k = boxes[m][5]
        ldc[i:i+b,j:j+l] += h
        plt.imshow(ldc,cmap='hot',vmin=0,vmax=ldc_ht)
        plt.savefig('Box_data/state_'+str(m)+'.jpg')