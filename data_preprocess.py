import pandas as pd
import numpy as  np
import json, os, shutil
from collections import ChainMap
from sklearn.model_selection import train_test_split
from typing import *



class preprocess:

    def __init__(self, data_dir: str, box_path: str, img_path: str, entities: List[str], test_split: float = 0.2) -> None:
        
        self.data_dir = data_dir
        self.box_path = box_path
        self.img_path = img_path
        self.test_split = test_split
        self.ent_path = self.data_dir+'entities/'
        self.entities = entities
        self.data_list = []


    def __call__(self) -> None:
        

        def move_data(file: Any, train: bool ) -> None:

            if train: 
                root_dir = 'train_data/'
                nm = 'train'
            else: 
                root_dir = 'test_data/'
                nm = 'test'

            bx_path = root_dir+'boxes_and_transcripts/'
            im_path = root_dir+'images/'
            en_path = root_dir+'entities/'

            for index, row in file.iterrows():
                shutil.copy(self.box_path+str(row[2])+".tsv", bx_path)
                shutil.copy(self.img_path+str(row[2])+".jpg", im_path)
                shutil.copy(self.ent_path +str(row[2])+".txt", en_path)
                
            file.drop(['Unnamed: 0'], axis = 1,inplace = True)
            file.reset_index(inplace= True)
            file.drop(['index'], axis = 1, inplace = True)
            file.to_csv(root_dir+nm+'_samples_list.csv', header = False)


        self.data_list = self.__create_entities
        train, test= train_test_split(self.data_list, test_size = self.test_split, random_state = 42) 
        move_data(train, True)
        move_data(test, False)
        



    def __create_entities(self) -> List[List[str]]:

        def class_extract(cls):
                idx = np.where(tag == cls)[0]
                return {cls:' '.join(text[idx])}

        data_list = []

        for file in os.listdir(self.box_path):
   
            data_list.append(['aadhar',file.replace('.tsv','')])

            df = pd.read_csv(self.box_path+file, sep = '\t', header = None, names = ['index','x1','y1','x2','y2','x3','y3','x4','y4','text','Ner_Tag'])
            classes = df['Ner_Tag'].unique()
            classes = np.delete(classes, np.where(classes=='others')[0])

            text, tag = df['text'].to_numpy() , df['Ner_Tag'].to_numpy()
            
            result = list(map(class_extract,classes))

            outputjson = dict(ChainMap(*result))

            text_file = self.ent_path + file.split(".")[0]+".txt"
            
            with open(text_file,"w") as txt:
                txt.write(json.dumps(outputjson))
        
        self.data_list = pd.DataFrame(self.data_list)
        self.data_list.to_csv(self.data_dir+'data_list.csv')
        
        return self.data_list