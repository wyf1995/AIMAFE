# -*- coding: utf-8 -*-

import os

path = './data/cc200_tichu_2500_1250_625/'
#path = './data/aaltichu_2500_1250_625/'
#path = './data/dosenbach160_tichu_2500_1250_625/'
for files in os.listdir(path):
    #filesgai = files[2:11]+files[13:1000]#aal
	filesgai = files[2:13]+files[15:1000]#cc200
    #filesgai = files[2:20] + files[22:1000]  # dos



    print filesgai

    os.rename('./data/cc200_tichu_2500_1250_625/'+files, './data/cc200_tichu_2500_1250_625/'+filesgai)
    #os.rename('./data/aal_tichu_2500_1250_625/'+files, './data/aal_tichu_2500_1250_625/'+filesgai)
   # os.rename('./data/dosenbach160_tichu_2500_1250_625/'+files, './data/dosenbach160_tichu_2500_1250_625/'+filesgai)