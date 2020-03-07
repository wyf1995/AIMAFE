Copyright (C) 2019 Yufei Wang(willem@csu.edu.cn)

Package Title:Autism spectrum disorder identification with multi-atlas deep feature representation and ensemble learning

Description: This package aims to achieve the automatically diagnose subjects with ASD based on multi-atlas deep feature representation and ensemble learning derived from their Resting-state functional magnetic resonance imaging (Rs-fMRI) brain scans.

The files in this project:
nn.py  The pretraining code
nn_evaluate.py The classification code for each feature set based on brain atlas


How to run this project:
This project must run in python==2.7, The following steps should be taken to run this project:
1. Using stacked denoising autoencoder (SDA) to pretrain for each feature set based on single brain atlas
  python nn.py --whole --cc200
  python nn.py --whole --aal
  python nn.py --whole --dosenbach160
  
2. Using multilayer perceptron (MLP) to classify the ASD and TC.
 python nn_evaluate --whole cc200
 python nn_evaluate --whole aal
 python nn_evaluate --whole dosenbach160
 
 3. Using Voting strategy based on three results of different feature sets
 
 

