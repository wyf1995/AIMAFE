
# -*- coding: utf-8 -*-

"""

Autoencoders evaluation.

Usage:
  nn_evaluate.py [--whole] [--male] [--threshold] [--leave-site-out] [<derivative> ...]
  nn_evaluate.py (-h | --help)

Options:
  -h --help           Show this screen
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  derivative          Derivatives to process

"""


import numpy as np
import pandas as pd
import math
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from docopt import docopt
from nn import nn
from utils import (load_phenotypes, format_config, hdf5_handler,
                   reset, to_softmax, load_ae_encoder, load_fold)
from sklearn.metrics import confusion_matrix

import xlwt


def nn_results(hdf5, experiment, code_size_1, code_size_2,code_size_3):

    exp_storage = hdf5["experiments"]['dosenbach160_whole']

    experiment="dosenbach160_whole"

    print exp_storage

    n_classes = 2

    results = []

    list=['']

    list2 =[ ]


    for fold in exp_storage:

        experiment_cv = format_config("{experiment}_{fold}", {
            "experiment": experiment,
            "fold": fold,
        })

        print "experiment_cv"

        print fold

        X_train, y_train, \
        X_valid, y_valid, \
        X_test, y_test,test_pid = load_fold(hdf5["patients"], exp_storage, fold)

        list.append(test_pid)

        print "X_train"

        print X_train.shape

        y_test = np.array([to_softmax(n_classes, y) for y in y_test])

        ae1_model_path = format_config("./data/dos_tichu_2500_1250_625/{experiment}_autoencoder-1.ckpt", {
            "experiment": experiment_cv,
        })
        ae2_model_path = format_config("./data/dos_tichu_2500_1250_625/{experiment}_autoencoder-2.ckpt", {
            "experiment": experiment_cv,
        })

        ae3_model_path = format_config("./data/dos_tichu_2500_1250_625/{experiment}_autoencoder-3.ckpt", {
            "experiment": experiment_cv,
        })

        nn_model_path = format_config("./data/dos_tichu_2500_1250_625/{experiment}_mlp.ckpt", {
            "experiment": experiment_cv,
        })

		
        try:

            model = nn(X_test.shape[1], n_classes, [
                {"size": 2500, "actv": tf.nn.tanh},
                {"size": 1250, "actv": tf.nn.tanh},
                {"size": 625, "actv": tf.nn.tanh},
            ])

            init = tf.global_variables_initializer()
            with tf.Session() as sess:

                sess.run(init)

                saver = tf.train.Saver(model["params"])

                print "savernn_model_path"

                print nn_model_path

                saver.restore(sess, nn_model_path)

                output = sess.run(
                    model["output"],
                    feed_dict={
                        model["input"]: X_test,
                        model["dropouts"][0]: 1.0,
                        model["dropouts"][1]: 1.0,
                        model["dropouts"][2]: 1.0,
                    }
                )

                np.set_printoptions(suppress=True)

                y_score = output[:,1]

                print "y_score"

                print y_score


                y_pred = np.argmax(output, axis=1)
						
                print "y_pred"
                print y_pred

                print "output"

                hang=output.shape[0]

                lie=output.shape[1]

                print hang

                print lie


                for tt in range(hang):
                    for xx in range(lie):

                        output[tt][xx]=round(output[tt][xx],4)

                        output[tt][xx]=str(output[tt][xx])

                aa=output[:,0]

                print type(aa)

                list2.append(output)



                list.append(y_pred)

                print "-------------------------------------"

                y_true = np.argmax(y_test, axis=1)

                list.append(y_true)

                print "y_true"
                print y_true

                auc_score = roc_auc_score(y_true, y_score)
                print auc_score

                [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
                accuracy = (TP+TN)/(TP+TN+FP+FN)

                print(TP)
                print(TN)
                print(FP)
                print(FN)
                specificity = TN/(FP+TN)
                precision = TP/(TP+FP)
                sensivity = recall = TP/(TP+FN)
                fscore = 2*TP/(2*TP+FP+FN)

                results.append([accuracy, precision, recall, fscore, sensivity, specificity,auc_score])
        finally:
            reset()



    workbook = xlwt.Workbook(encoding='utf-8')

    booksheet = workbook.add_sheet('Sheet 1',cell_overwrite_ok=True)

    wb = xlwt.Workbook(encoding='utf-8')

    worksheet = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)

    DATA = list

    print list2

    for i,row in enumerate(DATA):
        for j,col in enumerate(row):
            booksheet.write(j,i,col)
    workbook.save('./data/dos_tichu_2500_1250_625_xlst.xls')


    return [experiment] + np.mean(results, axis=0).tolist()

if __name__ == "__main__":

    reset()

    arguments = docopt(__doc__)

    pd.set_option("display.expand_frame_repr", False)

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed0.csv"
    pheno = load_phenotypes(pheno_path)

    hdf5 = hdf5_handler("./data/abide_dosenbach160_tichu.hdf5", "a")

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative
                   in arguments["<derivative>"]
                   if derivative in valid_derivatives]

    experiments = []

    for derivative in derivatives:

        config = {"derivative": derivative}

        if arguments["--whole"]:
            experiments += [format_config("{derivative}_whole", config)],

        if arguments["--male"]:
            experiments += [format_config("{derivative}_male", config)]

        if arguments["--threshold"]:
            experiments += [format_config("{derivative}_threshold", config)]

        if arguments["--leave-site-out"]:
            for site in pheno["SITE_ID"].unique():
                site_config = {"site": site}
                experiments += [
                    format_config("{derivative}_leavesiteout-{site}",
                                  config, site_config)
                ]

    # First autoencoder bottleneck
    code_size_1 = 2500

    # Second autoencoder bottleneck
    code_size_2 = 1250

    code_size_3 = 625

    results = []

    experiments = sorted(experiments)
    for experiment in experiments:
        results.append(nn_results(hdf5, experiment, code_size_1, code_size_2,code_size_3))

    cols = ["Exp", "Accuracy", "Precision", "Recall", "F-score",
            "Sensivity", "Specificity","AUC"]
    df = pd.DataFrame(results, columns=cols)

    print df[cols] \
        .sort_values(["Exp"]) \
        .reset_index()
