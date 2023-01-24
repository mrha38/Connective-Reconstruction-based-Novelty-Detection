
import os
from model import *
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.utils import shuffle
from tools import *
import imgaug.augmenters as iaa
import sklearn.metrics

latentSpaceSize = 256


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
x_train = tf.image.resize(x_train, [32,32])
x_test = np.expand_dims(x_test, axis=-1)
x_test = tf.image.resize(x_test, [32,32])
x_train = x_train / 255.
x_test = x_test / 255.
epochs = 100
batch_size = 128
x_train = x_train.numpy()
x_test = x_test.numpy()
landa = 0.5

for targetClass in range(0, 10):
    cPath = str(targetClass) + '/models/'

    if not os.path.exists(cPath):
        os.makedirs(cPath)

    connectiveModel = ConnectiveModel()

    selectedIndexKLBL = []
    selectedIndexULBL = []
    for i in range(len(y_train)):
        if y_train[i] == targetClass:
            selectedIndexKLBL.append(i)
        else:
            selectedIndexULBL.append(i)

    selectedIndexKLBL = np.array(selectedIndexKLBL)

    selectedIndexKLBLTest = []
    selectedIndexULBLTest = []
    for i in range(len(y_test)):
        if y_test[i] == targetClass:
            selectedIndexKLBLTest.append(i)
        else:
            selectedIndexULBLTest.append(i)

    selectedIndexKLBLTest = np.array(selectedIndexKLBLTest)
    selectedIndexULBLTest = np.array(selectedIndexULBLTest)

    selectedIndexKLBLEval = np.random.choice(len(selectedIndexKLBLTest), int((10 * len(selectedIndexKLBLTest)) / 100))
    selectedIndexKLBLEval = selectedIndexKLBLTest[selectedIndexKLBLEval]

    selectedIndexULBLEval = np.random.choice(len(selectedIndexULBLTest), int((10 * len(selectedIndexULBLTest)) / 100))
    selectedIndexULBLEval = selectedIndexULBLTest[selectedIndexULBLEval]


    mBlur = iaa.MotionBlur(k=10)
    gn = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
    CoarseDropout = iaa.CoarseDropout(0.5, size_percent=0.4)
    tempAUG0 = CoarseDropout(images = x_train[selectedIndexKLBL][:,:,:,0])
    tempAUG0 = np.expand_dims(tempAUG0, axis=-1)
    tempAUG1 = mBlur(images = x_train[selectedIndexKLBL][:,:,:,0])
    tempAUG1 = np.expand_dims(tempAUG1, axis=-1)
    CoarseDropout = iaa.CoarseDropout(0.7, size_percent=0.6)
    tempAUG2 = CoarseDropout(images = x_train[selectedIndexKLBL][:,:,:,0])
    tempAUG2 = np.expand_dims(tempAUG2, axis=-1)
    tempAUG3 = gn(images = x_train[selectedIndexKLBL][:,:,:,0])
    tempAUG3 = np.expand_dims(tempAUG3, axis=-1)

    augImages = np.concatenate(
        (
            tempAUG0, tempAUG1, tempAUG2, tempAUG3
        )
    )

    del tempAUG0
    del tempAUG1
    del tempAUG2
    del tempAUG3

    X = np.concatenate(
        (
            x_train[selectedIndexKLBL],
            augImages
        )
    )
    Y = np.concatenate(
        (
            x_train[selectedIndexKLBL],
            x_train[selectedIndexKLBL],
            x_train[selectedIndexKLBL],
            x_train[selectedIndexKLBL],
            x_train[selectedIndexKLBL]
        )
    )

    X, Y = shuffle(X, Y)


    maxScore = 0
    maxScoreVal = 0
    minLoss = 100
    
    for epoch in range(epochs):
        its = len(X) // batch_size
        encoderLoss = []
        decoderLoss = []
        aeLoss = []
        imageForPlot = []
        sampleCounter = 0

        for it in range(its):

            lossAE = connectiveModel.AE.train_on_batch(
                X[it * batch_size:(it + 1) * batch_size],
                Y[it * batch_size:(it + 1) * batch_size]
            )

            decoderLoss.append(lossAE)
        

        auc2 = calcAUC(connectiveModel.AE, x_test[selectedIndexKLBLEval], x_test[selectedIndexULBLEval])
        #auc2 = calcAUC(connectiveModel.AE, x_test[selectedIndexKLBLTest], x_test[selectedIndexULBLTest])
        print(
            epoch,
            'AE loss:', sum(decoderLoss) / len(decoderLoss),
            #'AUC:', auc2,
        )

        aeLoss.append([
            epoch,
            sum(decoderLoss) / len(decoderLoss),
            #auc2,
        ])

        if auc2 > maxScore:
            maxScore = auc2
            connectiveModel.AE.save(cPath + 'AE.h5')

        mBlur = iaa.MotionBlur(k=10)
        gn = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
        CoarseDropout = iaa.CoarseDropout(0.5, size_percent=0.4)
        tempAUG0 = CoarseDropout(images = x_train[selectedIndexKLBL][:,:,:,0])
        tempAUG0 = np.expand_dims(tempAUG0, axis=-1)
        tempAUG1 = mBlur(images = x_train[selectedIndexKLBL][:,:,:,0])
        tempAUG1 = np.expand_dims(tempAUG1, axis=-1)
        CoarseDropout = iaa.CoarseDropout(0.7, size_percent=0.6)
        tempAUG2 = CoarseDropout(images = x_train[selectedIndexKLBL][:,:,:,0])
        tempAUG2 = np.expand_dims(tempAUG2, axis=-1)
        tempAUG3 = gn(images = x_train[selectedIndexKLBL][:,:,:,0])
        tempAUG3 = np.expand_dims(tempAUG3, axis=-1)

        augImages = np.concatenate(
            (
                tempAUG0, tempAUG1, tempAUG2, tempAUG3
            )
        )

        del tempAUG0
        del tempAUG1
        del tempAUG2
        del tempAUG3

        X = np.concatenate(
            (
                x_train[selectedIndexKLBL],
                augImages
            )
        )
        Y = np.concatenate(
            (
                x_train[selectedIndexKLBL],
                x_train[selectedIndexKLBL],
                x_train[selectedIndexKLBL],
                x_train[selectedIndexKLBL],
                x_train[selectedIndexKLBL]
            )
        )

        X, Y = shuffle(X, Y)
    

    connectiveModel.AE = load_model(cPath + 'AE.h5', compile = False)
    X = x_train[selectedIndexKLBL]
    inliersRes = connectiveModel.AE.predict(X)
    randMotion = 5
    mBlur = iaa.MotionBlur(k=randMotion)
    AverageBlur = iaa.AverageBlur(k=(5, 20))
    GaussianBlur = iaa.GaussianBlur(sigma=(1.0, 3.0))

    x_inlier = []
    for makeSamples in range(len(inliersRes)):
        x_inlier.append(
            X[makeSamples] * landa + inliersRes[makeSamples] * (1 - landa)
        )

    tempAUG1 = mBlur(images = x_train[selectedIndexKLBL][:,:,:,0])
    tempAUG1 = np.expand_dims(tempAUG1, axis=-1)
    augImages = tempAUG1

    x_outlier = []
    for makeSamples in range(len(tempAUG1)):
        x_outlier.append(
            tempAUG1[makeSamples] * landa + X[makeSamples] * (1 - landa)
        )

    del augImages
    mX = np.concatenate(
        (x_inlier, x_outlier)
    )
    Y = np.zeros((len(X)))
    y_fake = np.ones((len(x_outlier)))
    mY = np.concatenate(
        (Y, y_fake)
    )
    mY = tf.keras.utils.to_categorical(mY, 2)
    mX, mY = shuffle(mX, mY)
    maxScore = 0
    
    for epoch in range(epochs):
        ixs = len(mX) // batch_size
        lossL = []
        for ix in range(ixs):

            loss = connectiveModel.classifier.train_on_batch(
                mX[ix * batch_size:(ix + 1) * batch_size],
                mY[ix * batch_size:(ix + 1) * batch_size]
            )

            lossL.append(loss)


        aucList = []

        #kT = connectiveModel.connective.predict(x_test[selectedIndexKLBLEval])
        kT = connectiveModel.connective.predict(x_test[selectedIndexKLBLTest])
        kT = kT[:,0]

        uT = connectiveModel.connective.predict(x_test[selectedIndexULBLEval])
        #uT = connectiveModel.connective.predict(x_test[selectedIndexULBLTest])
        uT = uT[:,0]

        scores = np.concatenate(
            (
                kT, uT
            )
        )

        labels = np.concatenate(
            (
                np.ones((len(selectedIndexKLBLTest))),
                np.zeros((len(selectedIndexULBLTest)))
            )
        )
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores, pos_label = True)
        auc = sklearn.metrics.auc(fpr, tpr)
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, scores)
        f1_scores = 2*recall*precision/(recall+precision)
        f1_scores = np.max(f1_scores)

        print(
            epoch,
            sum(lossL) / len(lossL),
            #'AUC: ',
            #auc, 
            'F1:', f1_scores
        )

        if maxScore > f1_scores:
            maxScore = f1_scores
            connectiveModel.connective.save(cPath + 'noveltyModel.h5')
            connectiveModel.classifier.save(cPath + 'classifier.h5')



