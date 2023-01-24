import tensorflow as tf
import numpy as np
import sklearn.metrics

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))

def mixData(para, landa = 0.5):
    x =para[0] 
    y =para[1]
    return x * landa + y * (1 - landa)

def calcAUC(model, inliers, outliers):
    
    inliersRes = model.predict(inliers)
    outliersRes = model.predict(outliers)

    inliersRes = np.reshape(inliersRes, (len(inliersRes), -1))
    outliersRes = np.reshape(outliersRes, (len(outliersRes), -1))
    inliers = np.reshape(inliers, (len(inliers), -1))
    outliers = np.reshape(outliers, (len(outliers), -1))

    diffInliers = inliers - inliersRes
    diffOutliers = outliers - outliersRes

    diffInliers = np.sum(diffInliers ** 2, axis = 1)
    diffOutliers = np.sum(diffOutliers ** 2, axis = 1)

    inliersLBL = np.zeros(
        (len(inliers))
    )

    outliersLBL = np.ones(
        (len(outliers))
    )

    diff = np.concatenate(
        (
            diffInliers, 
            diffOutliers
        )
    )

    label = np.concatenate(
        (
            inliersLBL, 
            outliersLBL
        )
    )

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, diff, pos_label = True)
    auc = sklearn.metrics.auc(fpr, tpr)
    return auc


def calcF1(scores,labels):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, scores)
    f1_scores = 2*recall*precision/(recall+precision)
    return np.max(f1_scores)
