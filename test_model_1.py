import tensorflow as tf
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer


def display_image_predictions(features, labels, predictions):
    label_binarizer = LabelBinarizer()  # 将图片进行矩阵二值化，即将读取的像素块只分为黑或白
    label_binarizer.fit(range(LABELS_COUNT))  # fit函数用来调用后面的数据训练模型
    label_ids = label_binarizer.inverse_transform(np.array(labels))  # 将标准化后的数据转化为原始数据

    fig, axies = plt.subplots(nrows=4, ncols=2)  #创建一个4行2列的图（幕布）
    fig.tight_layout()  #tight_layout会自动调整子图参数，使之填充整个图像区域
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)  #加标题

    n_predictions = 3
    margin = 0.05
    ind = np.arange(n_predictions)  #ind = [0,1,2,3]
    width = (1. - 2. * margin) / n_predictions

    for image_i, (feature, label_id, pred_indicies, pred_values) \
        in enumerate(zip(features, label_ids, predictions.indices, predictions.values)):   #enumerate将一个可遍历的序列整合成一个索引列表，zip用于将各个数组中对应的元素打包
        pred_names = [LABEL_NAMES[pred_i] for pred_i in pred_indicies]
        correct_name = LABEL_NAMES[label_id]

        axies[image_i][0].imshow(feature)  #显示一个图像
        axies[image_i][0].set_title(correct_name)  #加标题
        axies[image_i][0].set_axis_off()  #关闭x轴和y轴

        #绘制水平直方图
        axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
        axies[image_i][1].set_yticks(ind + margin)
        axies[image_i][1].set_yticklabels(pred_names[::-1])
        axies[image_i][1].set_xticks([0, 0.5, 1.0])


save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3
