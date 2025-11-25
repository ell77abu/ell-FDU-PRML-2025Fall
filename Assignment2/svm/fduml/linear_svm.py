from builtins import range
import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    结构化SVM损失函数，朴素实现（使用循环）。

    输入数据的维度为D，共有C个类别，我们在包含N个样本的小批量数据上操作。

    输入参数:
    - W: 形状为(D, C)的numpy数组，包含权重。
    - X: 形状为(N, D)的numpy数组，包含一个小批量数据。
    - y: 形状为(N,)的numpy数组，包含训练标签；y[i] = c 表示
      X[i]的标签为c，其中 0 <= c < C。
    - reg: (float) 正则化强度

    返回一个元组:
    - loss: 单个浮点数，表示损失值
    - gradient: 关于权重W的梯度；与W形状相同的数组
    """
    dW = np.zeros(W.shape)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # 同时计算梯度
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # 计算损失函数的梯度并将其存储在dW中。                                         #
    # 在计算损失的同时计算导数可能更简单                                           #
    # 可以修改上面的一些代码来计算梯度                                         # 
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 平均化梯度并添加正则化项梯度
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    结构化SVM损失函数，向量化实现。

    输入和输出与svm_loss_naive相同。
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    #############################################################################
    # TODO:                                                                     #
    # 实现结构化SVM损失的向量化版本，将结果存储在loss中。                              #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    
    # 计算所有得分: (N, C)
    scores = X.dot(W)
    
    # 获取正确类别的得分: (N,)
    correct_class_scores = scores[np.arange(num_train), y]
    
    # 计算margins: (N, C)
    margins = scores - correct_class_scores[:, np.newaxis] + 1
    
    # 将正确类别的margin设为0
    margins[np.arange(num_train), y] = 0
    
    # 只保留大于0的margins
    margins = np.maximum(0, margins)
    
    # 计算损失
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # 实现SVM损失梯度的向量化版本，将结果存储在dW中。                               #
    # 提示：与其从头计算梯度，重用一些损失计算时的中间值可能更容易                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 创建一个二进制掩码，标记哪些margin > 0
    binary = margins
    binary[margins > 0] = 1
    
    # 对于每个样本，计算有多少个错误类别的margin > 0
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -row_sum
    
    # 计算梯度
    dW = X.T.dot(binary)
    
    # 平均化并添加正则化项
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

