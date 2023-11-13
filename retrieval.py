import torch
import pickle
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
from mix_utils import ConfusionMatrix

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FC_Net(nn.Module):
    """
    在上面的simpleNet的基础上，在每层的输出部分添加了激活函数
    """

    def __init__(self, in_dim, n_hidden_1, hash_out, cls_out, dropout=0.5):
        super(FC_Net, self).__init__()
        # self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.Tanh())
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True), nn.Dropout(dropout))

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, cls_out))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_1, hash_out), nn.Tanh())

        self.txt_layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.ReLU(True), nn.Dropout(dropout))
        self.txt_layer2 = nn.Sequential(nn.Linear(n_hidden_1, hash_out), nn.Tanh())
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """

    def forward(self, x, t):
        x = self.layer1(x)
        cls_out = self.layer2(x)
        hash_code = self.layer3(x)
        t = self.txt_layer1(t)
        t_hash = self.txt_layer2(t)
        return cls_out, hash_code, x, t_hash

def plot_tsne(features, labels, save_path):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    tsne = TSNE(n_components=2, init='pca', random_state=7)

    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]

    fig = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                          palette=sns.color_palette("hls", class_num),
                          data=df)
    scatter_fig = fig.get_figure()
    scatter_fig.savefig(save_path, dpi=400)
    plt.clf()

def plot_multi_tsne(im_features, tx_features, im_labels, tx_labels, save_path):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    tsne = TSNE(n_components=2, init='pca', random_state=7)
    
    labels = np.concatenate([im_labels, tx_labels])
    classes = ['im']*len(im_labels) + ['tx']*len(tx_labels)
    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    features = np.concatenate([im_features, tx_features])
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]
    df["c"] = classes

    fig = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), style = df.c.tolist(),
                          palette=sns.color_palette("hls", class_num),
                          data=df)
    scatter_fig = fig.get_figure()
    scatter_fig.savefig(save_path, dpi=400)
    plt.clf()


    
def load_data(data_dir, prefix):
    with open(os.path.join(data_dir, prefix+'Train_token_feature.pkl'), 'rb') as f:
        pred_data = pickle.load(f)
    train_slides = pred_data['slide_total']
    train_features = pred_data['img_feature']
    train_text_features = pred_data['text_feature']
    train_labels = pred_data['labels']
    with open(os.path.join(data_dir, prefix+'Test_token_feature.pkl'), 'rb') as f:
        pred_data = pickle.load(f)
    test_slides = pred_data['slide_total']
    test_features = pred_data['img_feature']
    test_text_features = pred_data['text_feature']
    test_labels = pred_data['labels']
    return train_features, train_text_features, train_labels, train_slides, test_features, test_text_features, test_labels, test_slides

def load_data_prompt(data_dir, prefix):
    with open(os.path.join(data_dir, prefix+'Train_token_feature.pkl'), 'rb') as f:
        pred_data = pickle.load(f)
    train_slides = pred_data['slide_total']
    train_features = pred_data['img_feature']
    train_text_features = pred_data['text_feature']
    train_labels = pred_data['labels']
    train_prompt = pred_data['prompt']
    train_pmask = pred_data['pmask']
    # train_pred = pred_data['prompt_pred']
    with open(os.path.join(data_dir, prefix+'Test_token_feature.pkl'), 'rb') as f:
        pred_data = pickle.load(f)
    test_slides = pred_data['slide_total']
    test_features = pred_data['img_feature']
    test_text_features = pred_data['text_feature']
    test_labels = pred_data['labels']
    test_prompt = pred_data['prompt']
    test_pmask = pred_data['pmask']
    # test_pred = pred_data['prompt_pred']
    # return train_pred, train_features, train_text_features, train_labels, train_slides, train_prompt, train_pmask, test_pred, test_features, test_text_features, test_labels, test_slides, test_prompt, test_pmask
    return train_features, train_text_features, train_labels, train_slides, train_prompt, train_pmask, test_features, test_text_features, test_labels, test_slides, test_prompt, test_pmask


def mean_average_precision(correct_mat, num=2000):
    index_mat = np.asarray(correct_mat[:, 0:num], np.int32)
    tmp_mat = np.asarray(index_mat, np.float)

    ave_p = tmp_mat.copy()
    for i in range(num):
        ave_p[:, i] = np.mean(tmp_mat[:, 0:(i+1)], axis=1)

    acc = np.mean(ave_p, axis=0)
    ave_p[index_mat < 1] = 0
    mean_ave_p = ave_p.copy()
    for i in range(num):
        mean_ave_p[:, i] = np.sum(
            ave_p[:, 0:(i+1)], axis=1) / (np.sum(tmp_mat[:, 0:(i+1)], axis=1) + 0.0001)
    return np.mean(mean_ave_p, axis=0), acc


def prdh_loss(pred, target):
    # target=torch.sparse.torch.eye(2).index_select(0,target)
    num = pred.shape[0]
    feature_num = pred.shape[1]
    # pred=pred-1
    w_label = 2*torch.chain_matmul(target, torch.t(target))-1
    Y = torch.chain_matmul(pred, torch.t(pred))/feature_num-w_label
    rhl = torch.sum(torch.abs(torch.abs(pred)-1))/num/feature_num
    loss = torch.sum(Y*Y/num/(num-1))+0.3*rhl
    return loss

def multi_pair_loss_no_cross(im_pred, tx_pred, target):
    device = im_pred.device
    num = im_pred.shape[0]
    multi_pred = torch.cat((im_pred, tx_pred), dim=1)
    feature_num = multi_pred.shape[1]
    # target = torch.zeros(num, 6).to(device).scatter_(1, target.unsqueeze(1), 1)
    target = target.to(device)
    # pred=pred-1
    w_label = 2*torch.chain_matmul(target, torch.t(target))-1
    cross_Y = torch.chain_matmul(
        im_pred, torch.t(tx_pred))/im_pred.shape[1]-w_label
    Y = torch.chain_matmul(multi_pred, torch.t(multi_pred))/feature_num-w_label
    cross_Y = (cross_Y*cross_Y)[torch.eye(num).to(device) == 1]
    rhl = torch.sum(torch.abs(torch.abs(multi_pred)-1))/num/feature_num
    loss = torch.sum(Y*Y/num/(num-1))+0.3*rhl+0.3*torch.sum(cross_Y/num)
    return loss

def cross_loss(pred, label):
    label = label.long()
    if len(label.size()) > 1:
        loss = torch.mean(torch.sum(
            -label*F.log_softmax(pred, dim=1), dim=1))
    else:
        loss = F.cross_entropy(pred, label)
    return loss

def pair_loss(im_cls, text_cls):
    device = im_cls.device
    num = im_cls.shape[0]
    feature_num = im_cls.shape[1]
    w_label = torch.eye(num).to(device)
    Y = torch.chain_matmul(im_cls, torch.t(text_cls))/feature_num-w_label
    Y = Y*Y
    Y = Y[w_label == 1]
    rhl = (torch.sum(torch.abs(torch.abs(im_cls)-1)) +
           torch.sum(torch.abs(torch.abs(text_cls)-1)))/num/feature_num
    loss = torch.sum(Y/num)+0.3*rhl
    return loss


def feature_out(model, feature, text_feature):
    _, hash_feature, x, text_hash = model(feature.float().cuda(), text_feature.float().cuda())
    hash_feature = hash_feature.data.cpu().numpy()
    text_hash = text_hash.data.cpu().numpy()
    binary_feature = hash_feature.copy()
    binary_feature[hash_feature <= 0] = -1
    binary_feature[hash_feature > 0] = 1
    text_binary_feature = text_hash.copy()
    text_binary_feature[text_hash <= 0] = -1
    text_binary_feature[text_hash > 0] = 1
    return hash_feature, binary_feature, text_hash, text_binary_feature

def PR_curve(test_label,correct_result, class_num):
    recall = np.zeros(correct_result.shape)
    # class_num = np.sum(correct_result, axis=1)
    percision = np.zeros(correct_result.shape)
    result_m = np.cumsum(correct_result, axis = 1)
    result_num = np.array(class_num)[test_label]
    for i in range(correct_result.shape[1]):
        recall[:,i] = np.divide(result_m[:,i],result_num)
        # sum_i = np.sum(correct_result[:,:i], axis=1)
        # np.cumsum()recall[:,i-1] = sum_i / class_num
        percision[:,i] = result_m[:,i] / (i+1)
    x1 = np.mean(recall,axis=0)
    fig,ax = plt.subplots()
    ax.plot(x1,np.mean(percision,axis=0),label='train')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('PR_Curve')
    ax.legend()
    plt.show()
    plt.savefig('./temp.png')
    a=1

def slide_retrieval_func(train_feature, train_label, train_slide, test_feature, test_label, test_slide, prefix, save_dir, modal = 'im2im_', savere5 = False):
    time0 = time.time()
    matrix = np.matmul(train_feature, test_feature.T)
    train_equal_feature = train_feature * train_feature
    test_equal_feature = test_feature * test_feature
    train_equal_feature = np.sum(train_equal_feature, axis=1)
    test_equal_feature = np.sum(test_equal_feature, axis=1)
    train_equal_feature = np.tile(train_equal_feature, (len(test_feature), 1)).T
    test_equal_feature = np.tile(test_equal_feature, (len(train_feature), 1))
    matrix = train_equal_feature + test_equal_feature - 2 * matrix
    matrix = matrix.T
    sort_matrix = np.argsort(matrix, axis=1)
    # for i in range(matrix.shape[0]):
    #     sort_matrix[i] = np.random.permutation(matrix.shape[1])
    label = train_label[sort_matrix]
    # m = 93
    # show_result(m, test_label, test_slide, test_position, train_label, sort_matrix, train_slide, train_position)
    correct_result = label == np.repeat(test_label[:, np.newaxis], len(train_label), axis=1)

    if savere5:
        index = [1, 3, 5, 10]
        results = []
        for i in index:
            results.append(np.sum(np.sum(correct_result[:,0:i], axis = 1)>0)/correct_result.shape[0])
        save_path=os.path.join('/media/disk2/hudingyi/test_program/katcl/data/gastric-all-0831/kat_model/','R@5_class_results.csv')
        if not os.path.exists(save_path):
            with open(save_path, 'a') as f:
                f.write('Loss,modal,ACC@1,ACC@3,ACC@5,ACC@10,\n')
        with open(save_path, 'a') as f:
            f.write('%s,%s,%0.3f,%0.3f,%0.3f,%0.3f,\n' % (
                prefix, modal,
                results[0],results[1],results[2],results[3]
            ))
        return
    # class_num = [np.sum(train_label.astype(int)==i) for i in range(6)]
    # PR_curve(test_label, correct_result, class_num)
    MAP1, acc1 = mean_average_precision(correct_result)
    print(time.time() - time0)
    print(MAP1[9])
    print(MAP1[19])
    print(acc1[9])
    print(acc1[19])
    m=0
    # show_result(m,test_label,test_slide,train_label,sort_matrix,train_slide)

    file_path=os.path.join(save_dir,'P@5{}_class_results.csv'.format(prefix))
    if not os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write('Class,0,1,2,3,4,5,all,\n')
        name = 'test_num'
        test_num = np.zeros(6)
        for i in range(6):
            test_num[i] = np.sum(test_label.astype(int)==i)
        with open(file_path, 'a') as f:
            f.write('%s,%d,%d,%d,%d,%d,%d,%d,\n' % (
                name,
                test_num[0],test_num[1],test_num[2],test_num[3],test_num[4],test_num[5],len(test_label)
            ))
        for i in range(6):
            test_num[i] = np.sum(train_label.astype(int)==i)
        with open(file_path, 'a') as f:
            f.write('%s,%d,%d,%d,%d,%d,%d,%d,\n' % (
                'train_num',
                test_num[0],test_num[1],test_num[2],test_num[3],test_num[4],test_num[5],len(train_label)
            ))
    MAP10 = np.zeros(6)
    ACC10 = np.zeros(6)
    for i in range(6):
        if np.sum(test_label==i)>0:
            MAP,acc=mean_average_precision(correct_result[test_label==i])
            MAP10[i] = MAP[4]
            ACC10[i] = acc[4]
    p_3 = np.sum(np.sum(correct_result[:,0:3], axis = 1)>0)/correct_result.shape[0]
    with open(file_path, 'a') as f:
        f.write('%s,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,\n' % (
            modal+'MAP10',
            MAP10[0],MAP10[1],MAP10[2],MAP10[3],MAP10[4],MAP10[5],MAP1[4]
        ))
        f.write('%s,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,\n' % (
            modal+'ACC10',
            ACC10[0],ACC10[1],ACC10[2],ACC10[3],ACC10[4],ACC10[5],acc1[4], p_3
        ))        
    f.close()
    file_path=os.path.join(save_dir,'P@3{}_class_results.csv'.format(prefix))
    with open(file_path, 'a') as f:
        f.write('%s,%0.3f,%0.3f,%0.3f,' % (
            modal+'P@3/ACC@5/MAP@5',p_3,acc1[4], MAP1[4] 
        ))        
    f.close()

def newdataslide_retrieval_func(train_feature, train_label, train_slide, test_feature, test_label, test_slide, prefix, save_dir, modal = 'im2im_', savere5 = False):
    time0 = time.time()
    matrix = np.matmul(train_feature, test_feature.T)
    train_equal_feature = train_feature * train_feature
    test_equal_feature = test_feature * test_feature
    train_equal_feature = np.sum(train_equal_feature, axis=1)
    test_equal_feature = np.sum(test_equal_feature, axis=1)
    train_equal_feature = np.tile(train_equal_feature, (len(test_feature), 1)).T
    test_equal_feature = np.tile(test_equal_feature, (len(train_feature), 1))
    matrix = train_equal_feature + test_equal_feature - 2 * matrix
    matrix = matrix.T
    sort_matrix = np.argsort(matrix, axis=1)
    # for i in range(matrix.shape[0]):
    #     sort_matrix[i] = np.random.permutation(matrix.shape[1])
    label = train_label[sort_matrix]
    # m = 93
    # show_result(m, test_label, test_slide, test_position, train_label, sort_matrix, train_slide, train_position)
    correct_result = label == np.repeat(test_label[:, np.newaxis], len(train_label), axis=1)

    if savere5:
        index = [1, 3, 5, 10]
        results = []
        for i in index:
            results.append(np.sum(np.sum(correct_result[:,0:i], axis = 1)>0)/correct_result.shape[0])
        save_path=os.path.join('/media/disk2/hudingyi/test_program/katcl/data/newdata/kat_model/','R@5_class_results.csv')
        if not os.path.exists(save_path):
            with open(save_path, 'a') as f:
                f.write('Loss,modal,ACC@1,ACC@3,ACC@5,ACC@10,\n')
        with open(save_path, 'a') as f:
            f.write('%s,%s,%0.3f,%0.3f,%0.3f,%0.3f,\n' % (
                prefix, modal,
                results[0],results[1],results[2],results[3]
            ))
        return
    # class_num = [np.sum(train_label.astype(int)==i) for i in range(6)]
    # PR_curve(test_label, correct_result, class_num)
    MAP1, acc1 = mean_average_precision(correct_result, num=200)
    print(time.time() - time0)
    print(MAP1[9])
    print(MAP1[19])
    print(acc1[9])
    print(acc1[19])
    m=0
    # show_result(m,test_label,test_slide,train_label,sort_matrix,train_slide)

    file_path=os.path.join(save_dir,'P@5{}_class_results.csv'.format(prefix))

    p_3 = np.sum(np.sum(correct_result[:,0:3], axis = 1)>0)/correct_result.shape[0]
    with open(file_path, 'a') as f:
        f.write('%s,%0.3f,%0.3f,%0.3f,\n' % (
            modal+'MAP5/ACC5/p@3',
            MAP1[4], acc1[4], p_3
        ))
      
    f.close()

def calcu_prompt_acc(test_prompt, test_pmask, train_prompt, train_pmask, sort_matrix, retrieval_index, index):
    prompt_acc = []
    for i in index:
        try:
            query_prompt = test_prompt[i][test_pmask[i]>0]
            prompt_acc_index = []
            for j in range(retrieval_index):
                return_array = train_prompt[sort_matrix[i, j]][train_pmask[sort_matrix[i, j]]>0]
                union = np.union1d(query_prompt, return_array)  
                inter = np.intersect1d(query_prompt,return_array)  
                prompt_acc_index.append(len(inter)/len(union))
            prompt_acc.append(np.mean(prompt_acc_index))
        except:
            print('False{}'.format(i))
    return np.mean(prompt_acc)
          
def slide_retrieval_prompt(train_feature, train_label, train_slide, train_prompt, train_pmask, test_feature, test_label, test_slide, test_prompt, test_pmask, prefix, save_dir, modal = 'im2im_'):
    time0 = time.time()
    matrix = np.matmul(train_feature, test_feature.T)
    train_equal_feature = train_feature * train_feature
    test_equal_feature = test_feature * test_feature
    train_equal_feature = np.sum(train_equal_feature, axis=1)
    test_equal_feature = np.sum(test_equal_feature, axis=1)
    train_equal_feature = np.tile(train_equal_feature, (len(test_feature), 1)).T
    test_equal_feature = np.tile(test_equal_feature, (len(train_feature), 1))
    matrix = train_equal_feature + test_equal_feature - 2 * matrix
    matrix = matrix.T
    sort_matrix = np.argsort(matrix, axis=1)
    # for i in range(matrix.shape[0]):
    #     sort_matrix[i] = np.random.permutation(matrix.shape[1])
    label = train_label[sort_matrix]

    ###prompt_acc
    prompt_acc_10 = []
    prompt_acc_20 = []
    for i in range(6):
        p_acc = calcu_prompt_acc(test_prompt, test_pmask, train_prompt, train_pmask, sort_matrix, 10, np.arange(len(test_label))[test_label.astype(int)==i])
        prompt_acc_10.append(p_acc)
        p_acc = calcu_prompt_acc(test_prompt, test_pmask, train_prompt, train_pmask, sort_matrix, 20, np.arange(len(test_label))[test_label.astype(int)==i])
        prompt_acc_20.append(p_acc)
    p_acc = calcu_prompt_acc(test_prompt, test_pmask, train_prompt, train_pmask, sort_matrix, 10, range(len(test_label)))
    prompt_acc_10.append(p_acc)
    p_acc = calcu_prompt_acc(test_prompt, test_pmask, train_prompt, train_pmask, sort_matrix, 20, range(len(test_label)))
    prompt_acc_20.append(p_acc)
    
    file_path=os.path.join(save_dir,'prompt_acc_{}_class_results.csv'.format(prefix))
    if not os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write('Class,0,1,2,3,4,5,all,\n')
        name = 'test_num'
        test_num = np.zeros(6)
        for i in range(6):
            test_num[i] = np.sum(test_label.astype(int)==i)
        with open(file_path, 'a') as f:
            f.write('%s,%d,%d,%d,%d,%d,%d,%d,\n' % (
                name,
                test_num[0],test_num[1],test_num[2],test_num[3],test_num[4],test_num[5],len(test_label)
            ))
        for i in range(6):
            test_num[i] = np.sum(train_label.astype(int)==i)
        with open(file_path, 'a') as f:
            f.write('%s,%d,%d,%d,%d,%d,%d,%d,\n' % (
                'train_num',
                test_num[0],test_num[1],test_num[2],test_num[3],test_num[4],test_num[5],len(train_label)
            ))

    with open(file_path, 'a') as f:
        f.write('%s,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,\n' % (
            modal+'PAcc10',
            prompt_acc_10[0],prompt_acc_10[1],prompt_acc_10[2],prompt_acc_10[3],prompt_acc_10[4],prompt_acc_10[5],prompt_acc_10[6]
        ))
        f.write('%s,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,\n' % (
            modal+'PACC20',
            prompt_acc_20[0],prompt_acc_20[1],prompt_acc_20[2],prompt_acc_20[3],prompt_acc_20[4],prompt_acc_20[5],prompt_acc_20[6]
        ))        
    f.close()


def pred_retrieval_prompt(train_pred, train_label, train_slide, train_prompt, train_pmask, test_pred, test_label, test_slide, test_prompt, test_pmask, prefix, save_dir, modal = 'im2im_'):
    time0 = time.time()
    threshold = np.mean(train_pred)*1.5
    train_prompt_pred = np.zeros(train_pred.shape)
    # for i in range(train_pred.shape[0]):
    #     train_prompt_pred[i, train_prompt[i][train_pmask[i]>0]] = 1
    train_prompt_pred[train_pred>threshold] = 1
    test_prompt_pred = np.zeros(test_pred.shape)
    test_prompt_pred[test_pred>threshold] = 1
    matrix = np.matmul(test_prompt_pred, train_prompt_pred.T)
    sort_matrix = np.argsort(-matrix, axis=1)

    ###prompt_acc
    prompt_acc_10 = []
    prompt_acc_20 = []
    for i in range(6):
        p_acc = calcu_prompt_acc(test_prompt, test_pmask, train_prompt, train_pmask, sort_matrix, 10, np.arange(len(test_label))[test_label.astype(int)==i])
        prompt_acc_10.append(p_acc)
        p_acc = calcu_prompt_acc(test_prompt, test_pmask, train_prompt, train_pmask, sort_matrix, 20, np.arange(len(test_label))[test_label.astype(int)==i])
        prompt_acc_20.append(p_acc)
    p_acc = calcu_prompt_acc(test_prompt, test_pmask, train_prompt, train_pmask, sort_matrix, 10, range(len(test_label)))
    prompt_acc_10.append(p_acc)
    p_acc = calcu_prompt_acc(test_prompt, test_pmask, train_prompt, train_pmask, sort_matrix, 20, range(len(test_label)))
    prompt_acc_20.append(p_acc)
    
    file_path=os.path.join(save_dir,'prompt_acc_{}_class_pred_results.csv'.format(prefix))
    if not os.path.exists(file_path):
        with open(file_path, 'a') as f:
            f.write('Class,0,1,2,3,4,5,all,\n')
        name = 'test_num'
        test_num = np.zeros(6)
        for i in range(6):
            test_num[i] = np.sum(test_label.astype(int)==i)
        with open(file_path, 'a') as f:
            f.write('%s,%d,%d,%d,%d,%d,%d,%d,\n' % (
                name,
                test_num[0],test_num[1],test_num[2],test_num[3],test_num[4],test_num[5],len(test_label)
            ))
        for i in range(6):
            test_num[i] = np.sum(train_label.astype(int)==i)
        with open(file_path, 'a') as f:
            f.write('%s,%d,%d,%d,%d,%d,%d,%d,\n' % (
                'train_num',
                test_num[0],test_num[1],test_num[2],test_num[3],test_num[4],test_num[5],len(train_label)
            ))

    with open(file_path, 'a') as f:
        f.write('%s,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,\n' % (
            modal+'PAcc10',
            prompt_acc_10[0],prompt_acc_10[1],prompt_acc_10[2],prompt_acc_10[3],prompt_acc_10[4],prompt_acc_10[5],prompt_acc_10[6]
        ))
        f.write('%s,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,\n' % (
            modal+'PACC20',
            prompt_acc_20[0],prompt_acc_20[1],prompt_acc_20[2],prompt_acc_20[3],prompt_acc_20[4],prompt_acc_20[5],prompt_acc_20[6]
        ))        
    f.close()
    

def train():
    prefix = 'all_hash'
    batch_size = 50
    hash_bit = 32
    num_epochs = 25
    data_dir = '/media/disk2/hudingyi/test_program/katcl/data/gastric-all-0831/kat_model/multi_kernel_all[l1t224s112m500][p50n5i25][f5_t30][resnet50_td_cl][fs224][list_fold_0][m2048][d6_h_8_de256dm512dh64_cls][npk_144][t5]/'
    train_features, train_text_features, train_labels, train_slides, test_features, test_text_features, test_labels, test_slides = load_data(
        data_dir)

    train_feature = torch.from_numpy(train_features)
    train_text_feature = torch.from_numpy(train_text_features)
    train_label = torch.from_numpy(train_labels)
    # train_slide=torch.from_numpy(train_slide)
    train_data = TensorDataset(train_feature, train_text_feature, train_label)
    # train_loader=DataLoader(dataset=train_data,batch_size=batch_size,pin_memory=(torch.cuda.is_available()), sampler=sampler,num_workers=4)
    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_feature = torch.from_numpy(test_features)
    test_text_feature = torch.from_numpy(test_text_features)
    model = FC_Net(256, 128, hash_bit, 6)

    losses = AverageMeter()
    if torch.cuda.is_available():
        model = model.cuda()
    model_dir = os.path.join(data_dir, '{}_retreival_model.dat'.format(prefix))
    if os.path.exists(model_dir):
        model.load_state_dict(torch.load(model_dir))
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.0001)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * num_epochs, 0.75 * num_epochs],
        #                                                 gamma=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(0.2*num_epochs), eta_min=5e-8)
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        # 训练模型
        st = time.time()
        for i in range(num_epochs):
            iters = 0
            for data in train_loader:
                feature, text, label = data
                feature = feature.float()
                text = text.float()

                if torch.cuda.is_available():
                    feature = feature.cuda()
                    text = text.cuda()
                    y = torch.eye(6)
                    target = y[label.long()]
                    # target=y[label]
                    target = target.cuda()
                    label = label.cuda()
                else:
                    feature = Variable(feature)
                    text = Variable(text)
                    label = Variable(label)
                cls_out, hash_code, x, t_hash = model(feature, text)
                loss = prdh_loss(hash_code, target) + \
                    cross_loss(cls_out, label) + multi_pair_loss_no_cross(hash_code, t_hash, target)*0.4 + pair_loss(hash_code, t_hash)
                losses.update(loss.item(), batch_size)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iters += 1
                if iters % 10 == 0:
                    res = '\t'.join([
                        'Epoch: [%d]' % (i),
                        'Iter: [%d/%d]' % (iters, len(train_loader)),
                        'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                    ])
                    print(res)
            scheduler.step()

            print(time.time()-st)
        torch.save(model.state_dict(), model_dir)
    model.eval()
    with torch.no_grad():
        train_hash_feature, train_binary_feature, train_text_hash, train_text_binary = feature_out(
            model, train_feature, train_text_feature)
        test_hash_feature, test_binary_feature, test_text_hash, test_text_binary = feature_out(model, test_feature, test_text_feature)

        graph_save_path = os.path.join(data_dir, '{}_hash_feature.pkl'.format(prefix))
        with open(graph_save_path, 'wb') as f:
            graph = {
                'train_hash_feature': train_hash_feature,
                'train_binary_feature': train_binary_feature,
                'train_text_hash': train_text_hash,
                'train_text_binary': train_text_binary,
                'test_hash_feature': test_hash_feature,
                'test_binary_feature': test_binary_feature,
                'test_text_hash': test_text_hash,
                'test_text_binary': test_text_binary,
                'train_labels': train_labels,
                'test_labels': test_labels,
                'train_slides': train_slides,
                'test_slides': test_slides,
            }
            pickle.dump(graph, f)


def retrieval():
    data_dir = '/media/disk2/hudingyi/test_program/katcl/data/gastric-all-0831/kat_model/multi_kernel_all[l1t224s112m500][p50n5i25][f5_t30][resnet50_td_cl][fs224][list_fold_0][m2048][d6_h_8_de256dm512dh64_cls][npk_144][t5]/'
    with open(os.path.join(data_dir, 'hash_feature.pkl'), 'rb') as f:
        data = pickle.load(f)
    for class_index in range(6):
        train_feature = data['train_binary_feature']
        # train_feature = data['train_hash_feature']
        train_label = data['train_labels']
        train_slide = data['train_slides']

        test_feature = data['test_binary_feature']
        # test_feature = data['test_hash_feature']
        test_label = data['test_labels']
        test_slide = data['test_slides']

        label_index = test_label == class_index
        test_label = test_label[label_index]
        test_feature = test_feature[label_index]

        time0 = time.time()
        matrix = np.matmul(train_feature, test_feature.T)
        # train_equal_feature=train_feature*train_feature
        # test_equal_feature=test_feature*test_feature
        # train_equal_feature=np.sum(train_equal_feature,axis=1)
        # test_equal_feature=np.sum(test_equal_feature,axis=1)
        # train_equal_feature=np.tile(train_equal_feature,(len(test_feature),1)).T
        # test_equal_feature=np.tile(test_equal_feature,(len(train_feature),1))
        # matrix=train_equal_feature+test_equal_feature-2*matrix
        matrix = matrix.T
        # sort_matrix=np.argsort(-matrix, axis=1)
        sort_matrix = np.argsort(-matrix, axis=1)
        label = train_label[sort_matrix]
        # slide=train_slide[sort_matrix]
        # RAE_entropy_20=entropy(slide,slide_num=20,slide_index=int(np.max(train_slide)+1))
        # RAE_entropy_200=entropy(slide,slide_num=200,slide_index=int(np.max(train_slide)+1))
        correct_result = label == np.repeat(
            test_label[:, np.newaxis], len(train_label), axis=1)
        MAP1, acc1 = mean_average_precision(correct_result)
        print(time.time()-time0)
        print(MAP1[9])
        print(MAP1[19])
        print(acc1[9])
        print(acc1[19])

        file_path = os.path.join(data_dir, 'retrieval_acc.csv')
        if not os.path.exists(file_path):
            with open(file_path, 'a') as f:
                f.write('class,map10,map20,acc10,acc20,\n')
        with open(file_path, 'a') as f:
            f.write('%d,%0.3f,%0.3f,%0.3f,%0.3f,\n' % (
                class_index,
                MAP1[9], MAP1[19], acc1[9], acc1[19]
            ))

def matrix_plot(train_feature, train_label, test_feature, test_label, prefix, data_dir, modal='im2im_'):
    matrix = np.matmul(train_feature, test_feature.T)
    train_equal_feature = train_feature * train_feature
    test_equal_feature = test_feature * test_feature
    train_equal_feature = np.sum(train_equal_feature, axis=1)
    test_equal_feature = np.sum(test_equal_feature, axis=1)
    train_equal_feature = np.tile(train_equal_feature, (len(test_feature), 1)).T
    test_equal_feature = np.tile(test_equal_feature, (len(train_feature), 1))
    matrix = train_equal_feature + test_equal_feature - 2 * matrix
    matrix = matrix.T
    sort_matrix = np.argsort(matrix, axis=1)
    label = train_label[sort_matrix]
    img_path = os.path.join(data_dir, 'matrix')
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    class_list = [0, 1, 2, 3, 4, 5] 
    confusion_matrix = ConfusionMatrix(class_list,seve_dir=img_path+'/'+prefix+'_'+modal)
    for i in range(matrix.shape[0]):
        confusion_matrix.update_matrix([label[i, 0]], labels=[test_label[i]])
    confusion_matrix.plot_confusion_matrix()
    a =1



def cross_retrieval(prefix):
    # prefix = 'all_hash'
    # data_dir = '/media/disk2/hudingyi/test_program/katcl/data/gastric-all-0831/kat_model/multi_kernel_all[l1t224s112m500][p50n5i25][f5_t30][resnet50_td_cl][fs224][list_fold_0][m2048][d6_h_8_de256dm512dh64_cls][npk_144][t5]/'
    # with open(os.path.join(data_dir, '{}_hash_feature.pkl'.format(prefix)), 'rb') as f:
    #     data = pickle.load(f)    

    # train_im_feature = data['train_hash_feature']
    # train_tx_feature = data['train_text_hash']
    # # train_feature = data['train_hash_feature']
    # train_label = data['train_labels']
    # train_slide = data['train_slides']

    # test_im_feature = data['test_hash_feature']
    # test_tx_feature = data['test_text_hash']
    # # test_feature = data['test_hash_feature']
    # test_label = data['test_labels']
    # test_slide = data['test_slides']

    data_dir = '/media/disk2/hudingyi/test_program/katcl/data/gastric-all-0831/kat_model/{}[l1t224s112m500][p50n5i25][f5_t30][resnet50_td_cl][fs224][list_fold_0][m2048][d6_h_8_de256dm512dh64_cls][npk_144][t5]/'.format(prefix)
    train_im_feature, train_tx_feature, train_label, train_slide, test_im_feature, test_tx_feature, test_label, test_slide = load_data(
        data_dir, prefix)
    
    slide_retrieval_func(train_tx_feature, train_label, train_slide, test_im_feature, test_label, test_slide, prefix, data_dir, modal='im2tx_')
    slide_retrieval_func(train_im_feature, train_label, train_slide, test_tx_feature, test_label, test_slide, prefix, data_dir, modal='tx2im_')
    slide_retrieval_func(train_im_feature, train_label, train_slide, test_im_feature, test_label, test_slide, prefix, data_dir, modal='im2im_')
    slide_retrieval_func(train_tx_feature, train_label, train_slide, test_tx_feature, test_label, test_slide, prefix, data_dir, modal='tx2tx_')

def newdata_cross_retrieval(prefix):
    data_dir = '/media/disk2/hudingyi/test_program/katcl/data/newdata/kat_model/{}[l1t224s112m500][p50n5i25][f5_t30][resnet50_td_cl][fs224][list_fold_0][m2048][d6_h_8_de256dm512dh64_cls][npk_144][t5]/'.format(prefix)
    train_im_feature, train_tx_feature, train_label, train_slide, test_im_feature, test_tx_feature, test_label, test_slide = load_data(
        data_dir, prefix)
    
    newdataslide_retrieval_func(train_tx_feature, train_label, train_slide, test_im_feature, test_label, test_slide, prefix, data_dir, modal='im2tx_')
    newdataslide_retrieval_func(train_im_feature, train_label, train_slide, test_tx_feature, test_label, test_slide, prefix, data_dir, modal='tx2im_')
    newdataslide_retrieval_func(train_im_feature, train_label, train_slide, test_im_feature, test_label, test_slide, prefix, data_dir, modal='im2im_')
    newdataslide_retrieval_func(train_tx_feature, train_label, train_slide, test_tx_feature, test_label, test_slide, prefix, data_dir, modal='tx2tx_')

    train_im_feature, train_tx_feature, train_label, train_slide, train_prompt, train_pmask, test_im_feature, test_tx_feature, test_label, test_slide, test_prompt, test_pmask = load_data_prompt(
        data_dir, prefix) 
    slide_retrieval_prompt(train_tx_feature, train_label, train_slide, train_prompt, train_pmask, test_im_feature, test_label, test_slide, test_prompt, test_pmask, prefix, data_dir, modal='im2tx_')
    slide_retrieval_prompt(train_im_feature, train_label, train_slide, train_prompt, train_pmask, test_tx_feature, test_label, test_slide, test_prompt, test_pmask, prefix, data_dir, modal='tx2im_')
    slide_retrieval_prompt(train_im_feature, train_label, train_slide, train_prompt, train_pmask, test_im_feature, test_label, test_slide, test_prompt, test_pmask, prefix, data_dir, modal='im2im_')
    slide_retrieval_prompt(train_tx_feature, train_label, train_slide, train_prompt, train_pmask, test_tx_feature, test_label, test_slide, test_prompt, test_pmask, prefix, data_dir, modal='tx2tx_')
    
def draw_tsne(prefix):
    # prefix = 'all_hash'
    # data_dir = '/media/disk2/hudingyi/test_program/katcl/data/gastric-all-0831/kat_model/multi_kernel_all[l1t224s112m500][p50n5i25][f5_t30][resnet50_td_cl][fs224][list_fold_0][m2048][d6_h_8_de256dm512dh64_cls][npk_144][t5]/'
    # with open(os.path.join(data_dir, '{}_hash_feature.pkl'.format(prefix)), 'rb') as f:
    #     data = pickle.load(f)  
    # train_im_feature = data['train_hash_feature']
    # train_tx_feature = data['train_text_hash']
    # # train_feature = data['train_hash_feature']
    # train_label = data['train_labels']
    # train_slide = data['train_slides']

    # test_im_feature = data['test_hash_feature']
    # test_tx_feature = data['test_text_hash']    
    # test_label = data['test_labels']

    data_dir = '/media/disk2/hudingyi/test_program/katcl/data/gastric-all-0831/kat_model/{}[l1t224s112m500][p50n5i25][f5_t30][resnet50_td_cl][fs224][list_fold_0][m2048][d6_h_8_de256dm512dh64_cls][npk_144][t5]/'.format(prefix)
    train_im_feature, train_tx_feature, train_label, train_slide, test_im_feature, test_tx_feature, test_label, test_slide = load_data(
        data_dir, prefix)
    save_dir = os.path.join(data_dir, 'tsne_img')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plot_multi_tsne(train_im_feature, train_tx_feature, train_label, train_label, save_dir+'/{}_train_cross_tsne.jpg'.format(prefix))
    plot_multi_tsne(test_im_feature, test_tx_feature, test_label, test_label, save_dir+'/{}_test_cross_tsne.jpg'.format(prefix))
    plot_tsne(train_im_feature, train_label, save_dir+'/{}_train_img_tsne.jpg'.format(prefix))
    plot_tsne(train_tx_feature, train_label, save_dir+'/{}_train_text_tsne.jpg'.format(prefix))
    plot_tsne(test_im_feature, test_label, save_dir+'/{}_test_img_tsne.jpg'.format(prefix))
    plot_tsne(test_tx_feature, test_label, save_dir+'/{}_test_text_tsne.jpg'.format(prefix))

def deaw_matrix(prefix):
    data_dir = '/media/disk2/hudingyi/test_program/katcl/data/gastric-all-0831/kat_model/{}[l1t224s112m500][p50n5i25][f5_t30][resnet50_td_cl][fs224][list_fold_0][m2048][d6_h_8_de256dm512dh64_cls][npk_144][t5]/'.format(prefix)
    train_im_feature, train_tx_feature, train_label, train_slide, test_im_feature, test_tx_feature, test_label, test_slide = load_data(
        data_dir, prefix)
    matrix_plot(train_im_feature, train_label, test_im_feature, test_label, prefix, data_dir, modal='im2im_')    
    matrix_plot(train_tx_feature, train_label, test_im_feature, test_label, prefix, data_dir, modal='im2tx_')
    matrix_plot(train_tx_feature, train_label, test_tx_feature, test_label, prefix, data_dir, modal='tx2tx_')
    matrix_plot(train_im_feature, train_label, test_tx_feature, test_label, prefix, data_dir, modal='tx2im_')
    
def prompt_acc_retrieval(prefix):
    data_dir = '/media/disk2/hudingyi/test_program/katcl/data/gastric-all-0831/kat_model/{}[l1t224s112m500][p50n5i25][f5_t30][resnet50_td_cl][fs224][list_fold_0][m2048][d6_h_8_de256dm512dh64_cls][npk_144][t5]/'.format(prefix)
    # train_pred, train_im_feature, train_tx_feature, train_label, train_slide, train_prompt, train_pmask, test_pred, test_im_feature, test_tx_feature, test_label, test_slide, test_prompt, test_pmask = load_data_prompt(
    #     data_dir)
    train_im_feature, train_tx_feature, train_label, train_slide, train_prompt, train_pmask, test_im_feature, test_tx_feature, test_label, test_slide, test_prompt, test_pmask = load_data_prompt(
        data_dir, prefix)    
    # pred_retrieval_prompt(train_pred, train_label, train_slide, train_prompt, train_pmask, test_pred, test_label, test_slide, test_prompt, test_pmask, prefix, data_dir, modal = 'Pred_')
    
    slide_retrieval_prompt(train_tx_feature, train_label, train_slide, train_prompt, train_pmask, test_im_feature, test_label, test_slide, test_prompt, test_pmask, prefix, data_dir, modal='im2tx_')
    slide_retrieval_prompt(train_im_feature, train_label, train_slide, train_prompt, train_pmask, test_tx_feature, test_label, test_slide, test_prompt, test_pmask, prefix, data_dir, modal='tx2im_')
    slide_retrieval_prompt(train_im_feature, train_label, train_slide, train_prompt, train_pmask, test_im_feature, test_label, test_slide, test_prompt, test_pmask, prefix, data_dir, modal='im2im_')
    slide_retrieval_prompt(train_tx_feature, train_label, train_slide, train_prompt, train_pmask, test_tx_feature, test_label, test_slide, test_prompt, test_pmask, prefix, data_dir, modal='tx2tx_')

def PR_results(prefixs):
    fig,ax = plt.subplots()
    for prefix in prefixs:
        data_dir = '/media/disk2/hudingyi/test_program/katcl/data/gastric-all-0831/kat_model/{}[l1t224s112m500][p50n5i25][f5_t30][resnet50_td_cl][fs224][list_fold_0][m2048][d6_h_8_de256dm512dh64_cls][npk_144][t5]/'.format(prefix)
        train_feature, train_tx_feature, train_label, train_slide, test_feature, test_tx_feature, test_label, test_slide = load_data(
            data_dir, prefix)   
        matrix = np.matmul(train_feature, test_feature.T)
        train_equal_feature = train_feature * train_feature
        test_equal_feature = test_feature * test_feature
        train_equal_feature = np.sum(train_equal_feature, axis=1)
        test_equal_feature = np.sum(test_equal_feature, axis=1)
        train_equal_feature = np.tile(train_equal_feature, (len(test_feature), 1)).T
        test_equal_feature = np.tile(test_equal_feature, (len(train_feature), 1))
        matrix = train_equal_feature + test_equal_feature - 2 * matrix
        matrix = matrix.T
        sort_matrix = np.argsort(matrix, axis=1)
        # for i in range(matrix.shape[0]):
        #     sort_matrix[i] = np.random.permutation(matrix.shape[1])
        label = train_label[sort_matrix]
        # m = 93
        # show_result(m, test_label, test_slide, test_position, train_label, sort_matrix, train_slide, train_position)
        correct_result = label == np.repeat(test_label[:, np.newaxis], len(train_label), axis=1)  
        class_num = [np.sum(train_label.astype(int)==i) for i in range(6)]   
        recall = np.zeros(correct_result.shape)
        # class_num = np.sum(correct_result, axis=1)
        percision = np.zeros(correct_result.shape)
        result_m = np.cumsum(correct_result, axis = 1)
        result_num = np.array(class_num)[test_label]
        for i in range(correct_result.shape[1]):
            recall[:,i] = np.divide(result_m[:,i],result_num)
            # sum_i = np.sum(correct_result[:,:i], axis=1)
            # np.cumsum()recall[:,i-1] = sum_i / class_num
            percision[:,i] = result_m[:,i] / (i+1)
        x1 = np.mean(recall,axis=0) 
        ax.plot(x1,np.mean(percision,axis=0),label=prefix)

    graph_list_dir = './data/gastric-all-0831/RetCCL'
    train_data_path = os.path.join(graph_list_dir, 'train_data.pkl')
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    test_data_path = os.path.join(graph_list_dir, 'test_data.pkl')
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    retrieval_path = os.path.join(graph_list_dir, 'retrieval_50.pkl')
    with open(retrieval_path, 'rb') as f:
        retrieval_results = pickle.load(f)

    index_t = retrieval_results['index']
    test_label = np.zeros(index_t)
    for i in range(index_t):
        try:
            test_label[i] = test_data['label'][test_data['slide_idx']==i][0]
        except:
            print(i)
    # test_label = test_data['label'][0:index_t]
    train_label = train_data['label']
    sort_matrix = retrieval_results['sort_result'].astype(int)[0:index_t]
    label = train_label[sort_matrix]
    correct_result = label == np.repeat(test_label[:, np.newaxis], 50, axis=1)
    recall = np.zeros(correct_result.shape)
    # class_num = np.sum(correct_result, axis=1)
    percision = np.zeros(correct_result.shape)
    result_m = np.cumsum(correct_result, axis = 1)
    for i in range(correct_result.shape[1]):
        recall[:,i] = np.divide(result_m[:,i],result_num[:1079])
        # sum_i = np.sum(correct_result[:,:i], axis=1)
        # np.cumsum()recall[:,i-1] = sum_i / class_num
        percision[:,i] = result_m[:,i] / (i+1)
    x1 = np.mean(recall,axis=0) 
    ax.plot(x1,np.mean(percision,axis=0),label='RetCCL')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('PR_Curve')
    ax.legend()
    plt.show()
    plt.savefig('./temp.png')
    a=1


# prefixs = ['all_6_loss', 'wo_w_loss', 'wo_cls_loss', 'wo_cross_sim_loss', 'wo_mlm_loss', 'wo_ITA_loss', 'wo_ICLS_loss']
prefixs = ['wo_ICLS_loss', 'BIBE', 'vit']
# prefixs = ['prompt-text', 'all-loss-prompt', 'p_anchor_36', 'p_anchor_64', 'p_anchor_100']
# prefixs = ['weight_PCL', 'doubel_PCL_weight', 'doubel_PCL']
# prefixs = ['vit', 'text']
# prefixs = ['small_prompt', 'large_prompt']
# prefixs = ['all_6_loss', 'wo_w_loss', 'wo_cls_loss', 'wo_cross_sim_loss', 'wo_mlm_loss', 'wo_ITA_loss', 'wo_ICLS_loss', 'BIBE']
prefixs = ['propose', 'WO_PC', 'WO_CTA', 'WO_MLM', 'WO_WRA', 'WO_APA']
prefixs = ['p_anchor_36', 'p_anchor_64', 'p_anchor_100']
prefixs = ['cross-35', 'kat']
prefixs = ['setmil']
# prefixs = ['8vit', '8transmil', '8kat', '8DTN', '8flip']

# PR_results(prefixs)
# prefixs = ['cross-35', 'kat', 'vit']
for prefix in prefixs:
    print(prefix)
    data_dir = '/media/disk2/hudingyi/test_program/katcl/data/gastric-all-0831/kat_model/{}[l1t224s112m500][p50n5i25][f5_t30][resnet50_td_cl][fs224][list_fold_0][m2048][d6_h_8_de256dm512dh64_cls][npk_144][t5]/'.format(prefix)
    save_dir = os.path.join(data_dir, 'tsne_img')
    # if  os.path.exists(save_dir) :
    #     continue
    #    # prefix = 'all_6_loss'
    # newdata_cross_retrieval(prefix)
    cross_retrieval(prefix)
    # draw_tsne(prefix)
    # # deaw_matrix(prefix)
    prompt_acc_retrieval(prefix)    
# train()

# retrieval()


a = 1
