import networkx as nx  # 导入NetworkX库，用于处理图数据
import numpy as np  # 导入NumPy库，用于数值计算
import random  # 导入Random库，用于生成随机数

import math  # 导入Math库，用于数学计算

"""Adapted from https://github.com/weihua916/powerful-gnns/blob/master/util.py"""

# 定义一个S2VGraph类，用于表示图数据的结构
class S2VGraph(object):
    def __init__(self, g, node_features):
        '''
            g: 一个NetworkX图对象，表示图的拓扑结构
            neighbors: 邻居列表（不包括自环）
            node_features: 一个Torch张量，节点特征的one-hot表示，用作神经网络的输入
            edge_mat: 一个Torch长整型张量，包含边列表，将用于创建Torch稀疏张量
            max_neighbor存最大度
        '''
        self.g = g
        self.neighbors = []   # 邻居列表，每个节点的邻居节点都加入到整个列表中？
        self.node_features = node_features
        self.edge_mat = 0     # [[  0   0   0 ... 142 147 146] [  4  14  15 ... 141 143 145]] 后续会变成这样一个从上到下对应的连线
        self.max_neighbor = 0

# 定义加载数据的函数load_data
def load_data(dataset, file_list, c_i):
    '''
        dataset: 数据集的全部路径         ["D:\DataSet\DS\\node\\EMOTION\\Pears_148_obj_",
                                   A       "D:\DataSet\DS\\node\\WM\\Pears_148_obj_",
                                   B       "D:\DataSet\DS\edge\\structure_148_edge_unweight_obj_"]
        file_list: 包含数据文件列表的文件路径  C （是list.txt，里面是所有图的名称号，与edge和node的文件名称最后的位置对应）
        c_i: 当前数据集在全部路径的索引
    '''

    print('loading data')
    g_list = []  # 存储图数据的列表
    # feat_dict = {}  # 特征字典

    file_node = dataset[c_i]  # 节点特征文件的路径   A  "D:\DataSet\DS\\node\\EMOTION\\Pears_148_obj_", "D:\DataSet\DS\\node\\WM\\Pears_148_obj_".....
    file_edge = dataset[-1]   # 拓扑结构文件的路径   B  "D:\DataSet\DS\edge\\structure_148_edge_unweight_obj_" （-1指的就是列表里最后一个索引）

    with open(file_list, 'r') as f:
        num_list = f.readline().strip().split()  # 从文件 C 中读取数据文件的索引列表;readline()用于读取文件中的一行内容，返回的是一个字符串;strip()方法用于删除字符串开头和结尾的空格、换行符等空白字符;split()方法将字符串按照空格进行分割，返回一个由分割后的子字符串组成的列表

    # 读取的一个完整num_list，对应一个完整的路径如"D:\DataSet\DS\\node\\EMOTION\\Pears_148_obj_"，把里面所有的g_num都提出来做graph了
    for i in range(len(num_list)):
        name_file_node = (file_node + '%d.txt' % int(num_list[i]))  # 构建节点特征文件的路径  '%d.txt' % int(num_list[i])将整数值插入到占位符%d中，形成一个带有扩展名的文件名字符串。
        name_file_edge = (file_edge + '%d.txt' % int(num_list[i]))  # 构建拓扑结构文件的路径
        g = nx.Graph()  # 创建一个NetworkX图对象
        node_features = []  # 存储节点特征的列表，即当前nodePearson_11111.txt文件对应的FC矩阵，只不过每一个元素是一个Numpy数组

        with open(name_file_node, 'r') as f:   # 这里打开的就是上面拼好的全称文件了，也就是节点的特征文件（第一行是节点数量单独一个数字，后面是其特征矩阵）
            n_node = int(f.readline().strip())  # 读取节点数量  （读第一行的数字并删掉所有前后换行之类的符号）
            for j in range(n_node):
                g.add_node(j)  # 添加节点到图（节点读的就是从0-147,编号也就这么编了）

                row = f.readline().strip().split()  # 读取节点特征数据行 （同时把对应的节点对应的一行特征存到node_features中）
                attr = np.array(row, dtype=np.float32)  # 每行节点特征转换为NumPy数组
                node_features.append(attr)  # 将节点特征添加到列表中

        with open(name_file_edge, 'r') as f:
            n_edge = int(f.readline().strip())  # 读取边的数量
            for j in range(n_edge):
                row = f.readline().strip().split()  # 读取边的数据行  ['1', '5']  这种读取返回的是切开的字符串列表
                g.add_edge(int(row[0]) - 1, int(row[1]) - 1)  # 添加边到图，索引减一是因为从0开始  [(0, 4)]

        #print(node_features)
        #print(np.array(node_features))
        g_list.append(S2VGraph(g, np.array(node_features)))  # 将图数据构造为S2VGraph对象并添加到列表中
        #exit(0)

    # 添加标签和edge_mat属性
    for g in g_list:      # g是一个S2VGraph类型的对象，其中包含了networkX类型的图
        g.neighbors = [[] for i in range(len(g.g))]  # 初始化邻居列表 // 列表的列表[[],[],[]...]  len(g.g）是g.g这个图的节点数

        for i, j in g.g.edges():        # edges是一个（a,b），i,j分别承接a,b
            g.neighbors[i].append(j)    # 给每个点的邻接列表邻接上i-j
            g.neighbors[j].append(i)

        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))   # 把邻居数量加进去即为点的度
        g.max_neighbor = max(degree_list)  # 计算最大邻居数

        edges = [list(pair) for pair in g.g.edges()]  # 获取图的边   (把networkX类型图的每一条边对(1,2)转换成了列表形式[1,2])

        edges.extend([[i, j] for j, i in edges])  # 添加反向边

        # print(edges)  # np.array化了之后就相当于把[[0,1]] ---> [[0 1]]
        g.edge_mat = np.transpose(np.array(edges, dtype=np.int32), (1, 0))  # 转置边列表 (把原本的对应关系[[0 1][0 2]]换成了[[0 0][1 2]]
        # print(np.array(edges, dtype=np.int32))
        # print(np.transpose(np.array(edges, dtype=np.int32), (1, 0)))


    print("# data: %d" % len(g_list))  # 打印数据集大小，即这个装了很多S2VGraph类型的图像g的大小
    return g_list  # 返回图数据列表

# 定义数据分割函数separate_data （总共分了10折，fold_idx只是用来定义前多少折用于训练的）
def separate_data(graph_list, fold_idx, seed=0):
    assert 1 <= fold_idx and fold_idx < 10, "fold_idx must be from 1 to 10."

    one_sample = math.floor(len(graph_list) / 10)           # 计算每个折的样本数量，len(graph_list)表示图数据集的长度，除以10后向下取整，得到每个折的样本数量
    index = [i for i in range(0, len(graph_list))]          # 创建一个索引列表，包含了图数据集的索引，从0到len(graph_list)-1
    np.random.seed(123)                                     # 设置随机种子为123，确保每次运行时随机结果的可重复性
    np.random.shuffle(index)                                # 将索引列表进行随机打乱，以便随机划分训练集和测试集
    train_idx = index[0:one_sample * fold_idx]
    test_idx = index[one_sample * fold_idx: len(index)]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_idx, test_idx, train_graph_list, test_graph_list



from torch.utils.data import Dataset
# 继承自Dataset类并实现len和getitem方法的 导入fMRI数据集
class fMRIDataSet(Dataset):
    def __init__(self, data, label):   # 数据集，标签
        self.data = data
        self.label = label

    def __getitem__(self, index):      # 同时返回：对应index的data和其label
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)          # 返回数据集的长度（len哪个都一样，反正data--label一一对应一样长）
