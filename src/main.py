from model import Net_50, Net_18, LeNet
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter  # 导入命令行参数解析工具
from util import *    # 自己的导入数据用的工具包
from torch.utils.data import DataLoader, Dataset

if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False

# 检测是否有GPU可用，如果有就使用GPU，否则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建一个命令行参数解析器
parser = ArgumentParser("UGformer", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
# 添加命令行参数，这些参数用于配置脚本的运行
parser.add_argument("--run_folder", default="../", help="运行文件夹")
parser.add_argument("--dataset", default=[
                                          "D:\DataSet\DS\\node\\EMOTION\\Pears_148_obj_",
                                          "D:\DataSet\DS\\node\\GAMBLING\\Pears_148_obj_",
                                          "D:\DataSet\DS\\node\\LANGUAGE\\Pears_148_obj_",
                                          "D:\DataSet\DS\\node\\MOTOR\\Pears_148_obj_",
                                          "D:\DataSet\DS\\node\\RELATIONAL\\Pears_148_obj_",
                                          "D:\DataSet\DS\\node\\SOCIAL\\Pears_148_obj_",
                                          "D:\DataSet\DS\\node\\WM\\Pears_148_obj_",

                                          "D:\DataSet\DS\edge\\structure_148_edge_unweight_obj_"   ], help="数据集路径")  # dataset就是这样一个路径的列表，node和edge都要，最后那个“_”后面是节点的标号

parser.add_argument("--num_list", default="D:\DataSet\DS\\list.txt", help="节点标号列表路径")  # num_list是所用的节点的标号，文件名中都有
parser.add_argument("--num_node", default=148, help="节点数")
parser.add_argument("--batch_size", default=8, type=int, help="批处理大小")
parser.add_argument('--fold_idx', type=int, default=7, help='折叠索引，取值1-9')
parser.add_argument('--epoch_num', type=int, default=100, help='epoch轮次数量')
parser.add_argument('--lr', default=0.001, help='学习率')
args = parser.parse_args()  # 解析命令行参数并存储在args变量中

def load_fMRIdata():
    print("Loading data...")  # 输出提示信息，表示正在加载数据

    graphs = []  # S2VGraph类型的无向图集合，每个S2VGraph里包含一个g即networkX类型的图
    lable = []   # 对应上面无向图的类标
    # 循环遍历数据集路径，加载数据  （-1是去除了最后那个边的路径，只留FC矩阵含类标的）
    for i in range(len(args.dataset) - 1):
        graphs_c = load_data(args.dataset, args.num_list, i)  # i是当前所用的dataset路径序号
        lable_c = (np.zeros(
            len(graphs_c)) + i).tolist()  # np.zeros生成一个graphs_c那么长的列表，即对应每一张图都有一个标签（具体的标签其实是i,因为广播特性矩阵加一个i即每个元素都加了i，最后再转化为列表）
        graphs = graphs + graphs_c
        lable = lable + lable_c

    th = 0
    # 根据阈值th对图数据进行处理
    # 比阈值小的相关性都设为0
    for i in range(len(graphs)):
        graphs[i].node_features[graphs[i].node_features < th] = 0

    train_idx, test_idx, train_graphs, test_graphs = separate_data(graphs, args.fold_idx)  # 分割训练集和测试集

    # 把所有train数据和test数据拼成一张大表并转为tensor_3d 三维张量，第一维度给成序号
    def concat(S2VGraph_a):
        tensor_3d = torch.empty((len(S2VGraph_a), args.num_node, args.num_node))
        for i in range(len(S2VGraph_a)):
            tensor_3d[i] = torch.from_numpy(S2VGraph_a[i].node_features)

        # tensor_3d现在是一个三维张量，第一维度是批次序号
        # print(tensor_3d.shape)
        # exit(0)
        return tensor_3d

    train_all_ds = concat(train_graphs)
    test_all_ds = concat(test_graphs)

    train_label = [lable[i] for i in train_idx]  # 训练集标签
    test_label = [lable[i] for i in test_idx]  # 测试集标签

    feature_dim_size = graphs[0].node_features.shape[1]  # 获取节点特征的维度大小  (node_features也就是FC矩阵，行号对应具体点号，列数对应的是特征维度）

    print("######feature_dim_size:"+str(feature_dim_size))
    print("Loading data... finished!")  # 输出加载数据完成的信息
    print()

    return feature_dim_size, train_all_ds, train_label, test_all_ds, test_label

# 返回一个数据加载器，此处分别返回了train_iter和test_iter
def load_dataloader(data_train, data_test, label_train, label_test, batch_size = args.batch_size):  # 默认为args里的batchsize
    '''
    DataLoader是一个数据加载器，用于从Dataset对象中加载数据并生成小批量的数据。它提供了数据的批量加载、并行加载和数据重排的功能。
    返回一个数据加载器，此处分别返回了train和test的iter
    '''
    train_iter = DataLoader(dataset=fMRIDataSet(data_train, label_train),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1)

    test_iter = DataLoader(dataset=fMRIDataSet(data_test, label_test),
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=1)

    return train_iter, test_iter

def train(net, train_loader, epoch, optimizer, criterion):
    losses = []
    correct, total = 0, 0
    for i, (input, target) in enumerate(train_loader):
        if use_gpu:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        optimizer.zero_grad()
        outputs = net(input_var)
        #print(outputs)
        #print('###############')
        #print(target_var)

        pred = outputs.argmax(dim=1)  # output张量每行最大值的索引，每行都是num_class个维度，哪一维度值大就分到哪一维度对应的类
        correct += (pred == target_var).sum().item()
        total += len(target_var)
        accuracy = correct / total
        #print(accuracy)

        loss = criterion(outputs, target_var.long())
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("epoch:{}, batch:{}, loss:{}, acc:{}".format(epoch+1, i, loss.item(), accuracy))
        losses.append(loss.item())

    return sum(losses) / len(losses)


def test(net, test_loader):
    accs = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if use_gpu:
                input = input.cuda()
                target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            outputs = net(input_var)
            prec1, prec5 = accuracy(outputs.data, target, topk=(1, 3))
            accs.append(prec1.item())

    return sum(accs) / len(accs)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def plot_fig(epochs, test_accs, pic_name):
    import matplotlib.pyplot as plt
    plt.plot(epochs, test_accs)
    plt.xlabel("Epochs")
    plt.ylabel("{}".format(pic_name))
    plt.title("Epochs v.s. {}".format(pic_name))
    plt.savefig("../fig/{}.png".format(pic_name))
    plt.close()


if __name__ == '__main__':

    print("Loading net...")
    if use_gpu:
        net = Net_50().cuda()
    else:
        net = Net_50()
    print("Done")

    print("Loading dataset...")
    feature_dim_size, train_data, train_label, test_data, test_label = load_fMRIdata()
    #print(train_data.size())
    train_data = torch.unsqueeze(train_data, dim=1)
    test_data = torch.unsqueeze(test_data, dim=1)
    #print(train_data.size())
    #exit(0)
    train_loader, test_loader = load_dataloader(train_data, test_data, train_label, test_label)
    print("Done")

    print("Set optimizer and loss function")
    criterion = nn.CrossEntropyLoss().to(device)  # 使用这个函数需要注意：标签是整数，不要onehot，已经包含了softmax
    '''optimizer = torch.optim.SGD(net.parameters(), 0.001,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=True)'''
    optimizer = optim.Adam(net.parameters(),
                           lr=args.lr,
                           weight_decay=0.0001)
    print("Done")

    print("Training and test the model")

    epochs = args.epoch_num
    test_accs = []
    train_losses = []

    for epoch in tqdm(range(0, epochs)):
        loss = train(net, train_loader, epoch, optimizer, criterion)
        test_acc = test(net, test_loader)
        print("After traing {} epochs, loss is {}, test accuarcy is {}".format(epoch + 1, loss, test_acc))
        test_accs.append(test_acc)
        train_losses.append(loss)
    print("Done")
    '''
    net = torch.load('../models/net_ResNet-50.pth', map_location=torch.device('cpu'))
    '''
    print(f">>>>>>>Mean last_15  Test Accuracy:{sum(test_accs[-15:])/len(test_accs[-15:])}\n" )
    print("Ploting...")

    plot_fig(range(epochs), test_accs, "Test_acc")
    plot_fig(range(epochs), train_losses, "Train_loss")
    #epochs = 0
    #plot_roc_conf_matrix(net, epochs, test_loader)

    #print("Done")
    '''
    print("Saving the trained model")
    torch.save(net, '../models/net_50.pth')
    print("Done")
    '''

