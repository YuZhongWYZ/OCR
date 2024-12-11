#%%
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#%%

# Define PyTorch CNN model
class TorchCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        """
        PyTorch CNN 模型
        参数：
        - input_channels: 输入通道数，例如 MNIST 的灰度图像为 1
        - num_classes: 输出类别数，例如 MNIST 为 10 类
        """
        super(TorchCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # 卷积层，输出通道数 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化，减少尺寸
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 卷积层，输出通道数 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),  # 全连接层，输出维度 128
            nn.ReLU(),
            nn.Linear(128, num_classes)  # 全连接层，输出维度为类别数
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
#%%

# Define sklearn KNN model
def create_knn_model(n_neighbors=5):
    """
    sklearn KNN 模型
    参数：
    - n_neighbors: 邻居数量，默认 5
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors)

# Define sklearn SVM model
def create_svm_model(kernel='linear', C=1.0):
    """
    sklearn SVM 模型
    参数：
    - kernel: 核函数类型，例如 'linear', 'rbf' 等
    - C: 正则化参数，默认为 1.0
    """
    return SVC(kernel=kernel, C=C, probability=True)

#%%
#模型拓展，这里以MLP为例
class TorchMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TorchMLP, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)


#%%

# Wrapper to create models based on input
def get_model(model_type, **kwargs):
    """
    根据指定类型创建模型
    参数：
    - model_type: 模型类型，支持 'torch_cnn', 'knn', 'svm'
    - kwargs: 传递给具体模型的参数
    """
    if model_type == "torch_cnn":
        input_channels = kwargs.get("input_channels", 1)  # 输入通道数，默认 1
        num_classes = kwargs.get("num_classes", 10)  # 类别数，默认 10
        return TorchCNN(input_channels, num_classes)

    elif model_type == "knn":
        n_neighbors = kwargs.get("n_neighbors", 5)  # KNN 的邻居数量，默认 5
        return create_knn_model(n_neighbors)

    elif model_type == "svm":
        kernel = kwargs.get("kernel", 'linear')  # SVM 的核函数类型，默认 'linear'
        C = kwargs.get("C", 1.0)  # SVM 的正则化参数，默认 1.0
        return create_svm_model(kernel, C)

    #模型拓展，在上文添加模型后，在此处注册
    # elif model_type == "torch_mlp":
    #     input_size = kwargs.get("input_size", 28 * 28)
    #     hidden_size = kwargs.get("hidden_size", 128)
    #     num_classes = kwargs.get("num_classes", 36)
    #     return TorchMLP(input_size, hidden_size, num_classes)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
#%%
if __name__ == "__main__":
    # Example usage
    torch_model = get_model("torch_cnn", input_channels=1, num_classes=10)
    print(torch_model)

    knn_model = get_model("knn", n_neighbors=3)
    print(knn_model)

    svm_model = get_model("svm", kernel='rbf', C=2.0)
    print(svm_model)

    # mlp_model = get_model("torch_mlp", input_size=28 * 28, hidden_size=128, num_classes=36)
    # print(mlp_model)