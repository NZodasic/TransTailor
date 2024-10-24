import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16
import torch.nn as nn
import torch.optim as optim

# Kiểm tra và sử dụng GPU nếu có
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hàm để tải dữ liệu CIFAR-10
def load_cifar10():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
    return trainloader

# Hàm để tải mô hình VGG16 và in thông tin
def load_vgg16():
    model = vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 10)  # Thay đổi lớp cuối cùng cho CIFAR-10
    model = model.to(device)
    print(model)
    return model

# Hàm để prune một filter
def prune_filter(model, layer_index, filter_index):
    conv_layer = model.features[layer_index]
    new_out_channels = conv_layer.out_channels - 1
    new_conv = nn.Conv2d(conv_layer.in_channels, new_out_channels,
                         conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, conv_layer.dilation, conv_layer.groups, bias=conv_layer.bias is not None)
    
    # Sao chép trọng số, ngoại trừ filter bị prune
    new_weights = conv_layer.weight.data[torch.arange(conv_layer.out_channels) != filter_index]
    new_conv.weight.data = new_weights
    
    if conv_layer.bias is not None:
        new_bias = conv_layer.bias.data[torch.arange(conv_layer.out_channels) != filter_index]
        new_conv.bias.data = new_bias
    
    model.features[layer_index] = new_conv
    return model

# Hàm để tái cấu trúc mô hình
def restructure_model(model, layer_index, filter_index):
    for i in range(layer_index + 1, len(model.features)):
        if isinstance(model.features[i], nn.Conv2d):
            conv_layer = model.features[i]
            new_in_channels = conv_layer.in_channels - 1
            new_conv = nn.Conv2d(new_in_channels, conv_layer.out_channels,
                                 conv_layer.kernel_size, conv_layer.stride,
                                 conv_layer.padding, conv_layer.dilation,
                                 conv_layer.groups, bias=conv_layer.bias is not None)
            
            # Sao chép trọng số, ngoại trừ kênh đầu vào bị loại bỏ
            new_weights = conv_layer.weight.data[:, torch.arange(conv_layer.in_channels) != filter_index]
            new_conv.weight.data = new_weights
            
            if conv_layer.bias is not None:
                new_conv.bias.data = conv_layer.bias.data
            
            model.features[i] = new_conv
        elif isinstance(model.features[i], nn.BatchNorm2d):
            bn_layer = model.features[i]
            new_bn = nn.BatchNorm2d(bn_layer.num_features - 1)
            
            # Sao chép các tham số, ngoại trừ feature bị loại bỏ
            new_bn.weight.data = bn_layer.weight.data[torch.arange(bn_layer.num_features) != filter_index]
            new_bn.bias.data = bn_layer.bias.data[torch.arange(bn_layer.num_features) != filter_index]
            new_bn.running_mean = bn_layer.running_mean[torch.arange(bn_layer.num_features) != filter_index]
            new_bn.running_var = bn_layer.running_var[torch.arange(bn_layer.num_features) != filter_index]
            
            model.features[i] = new_bn
    
    return model

# Hàm để huấn luyện mô hình
def train_model(model, trainloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    print('Finished Training')

# Hàm tổng quát để prune một filter bất kỳ và tái cấu trúc mô hình
def prune_and_restructure(model, layer_index, filter_index):
    model = prune_filter(model, layer_index, filter_index)
    model = restructure_model(model, layer_index, filter_index)
    return model

# Thực hiện các nhiệm vụ
trainloader = load_cifar10()
model = load_vgg16()

print("Original model structure:")
print(model)

# Prune một filter và tái cấu trúc ngay lập tức
layer_index = 0  # Ví dụ: prune filter đầu tiên của lớp conv đầu tiên
filter_index = 0
model = prune_and_restructure(model, layer_index, filter_index)

print("\nModel structure after pruning and restructuring:")
print(model)

# Huấn luyện mô hình sau khi prune và tái cấu trúc
train_model(model, trainloader, epochs=1)

# Sử dụng hàm tổng quát để prune và tái cấu trúc một lần nữa
layer_index = 2  # Ví dụ: prune filter của lớp conv thứ 3
filter_index = 1
model = prune_and_restructure(model, layer_index, filter_index)

print("\nModel structure after second pruning and restructuring:")
print(model)

# Huấn luyện mô hình cuối cùng
train_model(model, trainloader, epochs=1)