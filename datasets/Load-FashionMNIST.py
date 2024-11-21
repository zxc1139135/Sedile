import numpy as np
import struct
def load_images(file_name):
    # 在读取或写入一个文件之前，你必须使用 Python 内置open()函数来打开它
    # file object = open(file_name [, access_mode][, buffering])
    # file_name是包含您要访问的文件名的字符串值
    # access_mode指定该文件已被打开，即读，写，追加等方式
    # 0表示不使用缓冲，1表示在访问一个文件时进行缓冲
    # 这里rb表示只能以二进制读取的方式打开一个文件
    binfile = open(file_name, 'rb')
    # 从一个打开的文件读取数据
    buffers = binfile.read()
    # 读取image文件前4个整型数字
    magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)
    # 整个images数据大小为60000*28*28
    bits = num * rows * cols
    # 读取images数据
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    # 关闭文件
    binfile.close()
    # 转换为[60000,784]型数组
    images = np.reshape(images, [num, rows * cols])
    return images

def load_labels(file_name):
    # 打开文件
    binfile = open(file_name, 'rb')
    # 从一个打开的文件读取数据
    buffers = binfile.read()
    # 读取label文件前2个整形数字，label的长度为num
    magic, num = struct.unpack_from('>II', buffers, 0)
    # 读取labels数据
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    # 关闭文件
    binfile.close()
    # 转换为一维数组
    labels = np.reshape(labels, [num, 1])
    return labels

def get_MINIST_data():
    filename_train_images = './FashionMNIST/train-images-idx3-ubyte'
    filename_train_labels = './FashionMNIST/train-labels-idx1-ubyte'
    filename_test_images = './FashionMNIST/t10k-images-idx3-ubyte'
    filename_test_labels = './FashionMNIST/t10k-labels-idx1-ubyte'
    train_images = load_images(filename_train_images)
    train_labels = load_labels(filename_train_labels)
    test_images = load_images(filename_test_images)
    test_labels = load_labels(filename_test_labels)
    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = get_MINIST_data()
print("Dim: train_images, train_labels, = ", np.shape(train_images), np.shape(train_labels))
print("Dim: test_images, test_labels = ", np.shape(test_images), np.shape(test_labels))

m, d = np.shape(train_images)
X_train = train_images/np.max(train_images)
X_test = test_images/np.max(test_images)

y_test = (test_labels + 1) / 2
y_train = (train_labels + 1) / 2

X = X_train

m, d = X.shape
# reshape row vector into a column vector
y = np.reshape(y_train, (m, 1))
y_test = np.reshape(y_test, (len(y_test),1))

# release the memory
X_train = None
y_train = None





