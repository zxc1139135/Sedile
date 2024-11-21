from scipy import io
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

data1 = io.loadmat('SVHN/train.mat')
data2 = io.loadmat('SVHN/test.mat')
# data3 = io.loadmat('extra.mat')

train_data = data1['X']
train_labels = data1['y']
test_data = data2['X']
test_labels = data2['y']
# extra_data=data3['X']
# extra_labels=data3['y']

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
# extra_data=extra_data.astype('float32')

train_data = np.transpose(train_data, (3, 0, 1, 2))
test_data = np.transpose(test_data, (3, 0, 1, 2))

# extra_data=np.transpose(extra_data,(3,0,1,2))


train_labels[train_labels == 10] = 0

test_labels[test_labels == 10] = 0

# extra_labels[extra_labels==10]=0

classes = 10

train_labels = train_labels[:, 0]
test_labels = test_labels[:, 0]


# extra_labels=extra_labels[:,0]

def one_hot(label, classes):
    label = np.array(label).reshape(-1)
    label = np.eye(classes)[label]
    return label


train_labels = one_hot(train_labels, classes)
test_labels = one_hot(test_labels, classes)
# extra_labels=OneHot(extra_labels,classes)

print('Train data:', train_data.shape, ', Train labels:', train_labels.shape)
print('Test data:', test_data.shape, ', Test labels:', test_labels.shape)

train_data, validation_train, train_labels, validation_label_final = train_test_split(train_data, train_labels,
                                                                                      train_size=70000,
                                                                                      random_state=106)
print('Train data:', train_data.shape, ', Train labels:', train_labels.shape)
print('Validation data:', validation_train.shape, ', Validation labels:', validation_label_final.shape)
pickle_file = 'SVHN.pickle'

dict_to_pickle = {
    'train_dataset':
        {
            'X': train_data,
            'y': train_labels
        },
    'test_dataset':
        {
            'X': test_data,
            'y': test_labels
        },

    'valid_dataset':
        {
            'X': validation_train,
            'y': validation_label_final
        },

}

with open(pickle_file, 'wb') as f:
    print("saving Pickle.W8")
    pickle.dump(dict_to_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved")

filename = 'SVHN.pickle'
with open(filename, 'rb') as f:
    other = pickle.load(f)
    # print(other)
    train_data = other['train_dataset']
    test_data = other['test_dataset']
    # print(test_data)

    del other

train_dataset = train_data['X']
test_dataset = test_data['X']
# print(train_dataset)
# train_data = np.transpose(train_data, (3, 0, 1, 2))
train_labels = train_data['y']
test_labels = test_data['y']

print(len(train_dataset))
print(len(train_labels))
print(train_dataset.shape)
print(len(test_dataset))
print(len(test_labels))
print(test_dataset.shape)

X_train = np.reshape(train_dataset, (train_dataset.shape[0], -1))
print(X_train.shape)
y_train = np.reshape(train_labels, (train_labels.shape[0], -1))
print(y_train.shape)
X_test = np.reshape(test_dataset, (test_dataset.shape[0], -1))
print(X_test.shape)
y_test = np.reshape(test_labels, (test_labels.shape[0], -1))
print(y_test.shape)
