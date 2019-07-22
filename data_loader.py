import os
import pickle

training_file = os.path.join("data", "train.p")
validation_file = os.path.join("data", "valid.p")
testing_file = os.path.join("data", "test.p")

with open(training_file, mode="rb") as f:
    train = pickle.load(f)
with open(validation_file, mode="rb") as f:
    valid = pickle.load(f)
with open(testing_file, mode="rb") as f:
    test = pickle.load(f)

x_train, y_train = train["features"], train["labels"]
x_validation, y_validation = valid["features"], valid["labels"]
x_test, y_test = valid["features"], valid["labels"]

if __name__ == '__main__':
    num_train = len(y_train)
    num_test = len(y_test)
    num_validation = len(y_validation)
    print("train samples:\t%d" % num_train)
    print("validation samples:\t%d" % num_validation)
    print("test  samples:\t%d" % num_test)
    print()
    print("shape of traffic sign image:", end="\t")
    print(x_train[0].shape)
    print("number of classes/labels:", end="\t")
    print(max(y_train) - min(y_train) + 1)
