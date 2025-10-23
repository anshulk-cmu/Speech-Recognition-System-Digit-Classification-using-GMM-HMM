import numpy as np

from model import DigitClassification


def label2class(label: str) -> int:
    if label=='z': return 0
    elif label=='o': return 10
    else: return int(label)

def class2label(class_id: int) -> str:
    if class_id==0: return 'z'
    elif class_id==10: return 'o'
    else: return str(class_id)


if __name__ == '__main__':

    state_size = 4
    component_size = 5
    feature_size = 39   # maximum 40
    vocab_size = 11     # don't change this; 1-9 digits + "z","o" for 0
    num_epoch = 5

    model = DigitClassification(state_size, component_size, feature_size, vocab_size, num_epoch)

    X_train = []
    y_train = []

    print("Loading data...")
    for line in open('train.txt', 'r'):
        filename, label = line.strip().split()
        feat = np.load('train/'+filename, allow_pickle=True)
        label = label2class(label)
        X_train.append(feat[:, :feature_size])
        y_train.append(label)

    X_test = []
    filename_lst = []

    X_test, y_ans = [], []
    for line in open('test.txt', 'r'):
        filename, _ = line.strip().split()
        feat = np.load('test/'+filename, allow_pickle=True)
        filename_lst.append(filename)
        X_test.append(feat[:, :feature_size])

    print("Start training!")
    model.fit(X_train, y_train)

    print("Finish training, predicting...")
    y_test = model.predict(X_test)

    output = open('test.txt', 'w')

    for filename, y in zip(filename_lst, y_test):
        output.write(f"{filename} {class2label(y)}\n")

    print("Done! Results saved to test.txt")
    output.close()
