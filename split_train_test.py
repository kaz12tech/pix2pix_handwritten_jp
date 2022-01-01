import argparse
import os
import glob

from sklearn.model_selection import train_test_split

TRAIN_LIST = 'train.txt'
TEST_LIST = 'test.txt'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dirA", default="./font_img/meiryo")
    parser.add_argument("--input_dirB", default="./font_img/mogihaPen")

    args = parser.parse_args()

    return (args)

def main():
    args = get_args()
    input_dirA = args.input_dirA
    input_dirB = args.input_dirB
    imgsA = glob.glob(os.path.join(input_dirA, '*.png'))
    imgsB = glob.glob(os.path.join(input_dirB, '*.png'))

    datalist = []
    for imgA in imgsA:
        isMatch = False
        for imgB in imgsB:
            if os.path.basename(imgA) in imgB:
                isMatch = True
        if isMatch:
            datalist.append(os.path.basename(imgA))

    print('datalist len:', len(datalist))
    
    train, test = train_test_split(datalist, test_size=0.2)
    print('train len:', len(train), 'test len:', len(test))

    with open(TRAIN_LIST, 'w') as f:
        for file_name in train:
            f.write(file_name + '\n')

    with open(TEST_LIST, 'w') as f:
        for file_name in test:
            f.write(file_name + '\n')


# python3 split_train_test.py \
# --input_dirA ./font_img/meiryo \
# --input_dirB ./font_img/mogihaPen
if __name__ == '__main__':
    main()