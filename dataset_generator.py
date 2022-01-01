import os
import argparse
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageFilter

TRAIN_LIST = 'train.txt'
TEST_LIST = 'test.txt'

def data_generate(dirname, datalist, args):
    img_size = args.img_size
    path_a = args.path_a
    path_b = args.path_b
    blur_size = args.blur_size

    for filename in datalist:
        if os.path.exists( os.path.join(path_a, filename) ) and \
            os.path.exists( os.path.join(path_b, filename) ):

            im_a = Image.open( os.path.join(path_a, filename) )
            im_b = Image.open( os.path.join(path_b, filename) )
            # blur画像生成
            im_a_blur = im_a.filter(ImageFilter.GaussianBlur(blur_size))
            im_b_blur = im_b.filter(ImageFilter.GaussianBlur(blur_size-2))
            # 横方向に連結
            new_im = Image.new('RGB', (img_size*3, img_size))
            new_im.paste(im_b, (0,0))
            new_im.paste(im_a_blur, (img_size,0))
            new_im.paste(im_b_blur, (img_size*2,0))
            # 保存
            new_im.save( os.path.join(dirname, filename) )

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path_a", default="font_img/meiryo")
    parser.add_argument("--path_b", default="font_img/mogihaPen")
    parser.add_argument("--img_size", default=64, type=int)
    parser.add_argument("--out_dir", default="./dataset")
    parser.add_argument("--blur_size", default=4, type=int)
    parser.add_argument("--crop_size", default=8, type=int)

    args = parser.parse_args()

    return (args)

def main():
    args = get_args()

    # ディレクトリ作成
    train_dir = os.path.join(args.out_dir, 'train')
    test_dir = os.path.join(args.out_dir, 'test')
    train_clear = os.path.join(args.out_dir, 'train_clear')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_clear, exist_ok=True)

    train, test = [], []
    with open(TRAIN_LIST, 'r') as f:
        train = [line.rstrip() for line in f]
    with open(TEST_LIST, 'r') as f:
        test = [line.rstrip() for line in f]

    data_generate(train_dir, train, args)
    data_generate(test_dir, test, args)
    args.blur_size = args.blur_size - 2
    data_generate(train_clear, train, args)

# python3 dataset_generator.py \
# --path_a font_img/meiryo \
# --path_b font_img/mogihaPen \
# --img_size 64 \
# --out_dir ./dataset \
# --blur_size 4
if __name__ == '__main__':
    main()