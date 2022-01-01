import argparse
import os
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

CHAR_LIST = "./jp_chars_v2.txt"

def text_to_img(char, font, img_size, dirname):
    w, h = font.getsize(char)
    
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), char, (0, 0, 0), font=font)
    
    img_array = np.array(img)
    centered_img_array = np.ones((128, 128, 3), dtype=np.uint8)*255
    h , w, d = np.where(img_array==0)
    height = max(h) - min(h)
    weight = max(w) - min(w)
    offset_h = (128 - height) //2
    offset_w = (128 - weight) //2
    centered_img_array[offset_h:offset_h + height + 1, offset_w:offset_w + weight + 1, :]=\
       img_array[min(h):max(h) + 1, min(w):max(w) + 1, :]
    centered_img = Image.fromarray(centered_img_array, 'RGB')

    centered_img = centered_img.resize((img_size, img_size), Image.ANTIALIAS).convert('L')
    centered_img = centered_img.resize((img_size, img_size), Image.ANTIALIAS)

    file_path = os.path.join( dirname, char + '.png' )
    centered_img.save(file_path)
    return file_path

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--font_path", default="font/meiryo.ttc")
    parser.add_argument("--font_size", default=92, type=int)
    parser.add_argument("--img_size", default=64, type=int)
    parser.add_argument("--out_dir", default="./font_img")

    args = parser.parse_args()

    return (args)

def main():
    args = get_args()

    font_path = args.font_path
    font_size = args.font_size
    font = ImageFont.truetype(font_path, font_size)
    font_name = os.path.splitext(os.path.basename(font_path))[0]
    
    img_size = args.img_size
    out_dir = args.out_dir
    dirname = os.path.join(out_dir, font_name)

    # ディレクトリ作成
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    # 作成する文字のリストを取得
    chars = []
    with open(CHAR_LIST) as f:
        for line in f:
            char = line.rstrip().split()[0]
            if not char in chars:
                chars.append( char )
    print('chars len:', len(chars))

    # 文字画像生成
    for char in chars:
        try:
            text_to_img(char, font, img_size, dirname)
        except Exception as e:
            print('Can not create img. char:', char)



# python3 text2img.py \
# --font_path font/meiryo.ttc \
# --font_size 92 \
# --img_size 64 \
# --out_dir ./font_img
if __name__ == '__main__':
    main()