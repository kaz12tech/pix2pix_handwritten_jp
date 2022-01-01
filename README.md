# 概要
https://github.com/ngduyanhece/cascade-gan  
上記をTensorflow 2.0以上で動作するように修正  
且つ、日本語のひらがな、カタカナ、半角英数字、常用漢字、人名漢字を学習するように修正

## setup
```
conda create -n pix2pix_handwritten_jp python=3.7 -y
conda activate pix2pix_handwritten_jp
cd pix2pix_handwritten_jp
git clone https://github.com/kaz12tech/pix2pix_handwritten_jp.git
cd pix2pix_handwritten_jp/
conda install -c anaconda tensorflow-gpu==2.2.0 pillow scikit-learn
```

## 入力画像生成
手書き風フォントを使用する場合はこちらからダウンロード下さい。  
https://ahito.com/item/desktop/font/mogihaPen/  

```
python3 text2img.py --font_path font/meiryo.ttc --font_size 92 --img_size 64 --out_dir ./font_img  
python3 text2img.py --font_path font/mogihaPen.ttf --font_size 92 --img_size 64 --out_dir ./font_img  
```

## 入力画像分割
生成した入力画像をランダムに学習用、テスト用に分割

```
python3 split_train_test.py --input_dirA ./font_img/meiryo --input_dirB ./font_img/mogihaPen
```

## データセット生成
```
python3 dataset_generator.py --path_a font_img/meiryo --path_b font_img/mogihaPen --img_size 64 --out_dir ./dataset --blur_size 4
```

## 学習実施
```
python3 pix2pix_2D_s.py --dataset ./dataset
```

## 手書き文字出力

```
python3 create_handwritten.py --text "この文字はAIが作った手書き文字です"
```