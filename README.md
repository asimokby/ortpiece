# ORTPiece: An ORT-Based Turkish Image Captioning Network Based on Transformers and WordPiece

This is a PyTorch implementation of the [ORTPiece paper](https://scholar.google.com/scholar?oi=bibs&hl=en&cluster=4146675600184051868) accepted in SIU2023. This repository is largely based on code from the Object Relation Transformer paper which you can find [here](https://github.com/yahoo/object_relation_transformer).

The primary additions are as follows:
* WordPiece Tokenization
* Modified and parallel scripts specialized for Turkish

## Pretrained models

You can download our best model from [here](https://drive.google.com/file/d/1Vz9qJtONm86F4G1Rew8EPp0Rhj9hK4Wl/view?usp=sharing).


## Requirements
* Python 2.7 (because there is no [coco-caption](https://github.com/tylin/coco-caption) version for Python 3)
  - NOTE: You can work with Python 3, but just with 2.7 at evaluation with another simple environment!
* PyTorch 0.4+ (along with torchvision)
* h5py
* scikit-image
* typing
* pyemd
* gensim
* [cider](https://github.com/ruotianluo/cider.git) (already added as a submodule). See `.gitmodules` and clone the referenced repo into
  the `object_relation_transformer` folder.  
* The [coco-caption](https://github.com/tylin/coco-caption) library,
  which is used for generating different evaluation metrics. To set it
  up, clone the repo into the `object_relation_transformer`
  folder. Make sure to keep the cloned repo folder name as
  `coco-caption` and also to run the `get_stanford_models.sh`
  script from within that repo.

## Data Preparation

### Download ResNet101 weights for feature extraction

Download the file `resnet101.pth` from [here](https://drive.google.com/drive/folders/0B7fNdx_jAqhtbVYzOURMdDNHSGM). Copy the weights to a folder `imagenet_weights` within the data folder:

```
mkdir data/imagenet_weights
cp /path/to/downloaded/weights/resnet101.pth data/imagenet_weights
```

### Download and preprocess the COCO captions

Download the [preprocessed COCO Turkish captions](https://drive.google.com/file/d/17B_dJCo5zQspLZhFcD_PW9XzufQPBmmX/view?usp=sharing), which is parallel to the English version by Karpathy. Extract `dataset_cocoturk.json` from the zip file and copy it into `data/`. This file provides preprocessed captions and also standard train-val-test splits.

Then run:

```
$ python scripts/prepro_labels_wordpiece.py --input_json data/dataset_cocoturk.json --output_json data/cocotalk_piece_20_3.json --output_h5 data/cocotalk_piece_20_3
```

`prepro_labels_wordpiece.py` will map all words that occur <= 3 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk_piece_20_3.json` and discretized caption data are dumped into `data/cocotalk_piece_20_3.h5`.

Next run:
```
$ python scripts/prepro_ngrams_tr.py --input_json data/dataset_cocoturk.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

This will preprocess the dataset and get the cache for calculating cider score.


### Download the COCO dataset and pre-extract the image features

Download the [COCO images](http://mscoco.org/dataset/#download) from the MSCOCO website.
We need 2014 training images and 2014 validation images. You should put the `train2014/` and `val2014/` folders in the same directory, denoted as `$IMAGE_ROOT`:

```
mkdir $IMAGE_ROOT
pushd $IMAGE_ROOT
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
popd
wget https://msvocds.blob.core.windows.net/images/262993_z.jpg
mv 262993_z.jpg $IMAGE_ROOT/train2014/COCO_train2014_000000167126.jpg
```

The last two commands are needed to address an issue with a corrupted image in the MSCOCO dataset (see [here](https://github.com/karpathy/neuraltalk2/issues/4)). The prepro script will fail otherwise.


Then run:

```
$ python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root $IMAGE_ROOT
```

`prepro_feats.py` extracts the ResNet101 features (both fc feature and last conv feature) of each image. The features are saved in `data/cocotalk_fc` and `data/cocotalk_att`, and resulting files are about 200GB. Running this script may take a day or more, depending on hardware.

(Check the prepro scripts for more options, like other ResNet models or other attention sizes.)

### Download the Bottom-up features

Download the pre-extracted features from [here](https://github.com/peteanderson80/bottom-up-attention). For the paper, the adaptive features were used.

Do the following:
```
mkdir data/bu_data; cd data/bu_data
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip
unzip trainval.zip

```
The .zip file is around 22 GB.
Then return to the base directory and run:
```
python scripts/make_bu_data.py --output_dir data/cocobu
```

This will create `data/cocobu_fc`, `data/cocobu_att` and `data/cocobu_box`.


### Generate the relative bounding box coordinates for the Relation Transformer

Run the following:
```
python scripts/prepro_bbox_relative_coords.py --input_json data/dataset_coco.json --input_box_dir data/cocobu_box --output_dir data/cocobu_box_relative --image_root $IMAGE_ROOT
```
This should take a couple hours or so, depending on hardware.

## Model Training and Evaluation

### Standard cross-entropy loss training

```
python train.py --id relation_transformer_bu --caption_model relation_transformer --input_json data/cocotalk_piece_20_3.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --input_label_h5 data/cocotalk_piece_20_3_label.h5 --checkpoint_path wordpiece_20_3_log_relation_transformer_bu --noamopt --noamopt_warmup 10000 --label_smoothing 0.0 --batch_size 15 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --save_checkpoint_every 6000 --language_eval 0 --val_images_use 5000 --max_epochs 50 --use_box 1  
```

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `save/`). We only save the best-performing checkpoint on validation and the latest checkpoint to save disk space.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

If you have tensorflow, the loss histories are automatically dumped into `--checkpoint_path`, and can be visualized using tensorboard.

The current command uses scheduled sampling. You can also set scheduled_sampling_start to -1 to disable it.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to download the [coco-caption code](https://github.com/tylin/coco-caption) into `coco-caption` directory. NOTE: If you have decided to have a python 2.7 environment separate for coco-caption then you would have a problem here!

For more options, see `opts.py`.

### Evaluate on Karpathy's test split
To evaluate the cross-entropy model, run:

```
python eval.py --dump_images 0 --num_images 500 --model wordpiece_20_3_log_relation_transformer_bu/model-best.pth --infos_path wordpiece_20_3_log_relation_transformer_bu/infos_relation_transformer_bu-best.pkl --image_root data --input_json data/cocotalk_piece_20_3.json --input_label_h5 data/cocotalk_piece_20_3_label.h5  --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --language_eval 1
```


## Citation

    @inproceedings{ersoy2023ortpiece,
    title={ORTPiece: An ORT-based Turkish image captioning network based on transformers and WordPiece},
    author={Ersoy, Asim and Y{\i}ld{\i}z, Olcay Taner and {\"O}zer, Sedat},
    booktitle={2023 31st Signal Processing and Communications Applications Conference (SIU)},
    pages={1--4},
    year={2023},
    organization={IEEE}
    }


## Acknowledgments

Thanks to [Yahoo](https://github.com/yahoo/object_relation_transformer) and [Ruotian Luo](https://github.com/ruotianluo) for the original code.




