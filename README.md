# Image Captioning using Reformulations

The code in this repository is based on the original [mPLUG repository](https://github.com/alibaba/AliceMind/tree/main/mPLUG).

## Introduction

<img src="pipeline.jpg" width="400" height="400">

We propose an inference-time feedback model for the task of image captioning with a novel type of feedback, namely reformulation.
In the context of this type of feedback, the human annotators receive an image and a textual description as input, and subsequently produce text that is as similar as possible to the input text but also incorporates an additional desired attribute (e.g., improved factuality or a desired style). The reformulation model is trained to predict human reformulations.

## Installation
```
pip install -r requirements.txt
```

## Training
If you want to train your own reformulation model:
1. Download the mPLUG model from the original [repo](https://github.com/alibaba/AliceMind/tree/main/mPLUG).
2. Prepare a reformulation dataset in json format, i.e., a list of dictionaries, each with the following fields:
   - question_id: Any unique id
   - image: Path to image
   - question: Original caption
   - Answer: Reformulated caption
3. Modify the 'train_file' field in the relevant config file (configs/re_mplug_base.yaml or configs/re_mplug_large.yaml).
4. Train, e.g.:
```
bash scripts/re_mplug_base.sh 
```
Or:
```
bash scripts/re_mplug_large.sh 
```

We provide 3 reformulation datasets in this repository:
1. An error-correction reformulation data containing 5208 collected using Amazon Mechanical Turk (data/error_correction_dataset.json).
2. A style-transfer reformulation data transfering to humorous captions, harvested from FlickrStyle and the original Flickr30K captions (data/humor_train.json and data/humor_test.json).
3. A style-transfer reformulation data transfering to romantic captions, harvested from FlickrStyle and the original Flickr30K captions (data/romantic_train.json and data/romantic_test.json).

## Reformulate
To reformulate existing captions, create a json file containing a list of dictionaries, each with the following fields:
- caption: The caption to be reformulated
- image_path: Path to relevant image

Train your own reformulation model or download one of our provided models:

- [Error correction reformulation model](https://drive.google.com/drive/folders/1POjbnc7f3fHtve3y8wqQQvd-hQ-DwHhA?usp=sharing) (mPLUG base)
- [Humor style-transfer reformulation model](https://drive.google.com/file/d/1Un85hb6mdCjMA6cilfXUcwtaf29uyf25/view?usp=sharing) (mPLUG large)
- [Romantic style-transfer reformulation model](https://drive.google.com/file/d/1TThQIYb0G8PFut-fYGRV2WUkmfzfgjGd/view?usp=sharing) (mPLUG large)

Then, use the reformulate.py script.:
```
python reformulate.py --model_path <path to reformulation model> --mplug_backbone <base/large> --input_file <path to input json file> --output_file <path to output file, json format>
```

## German Image Captioning

The code in this section is based on the [ClipCap repo](https://github.com/rmokady/CLIP_prefix_caption).

We provide a model for German image captioning, trained on the Multi30k dataset, and a pipeline for cross-lingual reformulation: generate a caption in German, translate to English, reformulate, translate back to German.

First, download our [German captioning model](https://drive.google.com/file/d/1LBCapDMsyRimYdkzHyAwRveqzKEGG2d2/view?usp=sharing).

### Generate German Captions

Use the german_captioning/inference.py script for generating German captions, e.g.
```
python german_captioning/inference.py --model_path <path to german model> --input_file <path to input file> --output_file <path to output file>
```
Where the input file is a list of dictionaries containing a single entry, 'image_path'.

If you want to use the reformulation pipeline, add the --reformulation_model flag with a path to the reformulation model, e.g.
```
python german_captioning/inference.py --model_path <path to german model> --input_file <path to input file> --output_file <path to output file> --reformulation_model error_correction_base.pth
```
Or
```
python german_captioning/inference.py --model_path <path to german model> --input_file <path to input file> --output_file <path to output file> --reformulation_model error_correction_large.pth --mplug_backbone large
```
This pipeline uses the [MarianMT open source translation models](https://huggingface.co/docs/transformers/model_doc/marian). If you prefer to use another translation API, you will need to generate the captions using the german_captioning/inference.py script, translate to English, reformulate using the reformulate.py script, and translate back to German.
