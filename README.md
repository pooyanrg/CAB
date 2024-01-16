# Official implementation of Co-Attention Bottleneck: Explainable and Causal Attention Emerged from Transformers Trained to Detect Images Changes

## Requirements

```sh
conda env create -f env.yml
conda activate cab
```


## Data Preparing

**For CLEVR-Change**

The official data can be found here: [google drive link](https://drive.google.com/file/d/1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe/view) provided by [Robust Change Captioning (ICCV19)](https://github.com/Seth-Park/RobustChangeCaptioning). 

Extracting this file will create data directory.

```sh
tar -xzvf clevr_change.tar.gz
```

For the convenience, you can also download the three json files from [link](https://drive.google.com/drive/folders/1g8QD6Y3La8cIamE7jeSSlXTw8G3r5Q8o?usp=sharing).

You would get

```
your_data_path
|–– clevr_change/
|   |–– data/
|   |   |–– images/
|   |   |–– nsc_images/
|   |   |–– sc_images/
|   |   |–– sc_images/
|   |   |–– change_captions.json
|   |   |–– no_change_captions.json
|   |   |–– splits.json
|   |   |–– type_mapping.json
```


## Pretrained Weight

```sh
cd ckpts
mkdir pretrained
mkdir trained
```

You can download the [Pretrained Weights](https://drive.google.com/drive/folders/1qOYVpZy57clJPF6AThsnO0Tfy4zq-gg1?usp=sharing) from the IDC Adaptation and the [Trained Weights](https://drive.google.com/drive/folders/18UfIvwKt0EE14EbogJycMmANpUJtsZbE?usp=sharing) from the IDC Finetuning. You would get

```
ckpts
|–– pretrained/
|   |–– pytorch_model.bin.clevr
|   |–– pytorch_model.bin.spot
|–– trained/
|   |–– pytorch_model.bin.clevr
|   |–– pytorch_model.bin.spot
```

Download CLIP (ViT-B/32) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

