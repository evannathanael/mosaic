---
language:
- en
license: cc-by-4.0
size_categories:
- 100K<n<1M
task_categories:
- text-to-image
- image-to-image
pretty_name: SID_Set
dataset_info:
  features:
  - name: img_id
    dtype: string
  - name: image
    dtype: image
  - name: mask
    dtype: image
  - name: width
    dtype: int64
  - name: height
    dtype: int64
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 124132341460.0
    num_examples: 210000
  - name: validation
    num_bytes: 16826718383.0
    num_examples: 30000
  download_size: 140056462172
  dataset_size: 140959059843.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
---

# Dataset Card for SID_Set

## Dataset Description
- **Project Page:** https://hzlsaber.github.io/projects/SIDA/
- **Repository:** https://github.com/hzlsaber/SIDA
- **Point of Contact:** [Zhenglin Huang](mailto:zhenglin@liverpool.ac.uk)

### Dataset Summary

We provide **Social media Image Detection dataSet (SID-Set)**, which offers three key advantages:

- **Extensive volume:** Featuring 300K AI-generated/tampered and authentic images with comprehensive annotations.
- **Broad diversity:** Encompassing fully synthetic and tampered images across various classes.
- **Elevated realism:** Including images that are predominantly indistinguishable from genuine ones through mere visual inspection.

Please check our [website](https://hzlsaber.github.io/projects/SIDA/) to explore more visual results.

#### Dataset Structure
- "img_id" (str): real image ids are same with [OpenImages V7](https://storage.googleapis.com/openimages/web/index.html). 


- "image" (str): there are three types of images, real images(from OpenImages V7), full_synthetic images, and tampered images.

- "mask" (str): Binary mask highlighting manipulated regions in tampered images


- "label" (int): Classification category:

  - 0: Real images
  - 1: Full synthetic images
  - 2: Tampered images

### Splits

- train: 210000 images.
- val: 30000 images.
- test: 60000 images(To prevent potential data leakage, please check our [repo](https://github.com/hzlsaber/SIDA) for information on obtaining the test set.)

### Licensing Information

This work is licensed under a Creative Commons Attribution 4.0 International License. Where this work incorporates material from the 
 [COCO](https://cocodataset.org/), 
[OpenImages V7](https://storage.googleapis.com/openimages/web/index.html), and [Flickr30k](https://arxiv.org/pdf/1505.04870). 
we will fully comply with the terms of these datasets' Creative Commons Attribution 4.0 International License, including providing appropriate attribution to the original creators and ensuring that the derived portions remain available under those terms.


## Citation Information

If you find this dataset useful, please consider citing our paper:

```
@misc{huang2025sidasocialmediaimage,
      title={SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model}, 
      author={Zhenglin Huang and Jinwei Hu and Xiangtai Li and Yiwei He and Xingyu Zhao and Bei Peng and Baoyuan Wu and Xiaowei Huang and Guangliang Cheng},
      year={2025},
      booktitle={Conference on Computer Vision and Pattern Recognition}
}
```