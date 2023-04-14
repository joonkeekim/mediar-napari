# MEDIAR napari plugin

A napari plugin for [MEDIAR: Harmony of Data-Centric and Model-Centric for Multi-Modality Microscopy](https://arxiv.org/abs/2212.03465)


----------------------------------

This [napari](https://github.com/napari/napari) plugin was generated with [Cookiecutter](https://github.com/audreyr/cookiecutter) using [@napari]'s [cookiecutter-napari-plugin](https://github.com/napari/cookiecutter-napari-plugin) template. Most of the UI design and implemenation is following the codes of [cellpose-napari](https://github.com/MouseLand/cellpose-napari/) [cellseg_sribd_napari](https://github.com/Lewislou/cellseg_sribd_napari).

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

```shell
conda create -y -n mediar-napari -c conda-forge python=3.8
conda activate mediar-napari
pip install "napari[all]"
cd mediar-napari
pip install -r requirements.txt
pip install -e .
```
## Weights

You can download our trained model `main_model.pth` and `sub_model.pth` at [google drive](https://drive.google.com/drive/folders/1nDNtnnx3itkfe6_pLEiuoKz9i3hCtLjF?hl=ko)

Then, locate these files as below

```
  mediar-napari
  ├── imgs
  ├── segmentation_models_pytorch
  ├── src
  ├── weights
  │   ├── main_model.pth
  │   └── sub_model.pth
  └── ...
```

We will soon modify the code for using our weights easily

## Running the software

```shell
napari -w mediar
```



## Source codes and training
The source codes of cellseg-sribd model and the training process are in [cellseg-sribd](https://github.com/Lewislou/cellseg-sribd/).

## License

Distributed under the terms of the [BSD-3](http://opensource.org/licenses/BSD-3-Clause) license,"cellseg-sribd" is free and open source software

