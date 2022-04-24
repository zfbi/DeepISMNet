# DeepSIMNet: using synthetic datasets to train an end-to-end Convolutional Neural Network for implicit structural modeling
Three-Dimensional Implicit Structural Modeling Using Convolutional Neural Network

**This is a [Pytorch](https://pytorch.org/) version of a deep learning method using a convolution neural network to predict a scalar field from sparse structural data associated with multiple distinct stratigraphic layers and faults.  

As described in **Three-Dimensional Implicit Structural Modeling Using Convolutional Neural Network** by Zhengfa Bi<sup>1</sup>, 
Xinming Wu<sup>1</sup>, 
Zhaoliang Li<sup>2</sup>, 
Dekuan Change<sup>3</sup> and
Xueshan Yong<sup>3</sup>.
<sup>1</sup>University of Science and Technology of China; <sup>2</sup>China Aero Geophysical Survey and Remote Sensing Center for Natural Resources; <sup>3</sup>Research Institute of Petroleum Exploration \& Development-NorthWest(NWGI), PetroChina.

## Requirments

```
python>=3.6
torch>=1.0.0
torchvision
torchsummary
natsort
numpy
pillow
plotly
pyparsing
scipy
scikit-image
sklearn
tqdm
```
Install all dependent libraries:
```bash
pip install -r requirements.txt
```
## Dataset

**To train our CNN network, we automatically created numerous structrual models and the associated data with distinct stratigraphic layers and faults, which were shown to be sufficient to train an excellent structural modeling network.** 

**The synthetic structural models can be downloaded from [here](https://doi.org/10.5281/zenodo.4536561), while the input data are randomly generated in training the CNN.**

## Training

Run train.ipynb to start training a new DeepISMNet model by using the synthetic dataset.

## License

This extension to the Pytorch library is released under a creative commons license which allows for personal and research use only. 
For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
