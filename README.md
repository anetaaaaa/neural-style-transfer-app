# Neural Style Transfer App  

This Application implements the Neural Style Transfer algorithm as described in “A Neural Algorithm of Artistic
Style” article by L. A. Gatys, S. A. Ecker and M. Bethge. The App implements the optimizer from PyTorch https://pytorch.org/tutorials/advanced/neural_style_tutorial.html.

The App allows the user to choose one content image and style image. These images generate a new image, containing the content and style of previously chosen images. The user can also visualize the feature maps that are generated after convolution layers. 

Example App behavior:

<div align="center">
<img src="app_screen1.jpg" height="1000"/>
</div>

<div align="center">
<img src="app_screen2.jpg" height="1000"/>
</div>

<div align="center">
<img src="app_layers.jpg" height="1000"/>
</div>

## Install

* Clone this repository with this html: 
* Go to the root of cloned repository
* Install dependencies by running `pip3 install -r requirements.txt`

## Run

Execute:

```
python3 object_detector.py
```

It will start a webserver on http://localhost:5000. Use any web browser to open the web interface.