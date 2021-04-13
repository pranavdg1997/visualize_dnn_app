# Dash app for visualizing DNNs

Using this app, we visualize how neural networks act as feature extractors. Across layers and accross epochs of training, we compress the data in 2d/3d space using TSNe/PCA/other similar methods, then proceed to visualize them using plotly. We then color code each point and observe cluster formation, and there's also the option of looking up inidividual data point to note outliers. You can look at the demo below - 

<div align="center">
  <a href="https://www.youtube.com/watch?v=Gcj6ArmyZog"><img src="https://github.com/pranavdg1997/visualize_dnn_app/blob/main/screenshot.JPG" alt="IMAGE ALT TEXT"></a>
</div>


# Getting started
Clone the library, then download the dataset and input assests from [here](https://drive.google.com/file/d/1Q35O8_7aT5nsLCzyCyvkwQLe5mm6WES8/view?usp=sharing), and make sure the extract the files dirctly into the directory of the app (ie, make sure it does not create seprate folder but installs the files directly inside the directory). \
Proceed to install the dependency using - 
```sh
pip install -r requirements.txt
```
You can then simply run the app, and click on the URL in the terminal to get started.

```sh
python app.py
```
