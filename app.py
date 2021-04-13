# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:28:22 2020

@author: Pranav
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.express as px
from dash_canvas.utils import array_to_data_url
import pickle
import time
import cv2
from skimage import img_as_ubyte

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
root_folder = os.getcwd()
print("ROOT FOLDER:{}".format(root_folder))
embeddings_dir_layer = os.path.join(root_folder,"embeddings_layer/")
embeddings_dir_epoch = os.path.join(root_folder,"embeddings_epoch/")
html_dir = os.path.join(root_folder,"htmls")


desc_dict = pickle.load(open(os.path.join(root_folder,"desc_dict.pkl"),"rb"))
sample_image = cv2.imread(os.path.join(root_folder,"sample_image.jpg"))

files = [file for file in os.listdir(embeddings_dir_layer) if "csv" in file]
file_df = pd.DataFrame()
file_df["files"] = files
file_df["dataset"] = file_df["files"].apply(lambda x:x.split("_")[0])
file_df["method"] = file_df["files"].apply(lambda x:x.split("_")[1])
method_options = list(file_df["method"].unique())
method_options.append("all")
dataset_options = file_df["dataset"].unique()

print("Files found:{}".format(files))

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True




def serialize(arr):
    return([{'label': i, 'value': i} for i in arr])
    
def scale_vals(df_all,cols,prog_col):
    df_new = pd.DataFrame()
    for prog in df_all[prog_col].unique():
        df_l = df_all.loc[df_all[prog_col]==prog,:]
        df_l[cols] = MinMaxScaler(feature_range=[0,1]).fit_transform(df_l[cols])
        df_new = pd.concat((df_new,df_l),axis=0)
    return(df_new)
    
def combine(x):
    return("serial_number:{}|label:{}".format(x[0],x[1]))
    
    
    

    
def get_viz_children(dataset,method,dimension,prog_type):
    print(dataset)
    print(method)
    
    heading = html.Div([html.H4("Visualization of {} dataset by using {} method.".format(dataset,method)),
                        html.H6(desc_dict[method])])
    csv_name = "_".join([dataset,method,dimension]) + ".csv"
    if(prog_type=="layer-wise"):
        df_all = pd.read_csv(os.path.join(embeddings_dir_layer,csv_name))
        prog_col = "layer"
        plot_title = "Progression of visuals accross layers"
    else:
        df_all = pd.read_csv(os.path.join(embeddings_dir_epoch,csv_name))
        prog_col = "epoch"
        plot_title = "Progression of visuals through epochs of training."
    print(df_all[prog_col].unique())
    df_all[prog_col] = pd.Categorical(df_all[prog_col],categories=df_all[prog_col].unique())
    
    #df_all["serial_number"] = np.arange(0,len(df_all))
    if(dataset=="FashionMNIST"):
        labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}
        df_all["label"] = df_all["label"].map(labels_map)
    elif(dataset=="CIFAR10"):
        labels_map = {0 : 'Airplane', 1 : 'Automobile (car)', 2 : 'Bird', 3 : 'Cat', 4 : 'Deer', 5 : 'Dog', 6 : 'Frog',
              7 : 'Horse', 8 : 'Ship', 9 : 'Truck'}
        df_all["label"] = df_all["label"].map(labels_map)
    df_all["label"] = df_all["label"].astype(np.str)
    df_all = df_all.sort_values(by=[prog_col,"label"],ascending=True)
    cols = [col for col in df_all.columns if "dm" in col]
#    print(df_all[cols].describe())
#    print(df_all[cols].skew())
    df_all = scale_vals(df_all,cols,prog_col)
#    print(df_all[cols].describe())
#    print(df_all[cols].skew())
    if(dimension=="3d"):
        fig = px.scatter_3d(df_all,x='dm1', y='dm2',z="dm3",
                color='label',title=plot_title,
                range_x=[0.0,1.1],range_y=[0.0,1.1],range_z=[0.0,1.1],
          animation_group="label",animation_frame=prog_col,height=800,hover_data=["serial_number","label"])
    else:
        fig = px.scatter(df_all,x='dm1', y='dm2',
        color='label',title=plot_title,hover_data=["serial_number","label"],
        range_x=[0.0,1.1],range_y=[0.0,1.1],animation_frame=prog_col,height=800)
    graph = dcc.Graph(id='network-graphic',figure=fig)
    #lv_id = dataset+"-"+method+"-lookup-value"
    lv = dcc.Input(id={"type":"lookup_val","dataset":dataset,"method":method},type="text",style={
                        'width': '99%',
                        'height': '30px',
                        'lineHeight': '30px',
                        'borderWidth': '2px',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },placeholder="Enter serial number..")
    lb = html.Button(id={"type":"lookup_button","dataset":dataset,"method":method},
                     children='FETCH',
                    style={
                        'width': '99%',
                        'height': '30px',
                        'lineHeight': '30px',
                        'borderWidth': '2px',
                        'borderRadius': '5px',
                        'textAlign': 'top',
                        'margin': '10px'
                    })
    #lk_id = dataset+"-"+method+"-lookup-img"
    lk = dcc.Loading([html.Img(src=array_to_data_url(img_as_ubyte(sample_image)),style={"width":"300px","height":"300px"})],
                      id={"type":"lookup_img","dataset":dataset,"method":method},type="default")
    child2 = html.Div([
            html.Div([graph],style={'width': '68%','height':'500px','display': 'inline-block'}),
            html.Div([html.H3("Lookup"),lv,lb,lk],
                      style={'width': '28%','float':'top','display': 'inline-block'})
            ])
    
    return(html.Div([heading,child2],style={'height':'1000px'}))
    
    


app.layout = html.Div([
    html.Div(
        [
            html.H3(
                "Dimension reduction visualization",
                style={"margin-bottom": "0px"},
            ),
            html.H5(
                "Information visualization project", style={"margin-top": "0px"}
            ),
        ]
    ),
    html.Div([

        html.Div([
            html.P("1.Select Dataset:",
            ),
            dcc.Dropdown(
                id='dataset',
                options=serialize(list(dataset_options))
            ),
            html.P("2.Method for dimensionality reduction :",
            ),
            dcc.Dropdown(
                id='method',
                options=serialize(list(method_options))
            ),
            html.P("3. Dimension:",
            ),
            dcc.Dropdown(
                id='dimension',
                options = serialize(["2d","3d"])
            ),
            html.P("4. Progression:(Layer wise/epoch wise):",
            ),
            dcc.Dropdown(
                id='prog-type',
                options = serialize(["layer-wise","epoch-wise"])
            )
                    
        ],
        style={'width': '98%', 'display': 'inline-block'})]),
    dcc.Loading(
            [html.H6("Select settings to update figure")],
            id="loading-graph",
            type="default"
        )
    ])


    
### Updating main graph   
@app.callback(
    Output('loading-graph', 'children'),
    [Input('dataset', 'value'),
     Input('method', 'value'),
     Input('dimension', 'value'),
     Input('prog-type', 'value')])
def update_graph(dataset,method,dimension,prog_type):
    print(method)
    #time.sleep(1)
    print()
    try:
        if(method!="all"):
            return(get_viz_children(dataset,method,dimension,prog_type))
        else:
            return(html.Div([get_viz_children(dataset,m,dimension,prog_type) for m in method_options[0:-1]]))
    except Exception as e:
        print("Figure update failed due to the following error:")
        print(e)
        error_heading = html.Div([html.H6("Embedding for this configuration not found")])
        return([error_heading])

@app.callback(
    Output({'type': 'lookup_img', 'dataset': MATCH,'method':MATCH}, 'children'),
    [Input({'type': 'lookup_button', 'dataset': MATCH,'method':MATCH}, 'n_clicks')],
    [State({'type': 'lookup_val','dataset': MATCH,'method':MATCH}, 'value'),
     State({'type': 'lookup_val','dataset': MATCH,'method':MATCH}, 'id'),
     State('prog-type', 'value')]
)
def lookup_callback(n_clicks,lookup_val,id_dict,prog_type):
    #time.sleep(1000)
    try:
        print("entered callback")
        print(lookup_val)
        lookup_val = int(lookup_val)
        print(id_dict)
        dataset = id_dict["dataset"]
        #method = id_dict["dataset"]
        pkl = dataset + ".pkl"
        if(prog_type=="layer-wise"):
            embeddings_dir = embeddings_dir_layer
        else:
            embeddings_dir = embeddings_dir_epoch
        img = pickle.load(open(os.path.join(embeddings_dir,pkl),"rb"))[lookup_val].astype(np.uint8)
        print(img.shape)
        child = html.Img(src=array_to_data_url(img_as_ubyte(img)),style={"width":"300px","height":"300px"})
        return([child])
    except Exception as e:
        print(e)
        raise PreventUpdate       
              
if __name__ == '__main__':
    app.run_server(debug=True)
    


    