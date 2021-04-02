import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2
import rad_target_detection as rtd
import tkinter as tk
from tkinter.filedialog import askopenfilenames
import os
import dash  # (version 1.12.0) pip install dash
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import glob
import base64
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Create server variable with Flask server object mmor use with gunicorn
# server = app.server

# encoding image
image_filename = 'original.png'
encoded_image_original = base64.b64encode(open(image_filename, 'rb').read())
image_filename = 'binary.png'
encoded_image_binary = base64.b64encode(open(image_filename, 'rb').read())
image_filename = 'contours.png'
encoded_image_contours = base64.b64encode(open(image_filename, 'rb').read())
image_filename = 'targets.png'
encoded_image_targets = base64.b64encode(open(image_filename, 'rb').read())

# App layout

app.layout = html.Div([

    
    html.H1("RAD Targets Detector", style={'text-align': 'center'}),
    html.H2("Adjust parameters to get clear targets", style={'text-align': 'center'}),

    html.Div(id='slider-d-output', style={'margin-top': 20}),
    dcc.Slider(
        id='d-value',
        min=0,
        max=25,
        step=1,
        marks={
            1: '1',
            5: '5',
            10: '10',
            15: '15',
            20: '20',
            25: '25'
        },
        value=10,
    ),
    
    html.Div(id='slider-sigma-output', style={'margin-top': 20}),
    dcc.Slider(
        id='sigma',
        min=0,
        max=250,
        step=10,
        marks={
            0: '0',
            50: '50',
            100: '100',
            150: '150',
            200: '200',
            250: '250'
        },
        value=100,
    ),
    
    html.Div(id='slider-b_size-output', style={'margin-top': 20}),
    dcc.Slider(
        id='b_size',
        min=3,
        max=31,
        step=2,
        marks={
            0: '0',
            5: '5',
            10: '10',
            15: '15',
            20: '20',
            25: '25',
            30: '30'
        },
        value=15,
    ),
    
    # html.Div(
    # dcc.Graph(id = 'img', figure= fig)
    # ),
    html.Div([

        html.Div([
        html.Img(id= 'original_image',src='data:image/png;base64,{}'.format(encoded_image_original.decode()),style={'width': '100%', 'height': '100%'}),
        ],style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
        html.Img(id= 'binary_image',src='data:image/png;base64,{}'.format(encoded_image_binary.decode()),style={'width': '100%', 'height': '100%'})
        ],style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ]),

    html.Div(html.Button('Find Rad Targets', id='submit-val', n_clicks=0,style={'horizontalAlign' : 'center', 'display': 'inline-block'})),
    dcc.Loading(id = 'loading',
    children = [html.Div(id='container-button-basic',
                children='After adjusting binarization press to find targets', style={'horizontalAlign' : 'center', 'display': 'inline-block'})]),

    html.Div([

        # html.Div([
        # html.Img(id= 'contours_image',src='data:image/png;base64,{}'.format(encoded_image_contours.decode()),style={'width': '100%', 'height': '100%'}),
        # ],style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
        html.Img(id= 'targets_image',src='data:image/png;base64,{}'.format(encoded_image_targets.decode()),style={'width': '100%', 'height': '100%'})
        ],style={ 'float': 'center', 'display': 'inline-block'})
    ])
        

])

@app.callback(Output('slider-d-output', 'children'),
              Input('d-value', 'drag_value'))
def display_value(drag_value):
    return f'd_value: {drag_value}'

@app.callback(Output('slider-sigma-output', 'children'),
              Input('sigma', 'drag_value'))
def display_value(drag_value):
    return f'sigma_value: {drag_value}'

@app.callback(Output('slider-b_size-output', 'children'),
            Input('b_size', 'drag_value'))
def display_value(drag_value):
    return f'b_size: {drag_value}'

@app.callback(
    Output('binary_image', 'src'),
    [
    Input('d-value', 'value'),
    Input('sigma', 'value'),
    Input('b_size', 'value')
    ]

)
def update_image(d, sigma, b):
    d = int(d)
    sigma = int(sigma)
    b = int(b)
    if b%2 == 0:
        b = b+1
    original_image = cv2.imread('original.png')
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  # gray image
    rgb_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # rgb image
    
    # binary image
    binary_img = rtd.binarize_image(gray,d=d, sig1=sigma, sig2=sigma, b_size=b, c=5)
    plt.imsave('binary.png', binary_img, cmap='gray')
    image_filename = 'binary.png'
    encoded_image_binary = base64.b64encode(open(image_filename, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image_binary.decode())

# @app.callback(
#     Output('container-button-basic', 'children'),
#     Input('submit-val', 'n_clicks'))
# def running_indication(n_clicks):
#     if n_clicks > 0:
#         return 'Running...'
#     return 'After adjusting binarization press to find targets (it might take some time...)'



@app.callback(
    [
    Output('targets_image', 'src'),
    Output('container-button-basic', 'children')
    ],
    [
    Input('submit-val', 'n_clicks')
    ]
    # dash.dependencies.Output('binary_image', 'src'),
    # [dash.dependencies.Input('original_image', 'src'),
    # dash.dependencies.Input('d-value', 'value')]
)
def update_targets_image(n_clicks):

    if n_clicks > 0:
        original_image = cv2.imread('original.png')
        rgb_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # rgb image

        # contours image
        binary_img = cv2.imread('binary.png')
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
        contour_thresh=10
        contours = rtd.contour_image(binary_img, contour_thresh)
        c_img = rgb_img.copy()
        cv2.drawContours(c_img, contours, -1, (255, 0, 0), 2)
        plt.imsave('contours.png', c_img)


        # targets image
        ellipses, hulls = rtd.find_ellipses(contours)
        rad_targets = rtd.find_rad_targets(ellipses, lower_thresh=3.5, upper_thresh=7.5)
        # coding each target by it's shape
        targets_df = rtd.targets_encoding(binary_img, rad_targets)
        # drawing found targets on img
        t_img = rgb_img.copy()
        rtd.draw_targets(t_img, targets_df)
        plt.imsave('targets.png', t_img)
        image_filename = 'targets.png'
        encoded_image_targets = base64.b64encode(open(image_filename, 'rb').read())
        return 'data:image/png;base64,{}'.format(encoded_image_targets.decode()), f'{targets_df.shape[0]} targets found' 
    image_filename = 'targets.png'
    encoded_image_targets = base64.b64encode(open(image_filename, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image_targets.decode()), 'After adjusting binarization press to find targets' 

# app.run_server(mode="inline")
if __name__ == '__main__':
    app.run_server(debug=True)
