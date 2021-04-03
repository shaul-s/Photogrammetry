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
from dash.dependencies import Input, Output, State
import base64
import plotly.express as px
from io import BytesIO
from PIL import Image

#----------------------------------------------------------------------------
# Target Detector

#  This is a tool we have build to analize our algorithm, we have fount it quite usefull.  
#  It can also be used to find the rad targets in a semi-automatic way and get goot results.

#  Instructions:  
#  1. Run the entire notebook, the tool will apear in the bottom.  
#  2. Adjust the sliders to get binary image where the targets can be seen as clearly as possible.  
#  3. Press on the find targets button and wait for the result. the process can take up to 5 minutes.  
#  4. After seeing the results try further adjusting the sliders to get better results
#----------------------------------------------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# encoding image
image_filename = 'table_targets\\20210325_121519.jpg'
# image_filename = 'original.png'
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

    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select an Image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-image-upload'),

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
                children='After adjusting binarization press to find targets (it might take few minutes)', style={'horizontalAlign' : 'center', 'display': 'inline-block'})]),

    html.Div([

        html.Div([
        html.Img(id= 'targets_image',src='data:image/png;base64,{}'.format(encoded_image_targets.decode()),style={'width': '100%', 'height': '100%'})
        ],style={ 'float': 'center', 'display': 'inline-block'})
    ])
        

])


def stringToImage(content):
    encoded_image = content.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    bytes_image = BytesIO(decoded_image)
    return np.array(Image.open(bytes_image).convert('RGB'))

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(content, filename, date):
    if content is not None:
        return html.Div([
        html.H5(filename),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=content,style={'width': '100%', 'height': '100%'})
        
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
    Output('original_image', 'src'),
    [
        Input('upload-image', 'contents'),
        Input('original_image', 'src'),
    ]
)
def update_image(content, default_src):
    if content is not None:
        original_image = stringToImage(content)
        plt.imsave('original.png', original_image)
        return content
    return default_src

@app.callback(
    Output('binary_image', 'src'),
    [
    Input('upload-image', 'contents'),
    Input('d-value', 'value'),
    Input('sigma', 'value'),
    Input('b_size', 'value')
    ]

)
def update_image(content, d, sigma, b):
    # original_image = cv2.imread('table_targets\\20210325_121519.jpg')
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # plt.imsave('original.png', original_image)
    d = int(d)
    sigma = int(sigma)
    b = int(b)
    if b%2 == 0:
        b = b+1
    if content is not None:
        original_image = stringToImage(content)
    else:
        original_image = cv2.imread('original.png')
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)  # gray image
    # rgb_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # rgb image
    
    # binary image
    binary_img = rtd.binarize_image(gray,d=d, sig1=sigma, sig2=sigma, b_size=b, c=5)
    plt.imsave('binary.png', binary_img, cmap='gray')
    image_filename = 'binary.png'
    encoded_image_binary = base64.b64encode(open(image_filename, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image_binary.decode())


@app.callback(
    [
    Output('targets_image', 'src'),
    Output('container-button-basic', 'children')
    ],
    [
    Input('submit-val', 'n_clicks'),
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
    return 'data:image/png;base64,{}'.format(encoded_image_targets.decode()), 'After adjusting binarization press to find targets (it might take few minutes)' 

# app.run_server(mode="inline")
if __name__ == '__main__':
    app.run_server(debug=True)
