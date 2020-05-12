# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go

# svm model
import numpy as np
from sklearn import svm

import sklearn.model_selection
import sklearn.preprocessing

# import other functions
import datagen
import utils.dash_reusable_components as drc
import utils.svm_figure as svm_figure
import utils.evaluation_components as ec

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'style.css']

# initialise the application
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# application title
app.title = "Visual ML"

# html
app.layout = html.Div(children=[
    # header
    html.Div(
        children=[
            html.Div(children = [
                html.H1(children = 'Visual ML',
                    style = {
                        'paddingBottom': "0",
                        'marginBottom': "10px",
                        'letterSpacing': "0.1em"
                    }
                ),
                html.P(children = "an interactive dashboard to explore classification models")
            ],  
            style = {
                'padding': "50px 0 40px 0",
                'margin': "0",
                'textAlign': 'center',
                'color': "#ffffff",
                'backgroundColor': "#6886C5",
                'letterSpacing': "0.05em"
            },
            className = "twelve columns"),
        ],
        className = "row"
    ),

    # main section
    html.Div(
        children = [
            # data section
            html.Div(
                children = [
                    # choose model
                    drc.NamedDropdown(
                        id = "model",
                        name = "Choose Model",
                        options = [
                            {'label': 'Support Vector Machine (SVM)', 'value': 'svm'},
                            {'label': 'Logistic Regression', 'value': 'lr'},
                            {'label': 'Naive Bayes', 'value': 'nb'},
                            {'label': 'Decision Tree', 'value': 'dt'},
                        ],
                        value = "svm"
                    ),
                    html.Hr(),

                    # choose dataset
                    html.Div(
                        children = [
                            html.H6(children = "Dataset Options"),
                            # dataset type
                            drc.NamedDropdown(
                                id = "dataset-type",
                                name = "Choose Dataset",
                                options=[
                                    {'label': 'Linear', 'value': 'linear'},
                                    {'label': 'Moons', 'value': 'moons'},
                                    {'label': 'Circles', 'value': 'circles'},
                                ],
                                value = 'linear',
                            ),
                            # sample size
                            drc.NamedLabelSlider(
                                id = "dataset-size",
                                name = "Sample Size",
                                min = 100,
                                max = 500,
                                step = 100,
                                marks = {
                                    i: str(i) for i in range(100, 501, 100)
                                },
                                value = 300
                            ),
                            # noise
                            drc.NamedLabelSlider(
                                id = "dataset-noise",
                                name = "Noise",
                                min = 0,
                                max = 1,
                                step = 0.2,
                                marks = {
                                    i: str(i) for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                                },
                                value = 0.4
                            ), 
                        ]
                    ),
                    html.Hr(),

                    # model parameters: svm
                    html.Div(
                        id = "svm-parameters",
                        children = [
                            # kernel
                            drc.NamedDropdown(
                                id = "svm-kernel",
                                name = "Kernel",
                                options = [
                                    {'label': 'Linear', 'value': 'linear'},
                                    {'label': 'Polynomial', 'value': 'poly'},
                                    {'label': 'Radial Basis Function (RBF)', 'value': 'rbf'},
                                    {'label': 'Sigmoid', 'value': 'sigmoid'}
                                ],
                                value = 'linear'
                            ),
                            # cost
                            drc.NamedLabelSlider(
                                id = "svm-cost",
                                name = "Cost (C)",
                                min = -2,
                                max = 3,
                                step = 1,
                                marks =  {
                                    i: str(10**i) for i in range(-2, 4)
                                },
                                value = -2
                            ),
                            # gamma
                            drc.NamedLabelSlider(
                                id = "svm-gamma",
                                name = "Gamma",
                                min = -5,
                                max = 0,
                                step = 1, 
                                marks = { 
                                    i: str(10**i) for i in range(-5, 1)
                                },
                                value = -1,
                            ),
                            # polynomial degree
                            drc.NamedLabelSlider(
                                id = "svm-degree",
                                name = "Degree",
                                min = 2,
                                max = 10,
                                step = 2, 
                                marks = { 
                                    i: str(i) for i in range(2, 11, 2)
                                },
                                value = 2,
                            ),
                        ],
                        style = {
                            "display": "block"
                        }
                    ),
                    html.Hr()
                ],
                style = {
                    "paddingLeft": "20px",
                    "paddingRight": "30px",
                },
                className = "three columns",         
            ),
            html.Div(
                id = "svm-charts",
                style = {
                    "display": "block"
                }
            ),
        ],
        className = "row",
        style = { 
            "marginTop": 20,
        }
    ),

    # footer
    html.Div(
        children = [
            html.P(children = "built using Dash"),
            html.P(children = "clone of : https://dash-gallery.plotly.host/dash-svm/")
        ],
        style = {
            'textAlign': 'center',
            'color': "#ffffff",
            'backgroundColor': "#6886C5",
            'letterSpacing': "0.05em",
            'padding': '20px 0 10px 0',
            'margin': '0',
            'fontSize': "12px"
        },
        className = "row"
    ),
])

#### SVM OPTIONS ####
# display svm parameter options if model type == svm
@app.callback(
    dash.dependencies.Output('svm-parameters', 'style'),
    [dash.dependencies.Input('model', 'value')]
)
def update_parameters(model_type):
    if model_type == "svm":
        return { "display": "block" }
    else: 
        return { "display": "none" }

# display svm model output if model type == svm
@app.callback(
    dash.dependencies.Output('svm-charts', 'style'),
    [dash.dependencies.Input('model', 'value')]
)
def update_parameters(model_type):
    if model_type == "svm":
        return { "display": "block" }
    else: 
        return { "display": "none" }

# create SVM charts and models
@app.callback(
    dash.dependencies.Output('svm-charts', 'children'),
    [   
        # dataset inputs
        dash.dependencies.Input('dataset-type', 'value'), # dataset type
        dash.dependencies.Input('dataset-size', 'value'), # dataset sample size
        dash.dependencies.Input('dataset-noise', 'value'), # dataset noise level

        # model inputs
        dash.dependencies.Input('svm-kernel', 'value'), 
        dash.dependencies.Input('svm-cost', 'value'),
        dash.dependencies.Input('svm-gamma', 'value'),
        dash.dependencies.Input('svm-degree', 'value')
    ]
)
def update_svm(dataset, sample_size, noise, kernel, cost_power, gamma_power, degree):
    # generate data
    X, y = datagen.generate_data(
        dataset = dataset, n_samples = sample_size, noise = noise
    )
    
    X = sklearn.preprocessing.StandardScaler().fit_transform(X)
    
    # train test split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size = 0.3, random_state = 123
    )

    h = 0.3 # mesh step
    x_min = X[:, 0].min() - .5
    x_max = X[:, 0].max() + .5
    y_min = X[:, 1].min() - .5
    y_max = X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
   
    # create svm model
    cost = 10 ** cost_power
    gamma = 10 ** gamma_power
    model = svm.SVC(kernel = kernel, C = cost, gamma = gamma, degree = degree)
    model.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(model, "decision_function"):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # svm description
    description = '''A support vector machine takes in data points and 
    outputs the hyperplane (in two dimensions this is a line) 
    that best separates the tags. 
    This line is the decision boundary: anything that falls to 
    one side of it we will classify as green, and anything that 
    falls to the other as red.'''
    
    # create figure
    figure = svm_figure.render(
            model = model,
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            Z = Z,
            xx = xx,
            yy = yy,
            mesh_step = h,
            threshold = 0.5)

    # create roc curve
    roc_curve = ec.roc(model, X_test, y_test)

    # create confusion matrix
    confusion_matrix = ec.confusionMatrix(model, X_test, y_test)

    return [
        html.Div(
            children = [
                html.H6("Support Vector Machine (SVM)"),
                html.P(
                    children = description
                ),
                html.Br(),
                dcc.Graph(
                    figure = figure
                ),
            ],
            className = "six columns"
        ), 
        html.Div(
            children = [
                html.Div(
                    children = [roc_curve]
                ),
                html.Hr(),
                html.Div(
                    children = [confusion_matrix]
                )
            ],
            style = {
                'width': '25%',
                'textAlign': 'center'
            },
            className = "three columns"
        )
    ]

# disable gamma
@app.callback(
    dash.dependencies.Output('svm-gamma', 'disabled'),
    [
        dash.dependencies.Input('svm-kernel', 'value')
    ]
)
def disable_slider_param_gamma(kernel):
    return kernel not in ['rbf', 'poly', 'sigmoid']

# disable degree
@app.callback(
    dash.dependencies.Output('svm-degree', 'disabled'),
    [
        dash.dependencies.Input('svm-kernel', 'value')
    ]
)
def disable_slider_param_gamma(kernel):
    return kernel != 'poly'

if __name__ == '__main__':
	app.run_server(debug = True)