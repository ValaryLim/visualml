import numpy as np
from sklearn import metrics

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
import plotly.figure_factory as ff

def roc(model, X_test, y_test):
    y_pred = model.decision_function(X_test)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    
    # AUC score
    auc_score = metrics.roc_auc_score(y_test, y_pred)

    # ROC curve
    roc = go.Scatter(
        x = fpr, 
        y = tpr, 
        mode = 'lines',
        name = "Test Data"
    )

    layout = go.Layout(
        title = f'ROC Curve (AUC = {auc_score:.3f})',
        xaxis = dict(
            title = 'False Positive Rate (FPR)'
        ),
        yaxis = dict(
            title = 'True Positive Rate (TPR)'
        ),
        legend = dict(
            x = 0,
            y = 0.5,
            orientation = 'h'
        ),
        margin = dict(l = 0, r = 0, t = 40, b = 0),
        height = 300,
        width = 300,
    )

    figure = go.Figure(
        data = [roc],
        layout = layout
    )
    
    return dcc.Graph(figure = figure)

def confusionMatrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

    figure = ff.create_annotated_heatmap(
        x = ["Positive", "Negative"],
        y = ["Positive", "Negative"],
        z = [[0, 0], [0, 0]],
        xgap = 5,
        ygap = 5,
        annotation_text = [[f'TP: {tp}', f'FP: {fp}'], [f'FN: {fn}', f'TN: {tn}']],
        colorscale=[[0, 'rgb(255,255,255)'], [1.0, 'rgb(179, 217, 255)']]
    )

    figure.layout.update(
        xaxis = {
            'title': 'Actual Values',
            'side': 'top',
        },
        yaxis = {
            'title': 'Predicted Values',
        },
        margin = {
            'l': 50,
            't': 50,
            'r': 10,
        },
        height = 350,
        width = 300,
    )

    
    return html.Div(
        children = [
            html.H6("Confusion Matrix"),
            dcc.Graph(
                figure = figure
            )
        ]
    )