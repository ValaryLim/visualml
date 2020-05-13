import plotly.graph_objects as go
import colorlover
import numpy as np
import dash_html_components as html


colorscale = [[0, '#FE346E'], [1, '#00A1AB']]
contourscale_zip = zip(np.arange(0, 1.01, 1 / 8), colorlover.scales['9']['div']['RdBu'])
contourscale = list(map(list, contourscale_zip))

def render(model, X_train, X_test, y_train, y_test, Z, xx, yy, mesh_step, threshold): 
    train_x_coord = [row[0] for row in X_train]
    train_y_coord = [row[1] for row in X_train]
    
    test_x_coord = [row[0] for row in X_test]
    test_y_coord = [row[1] for row in X_test]

    # layout
    layout = go.Layout(
        title = f'SVM',
        xaxis = dict(
            ticks='',
            showticklabels = False,
            showgrid = False,
            zeroline = False,
        ),
        yaxis = dict(
            ticks='',
            showticklabels = False,
            showgrid = False,
            zeroline = False,
        ),
        hovermode = 'closest',
        legend = dict(
            x = 0, 
            y = 0, 
            orientation = "h"),
        margin = dict(l = 0, r = 0, t = 0, b = 0),
        height = 700,
        width = 650,
    )


    # scatter plots
    train_plot = go.Scatter(
        x = train_x_coord,
        y = train_y_coord,
        mode = "markers",
        name = "Training Data",
        marker = {
            'size': 7,
            'symbol': 'circle',
            'color': y_train,
            'colorscale': colorscale,
        }
    )

    test_plot = go.Scatter(
        x = test_x_coord,
        y = test_y_coord,
        mode = "markers",
        name = "Testing Data",
        marker = {
            'size': 7,
            'symbol': 'square',
            'color': y_test,
            'colorscale': colorscale,
        }
    )

    # compute threshold
    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    range = max(abs(scaled_threshold - Z.min()),
                abs(scaled_threshold - Z.max()))

    # add contour
    contour = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        zmin=scaled_threshold - range,
        zmax=scaled_threshold + range,
        hoverinfo='none',
        showscale=False,
        contours=dict(
            showlines=False
        ),
        colorscale=contourscale,
        opacity=0.9
    )

    # plot threshold
    threshold_line = go.Contour(
        x = np.arange(xx.min(), xx.max(), mesh_step),
        y = np.arange(yy.min(), yy.max(), mesh_step),
        z = Z.reshape(xx.shape),
        showscale = False,
        hoverinfo = 'none',
        contours = dict(
            showlines = False,
            type = 'constraint',
            operation = '=',
            value = scaled_threshold,
        ),
        name=f'Threshold ({scaled_threshold:.3f})',
        line = dict(
            color = '#333333'
        )
    )

    return go.Figure(
        data = [contour, threshold_line, train_plot, test_plot], 
        layout = layout
    )