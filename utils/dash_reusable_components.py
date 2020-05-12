# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

def NamedDropdown(id, name, options, value):
    return html.Div(
        children = [
            html.P(
                children = f'{name}:',
                style = { 
                    'marginLeft': '3px',
                    'fontWeight': 'bold',
                }
            ),
            dcc.Dropdown(
                id = id,
                options = options,
                value = value,
                clearable = False,
                searchable = False,
            )
        ],
        style = { "margin": "20px 0" }
    )

def NamedLabelSlider(id, name, min, max, step, marks, value):
    return html.Div(
        children = [
            html.P(
                children = f'{name}:',
                style = {
                    'marginLeft': '3px',
                    'fontWeight': 'bold',
                }
            ),
            dcc.Slider(
                id = id,
                min = min, 
                max = max,
                step = step,
                marks = marks,
                value = value
            )
        ],
        style = { "margin": "20px 0" }
    )