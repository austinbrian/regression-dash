'''
This app will build a bar chart using dash that displays the relative scores of 
all possible logistic regressions for a given set of variables. 

We'll make the relevant bar light up a different color when you select it in 
the dropdown.

Author: @austinbrian
'''
### Standard imports
import pandas as pd

### Dash imports
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

## Custom import
from regressions import get_data, comb_vars

def make_data():
    data_file = './data/president_counties.csv'
    df_test, y_var = get_data(data_file)
    scores, probas = comb_vars(df_test, y_var)
    return scores, probas

scores, probas = make_data()
var_iterations = scores.keys()

app = dash.Dash()
app.css.append_css({'external_url':'https://codepen.io/chriddyp/pen/bWLwgP.css'})

app.layout = html.Div([
    html.H1("Regression Bar Chart"),
    html.Div([

            html.Label('Combinations of variables'),
            dcc.Dropdown(
                id='var-select',
                options=[{'label': i, 'value': i} for i in var_iterations]
            ),
            html.Label('Sort type'),
            dcc.RadioItems(
                id='sort-type',
                options=[{'label': i, 'value': i} for i in ['Increasing',
                    'Decreasing']],
                value='Decreasing',
                labelStyle={'display': 'inline-block'}
            ),
            html.Div(id='regression-score')
        ],
        style={'width': '48%', 'display': 'inline-block'}),
    dcc.Graph(id='barchart')
    ])

@app.callback(
        dash.dependencies.Output('regression-score','children'),
        [dash.dependencies.Input('var-select','value')])
def update_r_score(input_value):
    s = scores[input_value]
    s_pct = (s*100)
    return "{:.1f}% accuracy".format(s_pct)

@app.callback(
        dash.dependencies.Output('barchart','figure'),
        [dash.dependencies.Input('var-select','value'),
         dash.dependencies.Input('sort-type','value')]
         )
def update_graph(var_selection,sort):
    if sort=="Descending":
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    elif sort == 'Ascending':
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    return { 
            'data':[go.Bar(
            x = list(sorted_scores.keys()),
            y = list(sorted_scores.values()),
            )], 
            'layout': go.Layout(
            title = 'Range of regression values',
            showlegend=False,
            margin = go.Margin(l=40, r=0, t=40, b=30)
            )
        }

if __name__=='__main__':
    app.run_server(debug=True)

