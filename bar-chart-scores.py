'''
This app will build a bar chart using dash that displays the relative scores of 
all possible logistic regressions for a given set of variables. 

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

def make_dataframe():
    data_file = './data/president_counties.csv'
    df_test, y_var = get_data(data_file)
    scores, probas = comb_vars(df_test, y_var)
    return scores, probas


app = dash.Dash()
app.css.append_css({'external_url':'https://codepen.io/chriddyp/pen/bWLwgP.css'})

