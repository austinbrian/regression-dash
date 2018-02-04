'''
For running the app
'''

## Custom import
from regressions import calc_log_regs
from sklearn.linear_model import LogisticRegression

def get_scores():
    data_file = './data/president_counties.csv'
    return calc_log_regs(data_file)

def get_probabilities():
    pass
