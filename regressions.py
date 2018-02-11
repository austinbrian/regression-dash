'''
This piece of code analyzes all combinations of variables available to a 
logistic regression, and returns the r-value for each model, along with the 
logistic regression probabilities (i.e., the probaility of a 1 for the target
variable) for each row of data in a dataset.

Author: @austinbrian
'''
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import itertools
from time import time

def get_data(filepath):
    '''
    This preps the presidential-counties vs IRS income-by-county dataset
    with a target variable (in this case a "Clinton wins = 1" variable
    and a couple other useful analysis variables
    '''
    df = pd.read_csv(filepath)
    df['inc_per_filer'] = df.AGI/df.num_returns
    df['clinton_win'] = 0

    # set up target variable, ignore independent votes for the moment
    df.loc[df.clinton>df.trump,'clinton_win'] = 1
    test_cols = df.columns[8:-1]
    y_var = df.clinton_win
    test_df = df[test_cols]

    return test_df, y_var

def comb_vars(df,y):
    '''
    This function creates a series of containers for each of the possible 
    combinations of variables in our dataset. Containers include:
        - SCORES:   the r-value score of the logistic regression associated with
                    a set of variables
        - PROBAS:   the probability estimates that a sample predicts a 1 for the 
                    target variable for each set of variables from the initial 
                    data
    '''
    t0 = time()
    scores = {}
    models = {}
    probas = {}
    iterations = []
    
    # set up for logisitic regression, but could be extended to SVM, etc
    lr = LogisticRegression()
    for i in range(1,len(df.columns)):
        for combination in list(itertools.combinations(df.columns,i)):
            untuple = list(combination)
            iterations.append(untuple)
    for n,i in enumerate(iterations):
        print(f'Estimating iteration {n+1} of {len(iterations)}')
        if len(i)==1:
            X = df[i].values.reshape(-1,1)
        else:
            X = df[i]
        model = lr.fit(X,y)
        score = model.score(X,y)
        probabilities = model.predict_proba(X)
        proba_1 = [p[1] for p in probabilities]
        scores[tuple(i)]= score
        models[tuple(i)] = model
        probas[tuple(i)] = np.array(proba_1)
        sys.stdout.write('\033[F') # moves to the last line
        sys.stdout.write('\033[K') # erases the last line

    print("complete in {:.2f} seconds".format(time()-t0))
    return scores, probas

def get_probas(df, models):
    '''This is a way to convert the models into a dataframe, not fully 
    built out yet.'''
    pf = pd.DataFrame(data = df.copy())
    for features in models.keys(): 
        model = models[features]
        if len(features)>1:
            X_features = pf.loc[:, list(features)]
        elif len(features)==1:
            X_features = pf.loc[:,''.join(list(features))]
        feature_probas = model.predict_proba(df[X_features].values.reshape(1,-1))
        pf.loc[:,features] = feature_probas
    return pf


def calc_log_regs(filepath):
    df_test,y_var = get_data(filepath)
    scores, models = comb_vars(df_test,y_var)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(sorted_scores[:10])
    return sorted_scores

if __name__ == "__main__":
    # calc_log_regs('./data/president_counties.csv')

    df_test, y_var = get_data('./data/president_counties.csv')
    scores, probas = comb_vars(df_test,y_var)
    return scores, probas

