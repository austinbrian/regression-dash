import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import itertools
from time import time

def get_data(filepath):
    df = pd.read_csv(filepath)
    df['inc_per_filer'] = df.AGI/df.num_returns
    df['clinton_win'] = 0
    df.loc[df.clinton>df.trump,'clinton_win'] = 1 # set up target variable
    test_cols = df.columns[8:-1]
    y_var = df.clinton_win
    test_df = df[test_cols]

    return test_df, y_var

def comb_vars(df,y):
    t0 = time()
    scores = {}
    models = {}
    iterations = []
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
        scores[tuple(i)]= score
        models[tuple(i)] = model
        sys.stdout.write('\033[F') # moves to the last line
        sys.stdout.write('\033[K') # erases the last line

    print("complete in {:.2f} seconds".format(time()-t0))
    return scores, models

def get_probas(df, models):
    pf = pd.DataFrame(data = df.copy())
    for features in models.keys(): 
        model = models[features]
        if len(features)>1:
            X_features = pf.loc[:, list(features)]
        elif len(features)==1:
            X_features = pf.loc[:,features]
        pd.loc[:,features] = model.predict_proba(df[X_features])
    return pd


def calc_log_regs(filepath):
    df_test,y_var = get_data(filepath)
    scores, models = comb_vars(df_test,y_var)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(sorted_scores[:10])
    return sorted_scores

if __name__ == "__main__":
    # calc_log_regs('./data/president_counties.csv')

    df_test, y_var = get_data('./data/president_counties.csv')
    scores, models = comb_vars(df_test,y_var)
    df = get_probas(df_test, models)
    print(df.head())

