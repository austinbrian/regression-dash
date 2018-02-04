import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import itertools
from time import time

def get_data():
    df = pd.read_csv('./data/president_counties.csv')
    df['inc_per_filer'] = df.AGI/df.num_returns
    df['clinton_win'] = 0
    df.loc[df.clinton>df.trump,'clinton_win'] = 1 # set up target variabl
    test_cols = df.columns[8:-1]
    y_var = df.clinton_win
    test_df = df[test_cols]

    return test_df, y_var

# test each var individually
def one_var(df,y):
    scores = {}
    for var in test_df.columns:
        X = df[var]
        X_reshaped = X.values.reshape(-1,1)
        model = lr.fit(X_reshaped,y_var)
        score = model.score(X_reshaped,y_var)
        scores[var] = score
        print(var, score)
    return scores

def comb_vars(df,y):
    t0 = time()
    scores = {}
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
        sys.stdout.write('\033[F') # moves to the last line
        sys.stdout.write('\033[K') # erases the last line

    print("complete in {:.2f} seconds".format(time()-t0))
    return scores

def main():
    df_test,y_var = get_data()
    scores = comb_vars(df_test,y_var)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(sorted_scores[:10])

if __name__ == "__main__":
    main()
