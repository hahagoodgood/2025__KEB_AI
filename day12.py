import numpy as np
import pandas as pd
# from sklearn.impute import SimpleImputer

df = pd.DataFrame(
    {
        'A':[1, 2, np.nan, 4],
        'B':[np.nan, 12, 3, 4],
        'C':[1, 2, 3, 4]
    }
)
print(df)

df[['A', 'B']] = df[['A', 'B']].fillna(df[['A', 'B']].mean())
print(df)