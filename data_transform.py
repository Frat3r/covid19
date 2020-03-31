import pandas as pd
from sklearn.base import TransformerMixin


class DropColumns(TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, data_df, y=None):
        return self

    def transform(self, data_df):
        return data_df.drop(self.columns_to_drop, axis=1)


class GroupByCountrySum(TransformerMixin):
    def __init__(self, group_col='Country/Region'):
        self.group_col = group_col

    def fit(self, data_df, y=None):
        return self

    def transform(self, data_df):
        return data_df.groupby(self.group_col).sum()


class MeltBy(TransformerMixin):
    def __init__(self, melt_col='Country/Region', var_name='Date', value_name='Value', var_date=1,
                 rename_col=0):
        self.melt_col = melt_col
        self.var_name = var_name
        self.value_name = value_name
        self.var_date = var_date
        self.rename_col = rename_col

    def fit(self, data_df, y=None):
        return self

    def transform(self, data_df):
        new_data_df = data_df.reset_index().melt(id_vars=self.melt_col, var_name=self.var_name,
                                                 value_name=self.value_name)
        if self.var_date:
            new_data_df[self.var_name] = pd.to_datetime(new_data_df[self.var_name])

        if self.rename_col:
            new_data_df.rename(columns=self.rename_col, inplace=True)
        return new_data_df


class SinceOutbreak(TransformerMixin):
    def __init__(self, value_name='Value', min_cases=[1], group_col='Country/Region',
                 new_col_names=['Days_since_outbreak']):
        self.min_cases = min_cases
        self.value_name = value_name
        self.new_col_names = new_col_names
        self.group_col = group_col

    def fit(self, data_df, y=None):
        return self

    def transform(self, data_df):
        new_data_df = data_df.copy()
        for col_name, min_cases in zip(self.new_col_names, self.min_cases):
            new_data_df[col_name] = new_data_df.loc[new_data_df[self.value_name] >=
                                                    min_cases].groupby(self.group_col).cumcount()
        return new_data_df


class DailyDiff(TransformerMixin):
    def __init__(self, val_col='Value', group_col='Country/Region', diff_col='Daily_difference'):
        self.val_col = val_col
        self.group_col = group_col
        self.diff_col = diff_col

    def fit(self, data_df, y=None):
        return self

    def transform(self, data_df):
        new_data_df = data_df.copy()
        new_data_df[self.diff_col] = data_df.groupby(self.group_col)[self.val_col].diff()
        return new_data_df
