import pandas as pd
from sklearn.base import TransformerMixin


class DropColumns(TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, data_df, y=None):
        return self

    def transform(self, data_df):
        return data_df.drop(self.columns_to_drop, axis=1)


class DateToDatetime(TransformerMixin):
    def __init__(self, date_col='dateRep', new_date_col='Date'):
        self.date_col = date_col
        self.new_date_col = new_date_col

    def fit(self, data_df, y=None):
        return self

    def transform(self, data_df):
        new_data_df = data_df.copy()
        new_data_df[self.new_date_col] = pd.to_datetime(new_data_df[self.date_col], dayfirst=False, format='%d/%m/%Y')
        #new_data_df.drop(self.date_col, inplace=True)
        return new_data_df


class SortBy(TransformerMixin):
    def __init__(self, sort_list=['Country', 'Date']):
        self.sort_list = sort_list

    def fit(self, data_df, y=None):
        return self

    def transform(self, data_df):
        return data_df.sort_values(self.sort_list, ascending=True)


class CountryCumsum(TransformerMixin):
    def __init__(self, new_col, value_col, group_col='Country'):
        self.group_col = group_col
        self.new_col = new_col
        self.value_col = value_col

    def fit(self, data_df, y=None):
        return self

    def transform(self, data_df):
        new_data_df = data_df.copy()
        new_data_df[self.new_col] = new_data_df.groupby(self.group_col)[self.value_col].cumsum()
        return new_data_df


class SinceOutbreak(TransformerMixin):
    def __init__(self, value_name, min_cases=[1], group_col='Country',
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
