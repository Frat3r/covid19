import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import plotly.express as px
from data_transform import *
from sklearn.pipeline import Pipeline
MIN_NUM_OF_CASES = 2000


def plot_covid19(data, country_list, country_col='Country', **kwargs):
    fig = px.line(data.loc[data[country_col].isin(country_list)], **kwargs)
    for dt in fig.data:
        dt.update(mode='markers+lines')
    fig.show()


covid_cases = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv')
covid_deaths = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv')

min_cases = [1, 50, 100]
after_names = ['Days_after_outbreak%s' % min_case for min_case in min_cases]
drop_columns = DropColumns(['Province/State', 'Lat', 'Long'])
group_by_country_sum = GroupByCountrySum()
melt_by = MeltBy(rename_col={'Country/Region': 'Country'})
since_outbreak = SinceOutbreak(group_col='Country', min_cases=min_cases, new_col_names=after_names)
daily_diff = DailyDiff(group_col='Country')
cov_pipeline = Pipeline(steps=[('drop', drop_columns), ('group_by', group_by_country_sum), ('melt_by', melt_by),
                               ('daily_diff', daily_diff)])
transformed_cases = cov_pipeline.fit_transform(covid_cases)
transformed_cases = since_outbreak.transform(transformed_cases)
transformed_deaths = cov_pipeline.fit_transform(covid_deaths)
for after_name in after_names:
    transformed_deaths[after_name] = transformed_cases[after_name]

countries_to_plot = transformed_cases['Country'].loc[transformed_cases['Value'] >= MIN_NUM_OF_CASES].drop_duplicates()



