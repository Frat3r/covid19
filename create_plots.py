import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import plotly.express as px
from data_transform import *
from sklearn.pipeline import Pipeline

MIN_NUM_OF_CASES = 10000


def plot_covid19(data, country_list, country_col='Country', **kwargs):
    fig = px.line(data.loc[data[country_col].isin(country_list)], **kwargs)
    for dt in fig.data:
        dt.update(mode='markers+lines')
    fig.show()


covid_data = pd.read_csv('https://opendata.ecdc.europa.eu/covid19/casedistribution/csv')
covid_data.rename(columns={'countriesAndTerritories': 'Country', 'cases': 'Cases_daily',
                           'deaths': 'Deaths_daily'}, inplace=True)
min_cases = [1, 50, 100]
after_names = ['Days_after_outbreak%s' % min_case for min_case in min_cases]
drop_columns = DropColumns(['day', 'month', 'year', 'countryterritoryCode'])
date_to_datetime = DateToDatetime()
sort_by = SortBy()
country_cumsum_cases = CountryCumsum('Cases', 'Cases_daily')
country_cumsum_deaths = CountryCumsum('Deaths', 'Deaths_daily')
since_outbreak = SinceOutbreak(value_name='Cases', group_col='Country', min_cases=min_cases, new_col_names=after_names)
cov_pipeline = Pipeline(steps=[('drop_columns', drop_columns), ('date_to_datetime', date_to_datetime),
                               ('sort_by', sort_by), ('country_cumsum_cases', country_cumsum_cases),
                               ('country_cumsum_deaths', country_cumsum_deaths), ('since_outbreak', since_outbreak)])
transformed_cov = cov_pipeline.transform(covid_data)
countries_to_plot = transformed_cov['Country'].loc[(transformed_cov['Cases'] >= MIN_NUM_OF_CASES) |
                                                   (transformed_cov['Country'] == 'Poland')].drop_duplicates()
