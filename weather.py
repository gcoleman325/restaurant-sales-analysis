import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# import data
sales = pd.read_csv("C:/Users/ellie/Downloads/ItemSelectionDetails_2024_06_01-2024_11_30 - ItemSelectionDetails_2024_06_01-2024_11_30.csv")
weather = pd.read_csv("C:/Users/ellie/Downloads/23455 2024-06-01 to 2024-09-09.csv")

# preprocess
sales = sales[sales['Void?'] == False]

sales['weekday'] = pd.to_datetime(sales['Sent Date'], errors='coerce').dt.weekday
sales = sales[sales['weekday'].isin([4, 5, 6])]

weather = weather.drop(columns = ['description', 'sunrise', 'sunset', 'moonphase', 'snow', 'snowdepth','severerisk', 'name', 'conditions', 'icon', 'stations'])
sales = sales.drop(columns=['Location', 'Order #', 'Void?'])

weather['date'] = pd.to_datetime(weather['datetime']).dt.date
sales['date'] = pd.to_datetime(sales['Sent Date']).dt.date
sales.drop(columns = ['Sent Date'])
weather.drop(columns = ['datetime'])

sales_weather = pd.merge(sales, weather, left_on='date', right_on='date', how='inner')
sales_weather['total_sale'] = sales_weather['Net Price'] * sales_weather['Qty']
sales_weather = sales_weather[sales_weather['total_sale'] >= 3]

daily_sales = sales_weather.groupby('date').agg(
    daily_sales=('total_sale', 'sum'),
).reset_index()
daily_sales = pd.merge(daily_sales, weather, left_on='date', right_on='date', how='inner').drop(columns = ['datetime'])

alc_sales = sales_weather[sales_weather['Menu'] == 'BEVERAGES']
alc_sales = alc_sales[alc_sales['Menu Group'] != 'NA Beverages']

daily_alc_sales = alc_sales.groupby('date').agg(
    daily_alc_sales=('total_sale', 'sum'),
).reset_index()

food_sales = sales_weather[sales_weather['Sales Category'] == 'Food']
daily_food_sales = food_sales.groupby('date').agg(
    daily_food_sales=('total_sale', 'sum'),
).reset_index()

daily_sales = pd.merge(daily_alc_sales, daily_food_sales, on='date', how='outer')
daily_sales = pd.merge(daily_sales, weather, left_on='date', right_on='date', how='inner').drop(columns = ['datetime'])

# correlation matrix
numeric_cols = daily_sales.select_dtypes(include=['number'])
correlation_matrix = numeric_cols.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# regression analysis
X = daily_sales[['feelslike']]
y = daily_sales['daily_alc_sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

r2_score = model.score(X_test, y_test)
print(f"R-squared: {r2_score:.2f}")

feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(feature_importance)

# visualize
sns.regplot(x='feelslike', y='daily_alc_sales', data=daily_sales, 
            scatter_kws={'s': 10}, line_kws={'color': 'red'}, label='Alcohol Sales')

sns.regplot(x='feelslike', y='daily_food_sales', data=daily_sales,
            scatter_kws={'s': 10}, line_kws={'color': 'blue'}, label='Food Sales')
plt.xlabel('Heat Index')
plt.ylabel('Sales')
plt.title('Sales vs Heat Index')
plt.legend()
plt.show()