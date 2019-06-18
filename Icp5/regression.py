import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train = pd.read_csv('train.csv')

garage_field = train['GarageArea']
sale_price = train['SalePrice']
plt.scatter(garage_field,sale_price,alpha=.50,color='b')
plt.xlabel('garage_field')
plt.ylabel('sale_price')
plt.show()

outliers = train['GarageArea']>200
train=train[outliers]
outliers = train['GarageArea']<=1000
train=train[outliers]

garage_field = train['GarageArea']
sale_price = train['SalePrice']
plt.scatter(garage_field,sale_price,alpha=.50,color='b')
plt.xlabel('garage_field')
plt.ylabel('sale_price')
plt.show()
