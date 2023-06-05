import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings(action='ignore')

# Load flight and hotel data
flight = pd.read_csv("/content/flights.csv")
hotel = pd.read_csv("/content/hotels.csv")

# Set up the inputs
day = 5
total_price = 2000
location = 'Brasilia (DF)'
location = np.array(location).reshape(-1, 1)

# create dirty data
# make new samples
dirty_from1 = pd.DataFrame({'travelCode': [135800],
                           'userCode': [1324],
                           'from': ['-'],
                           'to': ['Salvador (BH)'],
                           'flightType': ['economic'],
                           'price': [600.38],
                           'time': [2.01],
                           'distance': [612.0],
                           'agency': ['CloudFy'],
                           'date': ['07/16/2020']})

dirty_from2 = pd.DataFrame({'travelCode': [135777],
                           'userCode': [1324],
                           'from': [np.nan],
                           'to': ['Florianopolis (SC)'],
                           'flightType': ['premium'],
                           'price': [588.08],
                           'time': [1.52],
                           'distance': [577.0],
                           'agency': ['FlyingDrops'],
                           'date': ['08/16/2020']})

dirty_to1 = pd.DataFrame({'travelCode': [135301],
                           'userCode': [1331],
                           'from': ['Salvador (BH)'],
                           'to': ['??'],
                           'flightType': ['firstClass'],
                           'price': [600.38],
                           'time': [5.5],
                           'distance': [4000.0],
                           'agency': ['FlyingDrops'],
                           'date': ['12/10/2021']})

dirty_to2 = pd.DataFrame({'travelCode': [135330],
                           'userCode': [1333],
                           'from': ['Brasilia (DF)'],
                           'to': [np.nan],
                           'flightType': ['premium'],
                           'price': [1385.49],
                           'time': [1.84],
                           'distance': [709.37],
                           'agency': ['Rainbow'],
                           'date': ['07/16/2020']})

dirty_price1 = pd.DataFrame({'travelCode': [135330],
                           'userCode': [1333],
                           'from': ['Brasilia (DF)'],
                           'to': ['Florianopolis (SC)'],
                           'flightType': ['economic'],
                           'price': [np.nan],
                           'time': [2.14],
                           'distance': [609.37],
                           'agency': ['Rainbow'],
                           'date': ['05/16/2020']})

dirty_price2 = pd.DataFrame({'travelCode': [135331],
                           'userCode': [1333],
                           'from': ['Brasilia (DF)'],
                           'to': ['Florianopolis (SC)'],
                           'flightType': ['economic'],
                           'price': [-130.22],
                           'time': [2.14],
                           'distance': [609.37],
                           'agency': ['Rainbow'],
                           'date': ['07/15/2022']})

dirty_price3 = pd.DataFrame({'travelCode': [135331],
                           'userCode': [1333],
                           'from': ['Brasilia (DF)'],
                           'to': ['Florianopolis (SC)'],
                           'flightType': ['economic'],
                           'price': [np.nan],
                           'time': [2.14],
                           'distance': [609.37],
                           'agency': ['Rainbow'],
                           'date': ['07/15/2022']})

dirty_distance1 = pd.DataFrame({'travelCode': [135337],
                           'userCode': [1323],
                           'from': ['Florianopolis (SC)'],
                           'to': ['Recife (PE)'],
                           'flightType': ['economic'],
                           'price': [640.22],
                           'time': [2.14],
                           'distance': [np.nan],
                           'agency': ['FlyingDrops'],
                           'date': ['07/16/2021']})

dirty_distance2 = pd.DataFrame({'travelCode': [135337],
                           'userCode': [1323],
                           'from': ['Florianopolis (SC)'],
                           'to': ['Recife (PE)'],
                           'flightType': ['economic'],
                           'price': [461.02],
                           'time': [2.14],
                           'distance': [-0.1],
                           'agency': ['FlyingDrops'],
                           'date': ['01/1/2020']})

dirty_distance3 = pd.DataFrame({'travelCode': [135338],
                           'userCode': [1324],
                           'from': ['Aracaju (SE)'],
                           'to': ['Florianopolis (SC)'],
                           'flightType': ['economic'],
                           'price': [777.82],
                           'time': [2.14],
                           'distance': [np.nan],
                           'agency': ['FlyingDrops'],
                           'date': ['08/12/2020']})

# insert dirty datas into the original dataset
flight = pd.concat([flight, dirty_from1],ignore_index=True)
flight = pd.concat([flight, dirty_from2],ignore_index=True)
flight = pd.concat([flight, dirty_to1],ignore_index=True)
flight = pd.concat([flight, dirty_to2],ignore_index=True)
flight = pd.concat([flight, dirty_price1],ignore_index=True)
flight = pd.concat([flight, dirty_price2],ignore_index=True)
flight = pd.concat([flight, dirty_price3],ignore_index=True)
flight = pd.concat([flight, dirty_distance1],ignore_index=True)
flight = pd.concat([flight, dirty_distance2],ignore_index=True)
flight = pd.concat([flight, dirty_distance3],ignore_index=True)
print(flight.info())

#전처리
print(flight.isnull().sum())
print()

# Checking for garbage values in 'from' and 'to' columns
garbage_from = flight[flight['from'].isin(['-', '??',np.nan])]
garbage_to = flight[flight['to'].isin(['-', '??',np.nan])]

# Printing the rows with garbage values
print("Garbage values in 'from' column:")
print(garbage_from)

print("\nGarbage values in 'to' column:")
print(garbage_to)

# Removing rows with garbage values in 'from' and 'to' columns
flight = flight[~flight.index.isin(garbage_from.index)]
flight = flight[~flight.index.isin(garbage_to.index)]

# Resetting the index
flight.reset_index(drop=True, inplace=True)
print('after delete')
print(flight.isnull().sum())

# Checking for garbage values in 'price' column
garbage_price = flight[flight['price'] <= 0]

# Printing the rows with garbage values
print("Garbage values in 'price' column:")
print(garbage_price)


# Calculate the mean of 'price' excluding null and negative values
mean_price = flight['price'][flight['price'] > 0].mean()
print('mean price')
print(mean_price)
# Fill null and negative values with the mean price
flight['price'].fillna(mean_price, inplace=True)
flight.loc[flight['price'] <= 0, 'price'] = mean_price

# Checking for garbage values in 'distance' column
garbage_price = flight[flight['distance'] <= 0]

# Printing the rows with garbage values
print("Garbage values in 'price' column:")
print(garbage_price)


# Calculate the mean of 'price' excluding null and negative values
distance_mean_price = flight['distance'][flight['distance'] > 0].mean()
print('distance_mean_price price')
print(distance_mean_price)
# Fill null and negative values with the mean price
flight['distance'].fillna(distance_mean_price, inplace=True)
flight.loc[flight['price'] <= 0, 'price'] = distance_mean_price

print('final')
print(flight.isnull().sum())
print(flight.info())


#----------------------------------------------------------------------------------
#put dirty datas into hotels dataset 

# outlier - hotels dataset에는 적용되는 outlier없음
# def remove_outliers(data, column):
#     # calculate outlier
#     Q1 = data[column].quantile(0.25)
#     Q3 = data[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_fence = Q1 - 1.5 * IQR
#     print(lower_fence)
#     upper_fence = Q3 + 1.5 * IQR
#     print(upper_fence)

#     # remove outliers
#     filtered_data = data[(data[column] >= lower_fence) & (data[column] <= upper_fence)]
#     return filtered_data

# remove 'price' column's outliers
# hotel = remove_outliers(hotel, 'total')

#redefine outlier 
lower_fence = 10
upper_fence = 1250
print(hotel.info())

#remove outlier
hotel = hotel[(hotel['total'] >= lower_fence) & (hotel['total'] <= upper_fence)]
print(hotel.info())


# Preprocessing
# Select relevant columns from flight and hotel data
flight = flight.loc[:, ["from", "to", "flightType", "price", "time", 'date', 'distance']]
hotel = hotel.loc[:, ["place", "price"]]

# Use label encoding to encode categorical variables in hotel and flight data
label_encoder = LabelEncoder()
label_encoder.fit(hotel["place"])
hotel["place"] = label_encoder.transform(hotel["place"])
flight["from"] = label_encoder.transform(flight["from"])
flight["to"] = label_encoder.transform(flight["to"])

# Convert the current location to its label-encoded form
current_location = label_encoder.transform(location)[0]

# Filter flight data based on the current location
new_data = flight.loc[flight['from'] == current_location, :]

# Use label encoding for the flightType column
label_encoder_flight = LabelEncoder()
label_encoder_flight.fit(flight["flightType"])
flight["flightType"] = label_encoder_flight.transform(flight["flightType"])

# Convert the "date" column to datetime format and scale it using MinMaxScaler
flight["date"] = pd.to_datetime(flight["date"])
flight["date"] = flight["date"].apply(lambda x: int(x.strftime('%m%d%Y')))
date_scaler = MinMaxScaler()
date_scaler.fit(flight[["date"]])
flight[["date"]] = date_scaler.transform(flight[["date"]])

today = pd.to_datetime('06/03/2023')
today = int(today.strftime('%m%d%Y'))
today = np.array(today).reshape(-1, 1)
today = date_scaler.transform(today)

# Scale the "price" column by dividing it by the "distance" column
flight["new_price"] = flight["price"].values.reshape(-1, 1) / flight["distance"].values.reshape(-1, 1)

new_data_economic = new_data.loc[new_data["flightType"] == "economic", :]
new_data_first = new_data.loc[new_data["flightType"] == "firstClass", :]
new_data_premium = new_data.loc[new_data["flightType"] == "premium", :]

# Calculate the average hotel price for each destination
new_data_economic['hotel_price'] = 0
for i in range(0, 9):
    new_data_economic.loc[new_data_economic['to'] == i, 'hotel_price'] = hotel.loc[hotel["place"] == i, "price"].mean()
    new_data_first.loc[new_data_first['to'] == i, 'hotel_price'] = hotel.loc[hotel["place"] == i, "price"].mean()
    new_data_premium.loc[new_data_premium['to'] == i, 'hotel_price'] = hotel.loc[hotel["place"] == i, "price"].mean()

# Calculate the number of travel days based on the given day and flight time
new_data_economic['travel_day'] = -1
new_data_economic['travel_day'] = round(day - new_data_economic['time'] * 2)

new_data_first['travel_day'] = -1
new_data_first['travel_day'] = round(day - new_data_first['time'] * 2)

new_data_premium['travel_day'] = -1
new_data_premium['travel_day'] = round(day - new_data_premium['time'] * 2)

total_MSE = 0
total_MAE = 0
k = 5 # K-fold time
kf = KFold(n_splits=k, shuffle=True)
# Perform linear regression for each flight type (economic, first class, premium)

# ----------------------------------------economic
# K-fold- Economic Flights
flight_x = np.array(flight_economic['date']).reshape(-1, 1)
flight_y = np.array(flight_economic['new_price'])

mse_scores = []
mae_scores = []
models_economic = []

for train_index, test_index in kf.split(flight_x):
    # Data Segmentation
    flight_x_train, flight_x_test = flight_x[train_index], flight_x[test_index]
    flight_y_train, flight_y_test = flight_y[train_index], flight_y[test_index]
    
    # Model initialization and learning
    LR_flight_economic = LinearRegression().fit(flight_x_train, flight_y_train)
    
    # Forecast and performance assessment
    y_pred = LR_flight_economic.predict(flight_x_test)
    mse = mean_squared_error(flight_y_test, y_pred)
    mae = mean_absolute_error(flight_y_test, y_pred)
    
    mse_scores.append(mse)
    mae_scores.append(mae)
    models_economic.append(LR_flight_economic)

# K-fold Cross-validation result output
print("Economic Flights - K-fold Cross-validation MSE:", np.min(mse_scores))
print("Economic Flights - K-fold Cross-validation MAE:", np.min(mae_scores))

# Choose the best performing model
best_model_economic = models_economic[np.argmin(mse_scores)]

# Predict flight price for the given day
flight_economic['predict_flight_price'] = best_model_economic.predict(today) * flight_economic['distance']

# Repeat the same steps for first class and premium flights
# -------------------------------------------first
flight_x = np.array(flight_first['date']).reshape(-1, 1)
flight_y = np.array(flight_first['new_price'])

mse_scores = []
mae_scores = []
models_first = []

for train_index, test_index in kf.split(flight_x):
    # Data Segmentation
    flight_x_train, flight_x_test = flight_x[train_index], flight_x[test_index]
    flight_y_train, flight_y_test = flight_y[train_index], flight_y[test_index]
    
    # Model initialization and learning
    LR_flight_first = LinearRegression().fit(flight_x_train, flight_y_train)
    
    # Forecast and performance assessment
    y_pred = LR_flight_first.predict(flight_x_test)
    mse = mean_squared_error(flight_y_test, y_pred)
    mae = mean_absolute_error(flight_y_test, y_pred)
    
    mse_scores.append(mse)
    mae_scores.append(mae)
    models_first.append(LR_flight_first)

# K-fold Cross-validation result output
print("First Class Flights - K-fold Cross-validation MSE:", np.min(mse_scores))
print("First Class Flights - K-fold Cross-validation MAE:", np.min(mae_scores))

# Choose the best performing model
best_model_first = models_first[np.argmin(mse_scores)]

# Predict flight price for the given day
flight_first['predict_flight_price'] = best_model_first.predict(today) * flight_first['distance']

# -------------------------------------------premium
flight_x = np.array(flight_premium['date']).reshape(-1, 1)
flight_y = np.array(flight_premium['new_price'])

mse_scores = []
mae_scores = []
models_premium = []

for train_index, test_index in kf.split(flight_x):
    # Data Segmentation
    flight_x_train, flight_x_test = flight_x[train_index], flight_x[test_index]
    flight_y_train, flight_y_test = flight_y[train_index], flight_y[test_index]
    
    # Model initialization and learning
    LR_flight_premium = LinearRegression().fit(flight_x_train, flight_y_train)
    
    # Forecast and performance assessment
    y_pred = LR_flight_premium.predict(flight_x_test)
    mse = mean_squared_error(flight_y_test, y_pred)
    mae = mean_absolute_error(flight_y_test, y_pred)
    
    mse_scores.append(mse)
    mae_scores.append(mae)
    models_premium.append(LR_flight_premium)

# K-fold Cross-validation result output
print("Premium Flights - K-fold Cross-validation MSE:", np.min(mse_scores))
print("Premium Flights - K-fold Cross-validation MAE:", np.min(mae_scores))

# Choose the best performing model
best_model_premium = models_premium[np.argmin(mse_scores)]

# Predict flight price for the given day
flight_premium['predict_flight_price'] = best_model_premium.predict(today) * flight_premium['distance']






# ------------------------------------------

new_data_economic['total_price'] = -1
new_data_economic['total_price'] = new_data_economic['travel_day'] * new_data_economic['hotel_price'] + flight_economic[
    'predict_flight_price']

new_data_first['total_price'] = -1
new_data_first['total_price'] = new_data_first['travel_day'] * new_data_first['hotel_price'] + flight_first[
    'predict_flight_price']

new_data_premium['total_price'] = -1
new_data_premium['total_price'] = new_data_premium['travel_day'] * new_data_premium['hotel_price'] + flight_premium[
    'predict_flight_price']

# Build a decision tree classifier using the total price, flight type, and origin as features
from sklearn.tree import DecisionTreeClassifier

new_data = pd.concat([new_data_economic, new_data_first, new_data_premium])

new_data["flightType"] = label_encoder_flight.transform(new_data["flightType"])
new_data["date"] = flight['date']

x_train, x_test, y_train, y_test = train_test_split(new_data[['total_price', "flightType", "from"]], new_data['to'],
                                                    test_size=0.3, shuffle=True, stratify=new_data['to'], random_state=1)

x = np.array(x_train).reshape(-1, 3)
y = np.array(y_train)

tree = DecisionTreeClassifier(random_state=1)
tree.fit(x_train, y_train)

score = tree.score(x_test, y_test)




# --------------------------print step
print("Decision Tree Accuracy: ", score)

# Visualize the decision tree
from sklearn import tree as sklearn_tree
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
sklearn_tree.plot_tree(tree,
                       feature_names=['total_price', 'flightType', 'from'],
                       class_names=label_encoder.classes_,
                       filled=True,
                       rounded=True,
                       ax=ax)
plt.show()

# Multiple Decision Trees Ensemble (Random Forest)
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, random_state=1)
forest.fit(x_train, y_train)

score = forest.score(x_test, y_test)
print("Random Forest Accuracy: ", score)

# Prediction using the random forest model
prediction = forest.predict([[total_price, 2, current_location]])
predicted_location = label_encoder.inverse_transform(prediction)
print("Predicted Location: ", predicted_location)