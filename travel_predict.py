import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
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

# Perform linear regression for each flight type (economic, first class, premium)

# ----------------------------------------economic
flight_economic = flight[flight['flightType'] == 0]
flight_first = flight[flight['flightType'] == 1]
flight_premium = flight[flight['flightType'] == 2]

flight_x_train, flight_x_test, flight_y_train, flight_y_test = train_test_split(
    flight_economic['date'],
    flight_economic['new_price'],
    test_size=0.3,
    random_state=20
)

flight_x_train = np.array(flight_x_train).reshape(-1, 1)
flight_y_train = np.array(flight_y_train)
LR_flight_economic = LinearRegression().fit(flight_x_train, flight_y_train)

flight_x_test = np.array(flight_x_test).reshape(-1, 1)
flight_y_test = np.array(flight_y_test)

# Calculate MSE and MAE for the linear regression model
mse = mean_squared_error(flight_y_test, LR_flight_economic.predict(flight_x_test))
total_MSE += mse
mae = mean_absolute_error(flight_y_test, LR_flight_economic.predict(flight_x_test))
total_MAE += mae

# Predict flight price for the given day
flight_economic['predict_flight_price'] = LR_flight_economic.predict(today) * flight_economic['distance']

# Repeat the same steps for first class and premium flights
# -------------------------------------------first
flight_x_train, flight_x_test, flight_y_train, flight_y_test = train_test_split(
    flight_first['date'],
    flight_first['new_price'],
    test_size=0.3,
    random_state=20
)

flight_x_train = np.array(flight_x_train).reshape(-1, 1)
flight_y_train = np.array(flight_y_train)
LR_flight_first = LinearRegression().fit(flight_x_train, flight_y_train)

flight_x_test = np.array(flight_x_test).reshape(-1, 1)
flight_y_test = np.array(flight_y_test)

# Calculate MSE and MAE for the linear regression model
mse = mean_squared_error(flight_y_test, LR_flight_first.predict(flight_x_test))
total_MSE += mse
mae = mean_absolute_error(flight_y_test, LR_flight_first.predict(flight_x_test))
total_MAE += mae

# Predict flight price for the given day
flight_first['predict_flight_price'] = LR_flight_first.predict(today) * flight_first['distance']

# -------------------------------------------premium
flight_x_train, flight_x_test, flight_y_train, flight_y_test = train_test_split(
    flight_premium['date'],
    flight_premium['new_price'],
    test_size=0.3,
    random_state=20
)

flight_x_train = np.array(flight_x_train).reshape(-1, 1)
flight_y_train = np.array(flight_y_train)
LR_flight_premium = LinearRegression().fit(flight_x_train, flight_y_train)

flight_x_test = np.array(flight_x_test).reshape(-1, 1)
flight_y_test = np.array(flight_y_test)

# Calculate MSE and MAE for the linear regression model
mse = mean_squared_error(flight_y_test, LR_flight_premium.predict(flight_x_test))
total_MSE += mse
mae = mean_absolute_error(flight_y_test, LR_flight_premium.predict(flight_x_test))
total_MAE += mae

# Predict flight price for the given day
flight_premium['predict_flight_price'] = LR_flight_premium.predict(today) * flight_premium['distance']

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
print("MSE for Linear Regression : ", total_MSE / 3)
print("MAE for Linear Regression : ", total_MAE / 3)
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
