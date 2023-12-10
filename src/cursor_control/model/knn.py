from unicodedata import decimal
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib as jl
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle as pkl


# Load the dataset
dataset_path = 'dataset/eye_movement.csv'
eye_data = pd.read_csv(dataset_path)


# Taking a look at first few rows
eye_data.head()

# Get basic information about the dataset
basicinfo = eye_data.info()

# Count of target values
eye_data['Target'].value_counts()

# Get summary statistics
summary_stats = eye_data.describe()
print(summary_stats)


#Display data types of columns
column_data_types = eye_data.dtypes
print(column_data_types)


#Distribution of individual features based on the target variable ('Target') 
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 10))
r = 0
c = 0
# Define a palette
colors = {0: 'skyblue', 1: 'salmon'}
for i in range(len(eye_data.columns) - 1):
    if c > 3:
        r += 1
        c = 0
    axes = ax[r, c]
    sns.boxplot(x=eye_data['Target'], y=eye_data[eye_data.columns[i]], ax=axes, palette=colors)
    c += 1
plt.tight_layout()
plt.suptitle("Individual Features by Class", y=1.02, fontsize=16)
plt.show()


# Distribution of Features in eye_data using histograms
sns.set_palette("Set3")
eye_data_hist = eye_data.hist(figsize=(20, 20), bins=50)
plt.show()


#Plot the counts of each target using a bar plot
# Define a color wheel
color = {0: "#0392cf", 1: "#7bc043"}

# Map colors to the "Target" column, handling potential None values
colors = eye_data["Target"].map(lambda x: color.get(x, "orange"))

# Plot the counts of each target using a bar plot
eye_data_bar = eye_data["Target"].value_counts().sort_index().plot(kind="bar", color=colors)
plt.xlabel("Target")
plt.ylabel("Count of Target")
plt.title("Distribution of Targets in eye_data")
# Show the plot
plt.show()

# Calculate the correlation matrix
eye_data_correlation_matrix = eye_data.corr()
# Create a heatmap
sns.heatmap(eye_data_correlation_matrix, annot=True, cmap="coolwarm")
plt.show()


# Convert all features to Decimal
eye_data[eye_data.columns.difference(['Target'])] = eye_data[eye_data.columns.difference(['Target'])].astype(float)


# Convert 'Target' column to int
eye_data['Target'] = eye_data['Target'].astype(int)


#Display data types of columns
column_data_types = eye_data.dtypes
print(column_data_types)

# Data Preprocessing

#Check for duplicates
if (eye_data.duplicated().any()):
    eye_data.drop_duplicates()

# Split dataset
dependent_value = 'Target'
X = eye_data.drop(dependent_value,axis=1)
y = eye_data[dependent_value]

# Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=1) 


# Train the machine learning model
knn_model.fit(X_train, y_train)


# Evaluate the model on the testing set
y_pred = knn_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Eye Testing Set: {mse}')

# Ensure 'y_test' is of the correct data type (int or bool)
y_test = y_test.astype(int)

# Check Accuracy
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy Score for KNN Model :  {accuracy}")
print(f'Confusion Matrix for KNN Model :{conf_matrix}')
print(f'Classification Report for KNN Model :{classification_rep}')

# Save the trained model to a file
model_filename = 'eye_movement_model.pkl'
jl.dump(knn_model, model_filename)
print(f'KNN Model has been saved to {model_filename}')


# Save the data columns from training set
model_columns = list(X.columns)
jl.dump(model_columns, 'eye_movement_columns.pkl')
print("KNN Model columns has been saved")