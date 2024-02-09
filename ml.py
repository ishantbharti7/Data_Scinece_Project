from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score

app = Flask(__name__)

data = pd.read_excel('train.xlsx')

# Extract features and target
features = data.drop('target', axis=1)  # Replace 'target_column' with your actual target column name
target = data['target']

# Convert target column to numerical labels
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Standardize the numeric features
numeric_features = features.select_dtypes(include=['int64', 'float64'])
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

# Combine scaled numeric features with categorical features
features_scaled = pd.concat([pd.DataFrame(numeric_features_scaled, columns=numeric_features.columns), features.select_dtypes(exclude=['int64', 'float64'])], axis=1)

# Find optimal number of clusters using silhouette score
best_score = -1
best_k = 2  # assuming there are at least 2 clusters
for k in range(2, 11):  # trying different numbers of clusters
    kproto = KPrototypes(n_clusters=k, random_state=42)
    clusters = kproto.fit_predict(features_scaled, categorical=list(range(len(numeric_features.columns), len(features_scaled.columns))))
    silhouette_avg = silhouette_score(features_scaled, clusters)
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_k = k

# Fit K-Prototype with the best number of clusters
kproto = KPrototypes(n_clusters=best_k, random_state=42)
clusters = kproto.fit_predict(features_scaled, categorical=list(range(len(numeric_features.columns), len(features_scaled.columns))))

# Assign cluster labels to the original dataset
data['cluster'] = clusters

# Function to identify cluster for a given data point
def identify_cluster(data_point):
    scaled_point_numeric = scaler.transform([data_point[0:len(numeric_features.columns)]])
    data_point_scaled = pd.concat([pd.DataFrame(scaled_point_numeric, columns=numeric_features.columns), pd.DataFrame([data_point[len(numeric_features.columns):]], columns=features.select_dtypes(exclude=['int64', 'float64']).columns)], axis=1)
    cluster_label = kproto.predict(data_point_scaled, categorical=list(range(len(numeric_features.columns), len(data_point_scaled.columns))))
    return cluster_label[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_data_point = []
        for i in range(1, 19):
            user_data_point.append(float(request.form[f't{i}']))
        user_data_point.append(request.form['target'])
        
        user_data_point_encoded = label_encoder.transform([user_data_point[-1]])[0]  # Encode the target column
        user_data_point = user_data_point[:-1] + [user_data_point_encoded]  # Replace the target column with the encoded value

        predicted_cluster = identify_cluster(user_data_point)
        return render_template('index.html', result=f'The provided data point belongs to Cluster {predicted_cluster}')

if __name__ == '__main__':
    app.run(debug=True)
