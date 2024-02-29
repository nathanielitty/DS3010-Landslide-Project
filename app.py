from flask import Flask, render_template, request
import numpy as np
import pickle
# I got help with this app https://www.youtube.com/watch?v=jQjjqEjZK58 and some other sources but I know Flask a little from other classes
app = Flask(__name__)

with open('logistic_regression_model.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)

with open('tuned_xgboost_model.pkl', 'rb') as f:
    xgboost_tuned = pickle.load(f)

with open('nn_xgb.pkl', 'rb') as f:
    nn_xgb = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('nn_svm_model.pkl', 'rb') as f:
    nn_svm_model = pickle.load(f)

with open('best_knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('voting_classifier_model.pkl', 'rb') as f:
    voting_classifier_model = pickle.load(f)

attribute_columns = ['Aspect', 'Curvature', 'Earthquake', 'Elevation', 'Flow', 'Lithology', 
                     'NDVI', 'NDWI', 'Plan', 'Precipitation', 'Profile', 'Slope']
model_names = ['Logistic Regression', 'Tuned XGBoost', 'NN-XGBoost', 'SVM', 'NN-SVM', 'KNN', 'Voting Classifier']

models = {
    'Logistic Regression': logistic_regression_model,
    'Tuned XGBoost': xgboost_tuned,
    'NN-XGBoost': nn_xgb,
    'SVM': svm_model,
    'NN-SVM': nn_svm_model,
    'KNN': knn_model,
    'Voting Classifier': voting_classifier_model
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_model_name = None
    slider_values = {col: 1 for col in attribute_columns}
    
    if request.method == 'POST':
        slider_values = {col: int(request.form.get(col, 1)) for col in attribute_columns}
        selected_model_name = request.form.get('model_selection')
        
        selected_model = models.get(selected_model_name)
        
        input_values = list(slider_values.values())
        
        if selected_model:
            prediction = selected_model.predict([input_values])[0]
        
    return render_template('index.html', attribute_columns=attribute_columns, model_names=model_names, prediction=prediction, slider_values=slider_values, selected_model_name=selected_model_name)

if __name__ == '__main__':
    app.run(debug=True)