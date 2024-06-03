from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('Student_Performance.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    math_score = int(request.form['math_score'])
    reading_score = int(request.form['reading_score'])
    writing_score = int(request.form['writing_score'])
    
    features = np.array([[math_score, reading_score, writing_score]])
    prediction = model.predict(features)
    
    return render_template('index.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
