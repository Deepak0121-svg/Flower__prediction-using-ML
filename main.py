from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
with open('C:\\Flower_copy\\rose_1 (2).pkl', 'rb') as f:
    RFC = pickle.load(f)

# Define a dictionary to map species numbers to names
species_dict = {0: 'Hibiscus🌺', 1: 'Rose🌹', 2: 'Shoeblack Plant🏵️🏵️'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values from form
            height_cm = float(request.form['height_cm'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])

            # Create an input array for the model
            input_features = np.array([[height_cm, petal_length, petal_width, sepal_length, sepal_width]])

            # Make a prediction
            prediction = RFC.predict(input_features)[0]

            # Get the species name from the prediction
            species_name = species_dict.get(prediction, "Unknown Species")

            result = f"The predicted species is {species_name}."

        except ValueError as e:
            result = f"Error in input values: {e}"
        except Exception as e:
            result = f"An error occurred: {e}"

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
