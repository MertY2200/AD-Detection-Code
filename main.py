from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
model_file = "random_forest_model.pkl"
with open(model_file, 'rb') as f:
    loaded_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    data = request.form['input_data']
    print("Here is data You have entered",data)
    # Parse the input string and convert it to a list of floats
    data = [float(x.strip()) for x in data.split(',')]
    print("Here is data After conversions", data)
    # Make predictions using the loaded model
    prediction = loaded_model.predict([data])
    # Convert prediction to a standard Python list
    prediction_list = prediction.tolist()

    # Make predictions using the loaded model
    # This will work on only preprocessed data
    prediction = loaded_model.predict([data])

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction_list})

if __name__ == '__main__':
    app.run(debug=True)
