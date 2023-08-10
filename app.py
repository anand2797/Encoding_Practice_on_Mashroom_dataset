from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

app = Flask(__name__)

model = joblib.load('logistic_regression_model.pkl')
data = pd.read_csv('mushrooms.csv')

# replace all '-' to '_' in column names
# data.columns = data.columns.str.replace('-', '_')

# get unique values for categorical columns.
cap_shape_values = data['cap-shape'].unique()
cap_surface_values = data['cap-surface'].unique()
cap_color_values = data['cap-color'].unique()
bruises_values = data['bruises'].unique()
odor_values = data['odor'].unique()
gill_attachment_values = data['gill-attachment'].unique()
gill_spacing_values = data['gill-spacing'].unique()
gill_size_values = data['gill-size'].unique()
gill_color_values = data['gill-color'].unique()
stalk_shape_values = data['stalk-shape'].unique()
stalk_root_values = data['stalk-root'].unique()
stalk_surface_above_ring_values = data['stalk-surface-above-ring'].unique()
stalk_surface_below_ring_values = data['stalk-surface-below-ring'].unique()
stalk_color_above_ring_values = data['stalk-color-above-ring'].unique()
stalk_color_below_ring_values = data['stalk-color-below-ring'].unique()
veil_type_values =  data['veil-type'].unique()
veil_color_values = data['veil-color'].unique()
ring_number_values = data['ring-number'].unique()
ring_type_values = data['ring-type'].unique()
spore_print_color_values = data['spore-print-color'].unique()
population_values = data['population'].unique()
habitat_values = data['habitat'].unique()

# fit onehot encoder on nominal column in dataset
onehot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
onehot_encoder.fit(data.drop('class', axis=1))

# fit label encoder on target column
label_encoder = LabelEncoder()
label_encoder.fit(data['class'])


@app.route('/')
def home():
    return render_template('index.html', cap_shape_values=cap_shape_values, cap_surface_values=cap_surface_values,
                           cap_color_values=cap_color_values, bruises_values=bruises_values, odor_values=odor_values,
                           gill_attachment_values=gill_attachment_values, gill_spacing_values=gill_spacing_values,
                           gill_size_values=gill_size_values, gill_color_values=gill_color_values,
                           stalk_shape_values=stalk_shape_values, stalk_root_values=stalk_root_values,
                           stalk_surface_above_ring_values=stalk_surface_above_ring_values,
                           stalk_surface_below_ring_values=stalk_surface_below_ring_values,
                           stalk_color_above_ring_values=stalk_color_above_ring_values,
                           stalk_color_below_ring_values=stalk_color_below_ring_values,
                           veil_type_values=veil_type_values, veil_color_values=veil_color_values,
                           ring_number_values=ring_number_values, ring_type_values=ring_type_values,
                           spore_print_color_values=spore_print_color_values,
                           population_values=population_values, habitat_values=habitat_values)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        cap_shape = request.form['cap_shape']
        cap_surface = request.form['cap_surface']
        cap_color = request.form['cap_color']
        bruises = request.form['bruises']
        odor = request.form['odor']
        gill_attachment = request.form['gill_attachment']
        gill_spacing = request.form['gill_spacing']
        gill_size = request.form['gill_size']
        gill_color =request.form['gill_color']
        stalk_shape = request.form['stalk_shape']
        stalk_root = request.form['stalk_root']
        stalk_surface_above_ring = request.form['stalk_surface_above_ring']
        stalk_surface_below_ring = request.form['stalk_surface_below_ring']
        stalk_color_above_ring = request.args.get('stalk_color_above_ring')

        stalk_color_below_ring = request.form['stalk_color_below_ring']
        veil_type = request.form['veil_type']
        veil_color = request.form['veil_color']
        ring_number = request.form['ring_number']
        ring_type = request.form['ring_type']
        spore_print_color = request.form['spore_print_color']
        population = request.form['population']
        habitat = request.form['habitat']

        # create dataframe
        input_df = pd.DataFrame([[cap_shape, cap_surface, cap_color, bruises, odor,
                                  gill_attachment, gill_spacing, gill_size, gill_color,
                                 stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring,
                                 stalk_color_above_ring, stalk_color_below_ring,
                                 veil_type, veil_color, ring_number, ring_type,spore_print_color,
                                 population, habitat]], columns= data.columns[1:])

       # encode the categorical column
        onehot_encoded = onehot_encoder.transform(input_df)

       # preprocessed data
        input_processed = onehot_encoded.toarray()

        # make the prediction
        prediction = model.predict(input_processed)[0]
        #decode the predicted class by inverse-transform
        prediction_decode = label_encoder.inverse_transform([prediction])[0]

        # render the result
        return render_template('index.html', prediction_decode = prediction_decode)
if __name__ == '__main__':
    app.run(debug=True)