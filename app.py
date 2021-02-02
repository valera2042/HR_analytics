# -*- coding: utf-8 -*-
import flask
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

# Use pickle to load in the pre-trained model

app = flask.Flask(__name__)
with open(f'model/Hready.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='back')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('mainm.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        city_development_index = flask.request.form['city_development_index']
        enrolled_university = flask.request.form['enrolled_university']
        relevent_experience = flask.request.form['relevent_experience']
        last_new_job = flask.request.form['last_new_job']
        company_size = flask.request.form['company_size']

        # Make DataFrame for model
        #input_variables = pd.DataFrame([[city_development_index, enrolled_university, relevent_experience, last_new_job]],
                                       #columns=['city_development_index', 'enrolled_university', 'relevent_experience', 'last_new_job'],
                                       #dtype=float,
                                       #index=['input'])
        int_features = [x for x in request.form.values()]
        final_features = np.array(int_features).reshape((1,-1))
      
    

        # Get the model's prediction
        prediction = model.predict(final_features)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('mainm.html',
                                     original_input={'city_development_index':city_development_index,
                                                     'enrolled_university':enrolled_university,
                                                     'relevent_experience':relevent_experience,
                                                     'last_new_job' : last_new_job,
                                                     'company_size' : company_size},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()
    
    


