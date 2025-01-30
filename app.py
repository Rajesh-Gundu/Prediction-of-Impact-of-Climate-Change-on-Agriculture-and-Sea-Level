from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import pandas as pd
import pickle 

#loading model
TemperatureModel = pickle.load(open('TrainedModels/TemperatureModel.pkl','rb'))
RainfallModel = pickle.load(open('TrainedModels/RainfallModel.pkl','rb'))
MaizePredictor = pickle.load(open('TrainedModels/MaizePredictor.pkl','rb'))
PotatoesPredictor = pickle.load(open('TrainedModels/PotatoesPredictor.pkl','rb'))
PaddyPredictor = pickle.load(open('TrainedModels/PaddyPredictor.pkl','rb'))
WheatPredictor = pickle.load(open('TrainedModels/WheatPredictor.pkl','rb'))
SeaLevelPredictor = pickle.load(open('TrainedModels/SeaLevelPredictor.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        year = request.form.get('year')
        year = int(year)
        
        predTemp = TemperatureModel.predict([[year]])
        predRain = RainfallModel.predict([[year]])
        predMaize = MaizePredictor.predict([[year, predRain[0], predTemp[0]]])
        predPotato = PotatoesPredictor.predict([[year, predRain[0], predTemp[0]]])
        predPaddy = PaddyPredictor.predict([[year, predRain[0], predTemp[0]]])
        predWheat = WheatPredictor.predict([[year, predRain[0], predTemp[0]]])
        predSealevel = SeaLevelPredictor.predict([[year]])

        results = {
            'temperature': predTemp[0].item(),
            'rainfall': predRain[0].item(),
            'maize': predMaize[0].item(),
            'potato': predPotato[0].item(),
            'paddy': predPaddy[0].item(),
            'wheat': predWheat[0].item(),
            'sealevel': predSealevel[0].item()
        }

        return render_template('index.html',results = results)


if __name__ == '__main__':
    app.run(debug=True)
