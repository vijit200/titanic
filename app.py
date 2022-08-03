from cProfile import label
import numpy as np
from flask import Flask,request,render_template
import pickle
model = pickle.load(open('model.pkl','rb'))
process = pickle.load(open('process.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])

def predict():
    label = {0:'not survived',1:'survived'}
    if request.method == 'POST':
        Pclass = int(request.form['Pclass'])
        Age = int(request.form['Age'])
        Sex= int(request.form['Sex'])
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])
        l = [Pclass,Sex,Age,SibSp,Parch,Fare]
        l1 = np.array(l)
        s = l1.reshape(1,-1)
        p = process.transform(s)
        model_eval = model.predict(p)[0]
        return  render_template('index.html',predection_eval = label[model_eval])

if __name__ == '__main__':
    app.run(debug=True)