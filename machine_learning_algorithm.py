from flask import Flask , render_template, request,url_for
from flask_bootstrap import Bootstrap

# ML
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def main():
    return render_template('crop.html')

@app.route('/crop_predict',methods=['GET','POST'])
def crop():

    data = pd.read_csv('Crop.csv')
    city = pd.read_csv('Ploted_600.csv')
            
    # data = shuffle(data)

    y = data.loc[:,'Crop']
    labelEncoder_y = LabelEncoder()
    y = labelEncoder_y.fit_transform(y)

    data['crop_num'] = y
    X = data.loc[:,['N','P','K','pH']].astype(float)
    y = data.loc[:,'crop_num']

    # Training Model
    
	from sklearn.svm import SVC
    clf= SVC(kernel = kernels[i])
	clf.fit(X, y)
    if request.method == 'POST':
        city_name = request.form['city']
        N = request.form['Nitrogen']
        P = request.form['Phosphorous']
        K = request.form['Potassium']
        pH = request.form['pH']



        if len(city_name) != 0:
            print(city_name)

            npk = city[city['Location'] == city_name]

            val = []
            for index, row in npk.iterrows():
                val = [row['N'],row['P'],row['K'],row['pH']]
            print(val)
            columns = ['N','P','K','pH']
            values = np.array([val[0],val[1],val[2],val[3]])
            pred = pd.DataFrame(values.reshape(-1, len(values)),columns=columns)

            prediction = clf.predict(pred)
            # distances, indices = clf.kneighbors(pred,  n_neighbors=10)
            # prediction
            real_pred = clf.predict_proba(pred)
            print(real_pred)

            lst = []
            for i in range(101):
                if real_pred[0][i] != 0.0:
                    lst.append(i)

            lt= []
            for i in range(10):

                load_data = data[data.index == lst[i]]
                for index, row in load_data.iterrows():
                    lt.append(row['Crop'])

        else:
            print(N,P,K,pH)
            columns = ['N','P','K','pH']
            values = np.array([N,P,K,pH])
            pred = pd.DataFrame(values.reshape(-1, len(values)),columns=columns)

            prediction = clf.predict(pred)
            # distances, indices = clf.kneighbors(pred,  n_neighbors=10)
            # prediction
            real_pred = clf.predict_proba(pred)

            lst = []
            for i in range(101):
                if real_pred[0][i] != 0.0:
                    lst.append(i)

            lt= []
            for i in range(10):
                load_data = data[data.index == lst[i]]
                for index, row in load_data.iterrows():
                    lt.append(row['Crop'])
        # print(city,N,P,K,pH)
    return render_template('crop.html',crops=lt,crop_num = len(lt),display=True)

if __name__ == "__main__":
    app.run(debug=True)
