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

@app.route('/crop_pre',methods=['GET','POST'])
def crop():
    
    val_final=[] 
    data = pd.read_csv(r'E:\--------------------------\crop_prediction\dataset\Crop.csv')
    #print(data)

    city = pd.read_csv(r'E:\--------------------------\crop_prediction\dataset\Ploted_600.csv')
    #print(city)

    X = data.loc[:,['N','P','K','pH']]

    y = data.loc[:,['Crop']]
    X = data.loc[:,['N','P','K','pH']].astype(float)

    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_jobs=10, n_neighbors=10,weights='distance')
    clf.fit(X,y)

    
    

    if request.method == 'POST':
        crop = request.form['crop']
        N1 = request.form['Nitrogen']
        P1 = request.form['Phosphorous']
        K1 = request.form['Potassium']
        pH1 = request.form['pH']

        

          
        
        #crop = str(input('Crop_name'))
    
        if len(crop) != 0:
            print(crop)

            npk = data[data['Crop'] == crop]

            val = []
            for index, row in npk.iterrows():
                val = [row['N'],row['P'],row['K'],row['pH']]
            print(val)
            a=float(N1)
            b=float(P1)
            c=float(K1)
            d=float(pH1)
            
            val1=tuple(val)
            #val2=np.array(val0)        
            val0=(a,b,c,d)
            val2=tuple(val0)
            val_final=(val1[0]-val2[0],val1[1]-val2[1],val1[2]-val2[2],val1[3]-val2[3])

            df=pd.DataFrame(val_final,columns=['as'])
            


            lst = []
            for i in val_final:
                
                lst.append(i)

                
             
            lt = [] 
            for i in range(4):
                load_data = df[df.index == lst[i]]
                for index, row in load_data.iterrows():
                    lt.append(row['as'])     

            
        
    return render_template('soil_parameters.html',crops=lst,val_final=len(lst),display=True)

if __name__ == "__main__":
    app.run(debug=True)
