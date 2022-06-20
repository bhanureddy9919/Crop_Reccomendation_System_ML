from flask import Flask ,render_template,request,jsonify,session
import sqlite3 as sql
import base64
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from flask_bootstrap import Bootstrap
import numpy as np
from sklearn.utils import shuffle


app = Flask(__name__)
app.secret_key = 'any random string'

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

   
def validate(username,password):
    con = sql.connect('static/chat.db')
    completion = False
    with con:
        cur = con.cursor()
        cur.execute('SELECT * FROM persons')
        rows = cur.fetchall()
        for row in rows:
            dbuser = row[1]
            dbpass = row[2]
            if dbuser == username:
                completion = (dbpass == password)
    return completion


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        completion = validate(username,password)
        if completion == False:
            error = 'invalid Credentials. please try again.'
        else:
            session['username'] = request.form['username']
            return render_template('index111.html')
    return render_template('index.html', error=error)


@app.route('/view', methods=['GET', 'POST'])
def view():
    
    return render_template('index111.html')

@app.route('/cropa', methods=['GET', 'POST'])
def cropa():
    

    return render_template('cropdetails.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
    

    return render_template('About.html')



    
@app.route('/register', methods = ['GET','POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            username = request.form['username']
            password = request.form['password']
            with sql.connect("static/chat.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO persons(name,username,password) VALUES (?,?,?)",(name,username,password))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"
        finally:
            return render_template("index.html",msg = msg)
            con.close()
    return render_template('register.html')


@app.route('/list')
def list():
   con = sql.connect("static/chat.db")
   con.row_factory = sql.Row
   
   cur = con.cursor()
   cur.execute("select * from persons")
   
   rows = cur.fetchall();
   return render_template("list.html",rows = rows)

@app.route('/crop_predict',methods=['GET','POST'])
def cr():
    lt= []
    lst = []    
    data = pd.read_csv(r'dataset\Crop.csv')
    city = pd.read_csv(r'dataset\Ploted_600.csv')
            
    # data = shuffle(data)

    y = data.loc[:,'Crop']
    labelEncoder_y = LabelEncoder()
    y = labelEncoder_y.fit_transform(y)

    data['crop_num'] = y
    X = data.loc[:,['N','P','K','pH']].astype(float)
    y = data.loc[:,'crop_num']

    # Training Model
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_jobs=10, n_neighbors=10,weights='distance')
    clf.fit(X,y)

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

            
            for i in range(101):
                if real_pred[0][i] != 0.0:
                    lst.append(i)

            
            for i in range(10):
                load_data = data[data.index == lst[i]]
                for index, row in load_data.iterrows():
                    lt.append(row['Crop']) 
        # print(city,N,P,K,pH)
    return render_template('crop.html',crops=lt,crop_num = len(lt),display=True)


@app.route('/crop_pre',methods=['GET','POST'])
def crop():
    
    val_final=[] 
    data = pd.read_csv(r'dataset\Crop.csv')
    #print(data)

    city = pd.read_csv(r'dataset\Ploted_600.csv')
    #print(city)

    X = data.loc[:,['N','P','K','pH']]

    y = data.loc[:,['Crop']]
    X = data.loc[:,['N','P','K','pH']].astype(float)

    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_jobs=10, n_neighbors=10,weights='distance')
    clf.fit(X,y)

    lst = []
    lt = [] 
    

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
            


            
            for i in val_final:
                
                lst.append(i)

                
             
            
            for i in range(4):
                load_data = df[df.index == lst[i]]
                for index, row in load_data.iterrows():
                    lt.append(row['as'])     

            
        
    return render_template('soil_parameters.html',crops=lst,val_final=len(lst),display=True)


if __name__ == '__main__':
   app.run(debug = True )
