from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
#import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import sqlite3

# Ruta completa del directorio donde se encuentra el script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Cambiar el directorio de trabajo al directorio del script
os.chdir(script_dir)

#os.chdir("C:\Users\Admin\OneDrive\Escritorio\BBK_The Bridge\Alumno\00_Material Pull\18 mayo\Model\ejercicio")
#os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo Silver"


#1. Ofrezca la predicción de ventas a partir de todos los valores de gastos en publicidad. (/v2/predict)
@app.route('/v2/predict', methods=['GET'])
def predict():
    model = pickle.load(open('modelo_decission_tree.pkl','rb'))

    Suma = request.args.get('Suma', None)
    ID_Agente = request.args.get('ID_Agente', None)
    Edad = request.args.get('Edad', None)
    Sexo_Femenino = request.args.get('Sexo_Femenino', None)
    Sexo_Masculino = request.args.get('Sexo_Masculino', None)
    Municipio_Amorebieta_Etxano = request.args.get('Municipio_Amorebieta_Etxano', None)
    Municipio_Arratia = request.args.get('Municipio_Arratia', None)
    Municipio_Barakaldo = request.args.get('Municipio_Barakaldo', None)
    Municipio_Basauri_Etxebarri = request.args.get('Municipio_Basauri_Etxebarri', None)
    Municipio_Bermeo = request.args.get('Municipio_Bermeo', None)
    Municipio_Bilbao = request.args.get('Municipio_Bilbao', None)
    Municipio_Busturialdea = request.args.get('Municipio_Busturialdea', None)
    Municipio_Durango = request.args.get('Municipio_Durango', None)
    Municipio_Erandio = request.args.get('Municipio_Erandio', None)
    Municipio_Ermua_Mallabia = request.args.get('Municipio_Ermua_Mallabia', None)
    Municipio_Galdakao = request.args.get('Municipio_Galdakao', None)
    Municipio_Getxo = request.args.get('Municipio_Getxo', None)
    Municipio_Goi_Enkarterri = request.args.get('Municipio_Goi_Enkarterri', None)
    Municipio_Lea_Artibai = request.args.get('Municipio_Lea_Artibai', None)
    Municipio_Leioa = request.args.get('Municipio_Leioa', None)
    Municipio_Meatzaldea = request.args.get('Municipio_Meatzaldea', None)
    Municipio_Mungialde = request.args.get('Municipio_Mungialde', None)
    Municipio_Nerbioi = request.args.get('Municipio_Nerbioi', None)
    Municipio_Portugalete = request.args.get('Municipio_Portugalete', None)
    Municipio_Santurtzi = request.args.get('Municipio_Santurtzi', None)
    Municipio_Sestao = request.args.get('Municipio_Sestao', None)
    Municipio_Trasladados = request.args.get('Municipio_Trasladados', None)
    Municipio_Txorierri = request.args.get('Municipio_Txorierri', None)
    Municipio_Uribe_Kosta = request.args.get('Municipio_Uribe_Kosta', None)
    Estado_Civil_Casado = request.args.get('Estado_Civil_Casado', None)
    Estado_Civil_Divorciado = request.args.get('Estado_Civil_Divorciado', None)
    Estado_Civil_Soltero = request.args.get('Estado_Civil_Soltero', None)
    Estado_Civil_Viudo = request.args.get('Estado_Civil_Viudo', None)
    
   

    if Suma is None or ID_Agente is None or Edad is None or Sexo_Femenino is None or Sexo_Masculino is None\
        or Municipio_Amorebieta_Etxano is None or Municipio_Arratia is None or Municipio_Barakaldo is None or Municipio_Basauri_Etxebarri is None\
            or Municipio_Bermeo is None or Municipio_Bilbao is None or Municipio_Busturialdea is None or Municipio_Durango is None\
                or Municipio_Erandio is None or Municipio_Ermua_Mallabia is None or Municipio_Galdakao is None or Municipio_Getxo is None\
                    or Municipio_Goi_Enkarterri is None or Municipio_Lea_Artibai is None or Municipio_Leioa is None or\
                        Municipio_Meatzaldea is None or Municipio_Mungialde is None or Municipio_Nerbioi is None or Municipio_Portugalete is None\
                            or Municipio_Santurtzi is None or Municipio_Sestao is None or Municipio_Trasladados is None\
                                or Municipio_Txorierri is None or Municipio_Uribe_Kosta is None or Estado_Civil_Casado is None\
                                    or Estado_Civil_Divorciado is None or Estado_Civil_Soltero is None or Estado_Civil_Viudo is None:
        return "Falta algún input por introducir"
    else:
        datos = np.array([Suma,ID_Agente,Edad,Sexo_Femenino,Sexo_Masculino,Municipio_Amorebieta_Etxano,Municipio_Arratia,Municipio_Barakaldo,Municipio_Basauri_Etxebarri\
            ,Municipio_Bermeo,Municipio_Bilbao,Municipio_Busturialdea,Municipio_Durango,Municipio_Erandio,Municipio_Ermua_Mallabia,Municipio_Galdakao\
                ,Municipio_Getxo,Municipio_Goi_Enkarterri,Municipio_Lea_Artibai,Municipio_Leioa,Municipio_Meatzaldea,Municipio_Mungialde,Municipio_Nerbioi\
                    ,Municipio_Portugalete,Municipio_Santurtzi,Municipio_Sestao,Municipio_Trasladados,Municipio_Txorierri,Municipio_Uribe_Kosta\
                        ,Estado_Civil_Casado,Estado_Civil_Divorciado,Estado_Civil_Soltero,Estado_Civil_Viudo])
        datos=datos.reshape(1, -1)
        prediction=model.predict(datos)
        prediction_string = ', '.join(prediction)
        resultado_lineas = prediction_string.replace(', ', '\n')
        return f"Las actividades recomendadas según el perfil son:\n {resultado_lineas}"

@app.route('/v2/retrain', methods=['PUT'])
def retrain():
    connection = sqlite3.connect('data/advertising.db')
    cursor = connection.cursor()
    query = '''SELECT * FROM campañas'''
    result = cursor.execute(query).fetchall()
    
    columns = []
    for i in cursor.description:
        columns.append(i[0])

    df = pd.DataFrame(data=result, columns=columns)
    X = df.drop(columns='sales')
    Y = df['sales']

    model = pickle.load(open('data/advertising_model','rb'))
    model.fit(X, Y)
    pickle.dump(model, open('data/advertising_model','wb'))

    new_score = cross_val_score(model, X, Y, cv = 10, scoring= 'neg_mean_absolute_error').mean()

    return f'El score de tu modelo re-entrenado es de {new_score}'



    return result



app.run()
