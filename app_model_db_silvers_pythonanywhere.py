from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import sqlite3
import pandas as pd
from flask_cors import CORS, cross_origin
import json

# Ruta completa del directorio donde se encuentra el script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Cambiar el directorio de trabajo al directorio del script
os.chdir(script_dir)

app = Flask(__name__)
app.config['DEBUG'] = True

CORS(app, support_credentials=True)

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo Silver"


# 1. Recomendación de actividades. (/v2/predict)
@app.route('/v2/predict', methods=['GET'])
def predict():
    model = pickle.load(open('modelo_decission_tree.pkl','rb'))

    # Obtener los valores de los parámetros de la solicitud
    params = request.args

    # Verificar si se proporcionaron todos los parámetros necesarios
    required_params = ['Suma', 'ID_Agente', 'Edad', 'Sexo_Femenino', 'Sexo_Masculino', 'Municipio_Amorebieta_Etxano',
                       'Municipio_Arratia', 'Municipio_Barakaldo', 'Municipio_Basauri_Etxebarri', 'Municipio_Bermeo',
                       'Municipio_Bilbao', 'Municipio_Busturialdea', 'Municipio_Durango', 'Municipio_Erandio',
                       'Municipio_Ermua_Mallabia', 'Municipio_Galdakao', 'Municipio_Getxo', 'Municipio_Goi_Enkarterri',
                       'Municipio_Lea_Artibai', 'Municipio_Leioa', 'Municipio_Meatzaldea', 'Municipio_Mungialde',
                       'Municipio_Nerbioi', 'Municipio_Portugalete', 'Municipio_Santurtzi', 'Municipio_Sestao',
                       'Municipio_Trasladados', 'Municipio_Txorierri', 'Municipio_Uribe_Kosta', 'Estado_Civil_Casado',
                       'Estado_Civil_Divorciado', 'Estado_Civil_Soltero', 'Estado_Civil_Viudo']

    missing_params = [param for param in required_params if param not in params]

    if missing_params:
        return f"Faltan los siguientes parámetros: {', '.join(missing_params)}"

    # Convertir los valores de los parámetros en un arreglo numpy
    datos = np.array([params[param] for param in required_params]).reshape(1, -1)

    # Realizar la predicción utilizando el modelo
    prediction = model.predict(datos)

    # Convertir la predicción en una lista de respuestas
    #respuestas_recomendadas = prediction.tolist()


    # Crear el diccionario de respuesta con campos individuales
    response_data = {}
    activities = prediction[0].split(", ")
    for i, activity in enumerate(activities):
        response_data[f"actividad_{i+1}"] = activity

    # Establecer la cabecera Content-Type en application/json
    response_headers = {
        'Content-Type': 'application/json'
    }

    # Crear la respuesta HTTP con el objeto JSON y las cabeceras
    return jsonify(response_data), 200, response_headers

@app.route('/v2/retrain', methods=['GET', 'PUT', 'POST'])
def retrain():

    Suma = request.args.get('Suma', None)
    Suma_ant = request.args.get('Suma_ant', None)
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
    recursos_ut= request.args.get("recursos_ut",None)

    if recursos_ut is not None:
        recursos_ut = recursos_ut.replace("_", " ")
        recursos_ut = recursos_ut.replace(",", ", ")

    if Suma_ant is not None and Suma is not None:
        if int(Suma_ant) - int(Suma) >= 3:#Necesito que la llamada a la API incluya la suma del formulario anterior además del último para identificar casos de éxito
            connection = sqlite3.connect('silvers.sqlite')
            cursor = connection.cursor()
            query_insert = """
                INSERT INTO silvers_modelo (Suma, ID_Agente, Edad, Sexo_Femenino, Sexo_Masculino, "Municipio_Amorebieta-Etxano", Municipio_Arratia, Municipio_Barakaldo, "Municipio_Basauri/Etxebarri", Municipio_Bermeo,
    Municipio_Bilbao, Municipio_Busturialdea, Municipio_Durango, Municipio_Erandio,
    "Municipio_Ermua/Mallabia", Municipio_Galdakao, Municipio_Getxo, "Municipio_Goi Enkarterri",
    "Municipio_Lea Artibai", Municipio_Leioa, Municipio_Meatzaldea, Municipio_Mungialde,
    Municipio_Nerbioi, Municipio_Portugalete, Municipio_Santurtzi, Municipio_Sestao,
    Municipio_Trasladados, Municipio_Txorierri, "Municipio_Uribe Kosta", Estado_Civil_Casado,
    Estado_Civil_Divorciado, Estado_Civil_Soltero, Estado_Civil_Viudo, recursos_ut)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            values = (Suma, ID_Agente, Edad, Sexo_Femenino, Sexo_Masculino, Municipio_Amorebieta_Etxano, Municipio_Arratia, Municipio_Barakaldo, Municipio_Basauri_Etxebarri, Municipio_Bermeo,
                Municipio_Bilbao, Municipio_Busturialdea, Municipio_Durango, Municipio_Erandio,
                Municipio_Ermua_Mallabia, Municipio_Galdakao, Municipio_Getxo, Municipio_Goi_Enkarterri,
                Municipio_Lea_Artibai, Municipio_Leioa, Municipio_Meatzaldea, Municipio_Mungialde,
                Municipio_Nerbioi, Municipio_Portugalete, Municipio_Santurtzi, Municipio_Sestao,
                Municipio_Trasladados, Municipio_Txorierri, Municipio_Uribe_Kosta, Estado_Civil_Casado,
                Estado_Civil_Divorciado, Estado_Civil_Soltero, Estado_Civil_Viudo, recursos_ut)
            cursor.execute(query_insert, values)
            connection.commit()
            query = '''SELECT * FROM silvers_modelo'''
            result = cursor.execute(query).fetchall()

            columns = []
            for i in cursor.description:
                columns.append(i[0])

            df = pd.DataFrame(data=result, columns=columns)
            X = df[['Suma', 'ID_Agente', 'Edad', 'Sexo_Femenino', 'Sexo_Masculino', 'Municipio_Amorebieta-Etxano',
                    'Municipio_Arratia', 'Municipio_Barakaldo', 'Municipio_Basauri/Etxebarri', 'Municipio_Bermeo',
                    'Municipio_Bilbao', 'Municipio_Busturialdea', 'Municipio_Durango', 'Municipio_Erandio',
                    'Municipio_Ermua/Mallabia', 'Municipio_Galdakao', 'Municipio_Getxo', 'Municipio_Goi Enkarterri',
                    'Municipio_Lea Artibai', 'Municipio_Leioa', 'Municipio_Meatzaldea', 'Municipio_Mungialde',
                    'Municipio_Nerbioi', 'Municipio_Portugalete', 'Municipio_Santurtzi', 'Municipio_Sestao',
                    'Municipio_Trasladados', 'Municipio_Txorierri', 'Municipio_Uribe Kosta', 'Estado_Civil_Casado',
                    'Estado_Civil_Divorciado', 'Estado_Civil_Soltero', 'Estado_Civil_Viudo']]
            Y = df['recursos_ut']

            model = pickle.load(open('modelo_decission_tree.pkl','rb')) #Abre el modelo entrenado
            model.fit(X, Y) #Reentrena el modelo
            pickle.dump(model, open('modelo_decission_tree.pkl','wb')) #Vuelve a guardar el modelo reentrenado


            # Cerrar la conexión a la base de datos
            cursor.close()
            connection.close()

            return "Caso de éxito añadido al modelo"
        else:
            return "La diferencia entre Suma_ant y Suma no cumple el requisito."

    else:
        return "Los valores de Suma_ant y Suma son requeridos."

@app.route('/v2/last_line', methods=['GET'])
def last_line():
    connection = sqlite3.connect('silvers.sqlite')
    cursor = connection.cursor()
    query = '''SELECT * FROM silvers_modelo ORDER BY ROWID DESC LIMIT 1'''
    result = cursor.execute(query).fetchone()
    connection.close()

    if result:
        # Convertir la fila en un diccionario para facilitar la manipulación de los datos
        columns = [column[0] for column in cursor.description]
        row_dict = dict(zip(columns, result))
        return str(row_dict), 200  # Devuelve una tupla con la cadena de texto y el código de estado 200

    else:
        return "No se encontró ninguna fila en la tabla 'silvers_modelo'", 404  # Devuelve una tupla con la cadena de texto y el código de estado 404



@app.route('/v2/first_line', methods=['GET'])
def first_line():
    connection = sqlite3.connect('silvers.sqlite')
    cursor = connection.cursor()
    query = '''SELECT * FROM silvers_modelo ORDER BY ROWID ASC LIMIT 1'''
    result = cursor.execute(query).fetchone()
    connection.close()

    if result:
        # Convertir la fila en un diccionario para facilitar la manipulación de los datos
        columns = [column[0] for column in cursor.description]
        row_dict = dict(zip(columns, result))
        return str(row_dict), 200  # Devuelve una tupla con la cadena de texto y el código de estado 200

    else:
        return "No se encontró ninguna fila en la tabla 'silvers_modelo'", 404  # Devuelve una tupla con la cadena de texto y el código de estado 404



app.run()

