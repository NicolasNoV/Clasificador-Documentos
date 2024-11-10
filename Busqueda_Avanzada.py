from flask import Flask, request, jsonify
import spacy
import mysql.connector
from mysql.connector import Error

# Inicializa la aplicación de Flask y carga el modelo de spaCy
app = Flask(__name__)
nlp = spacy.load("es_core_news_sm")

# Configuración de la base de datos
db_config = {
    'host': 'localhost',
    'user': 'admin',         # Cambia a tu usuario de MySQL
    'password': '1234',   # Cambia a tu contraseña de MySQL
    'database': 'ClasificadorDocumentos'
}

# Función para conectar a la base de datos
def conectar_db():
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except Error as e:
        print("Error al conectar con MySQL:", e)
        return None

# Función para procesar y tokenizar la búsqueda
def procesar_busqueda(busqueda):
    doc = nlp(busqueda)
    tokens_clave = []

    for token in doc:
        # Filtra palabras innecesarias (stop words) y símbolos de puntuación
        if not token.is_stop and not token.is_punct:
            tokens_clave.append(token.lemma_)  # Usa el lematizador para una búsqueda más amplia

    return tokens_clave

# Ruta para realizar la búsqueda de documentos
@app.route('/buscar_documento', methods=['GET'])
def buscar_documento():
    busqueda = request.args.get('query')
    if not busqueda:
        return jsonify({"error": "No se proporcionó una consulta de búsqueda"}), 400

    # Procesa la búsqueda para obtener términos clave
    tokens_clave = procesar_busqueda(busqueda)
    if not tokens_clave:
        return jsonify({"error": "La búsqueda no generó términos clave válidos"}), 400

    # Conecta a la base de datos
    connection = conectar_db()
    if connection is None:
        return jsonify({"error": "No se pudo conectar a la base de datos"}), 500

    try:
        cursor = connection.cursor(dictionary=True)
        
        # Crear la consulta SQL para buscar coincidencias
        placeholders = ' OR '.join(['contenido_texto LIKE %s' for _ in tokens_clave])
        sql_query = f"SELECT * FROM documentos WHERE {placeholders}"
        
        # Preparar los parámetros de búsqueda agregando comodines '%' a los tokens
        parametros = [f"%{token}%" for token in tokens_clave]
        
        # Ejecutar la consulta
        cursor.execute(sql_query, parametros)
        resultados = cursor.fetchall()

        # Si no se encuentran resultados
        if not resultados:
            return jsonify({"message": "No se encontraron documentos coincidentes"}), 404

        # Retornar los resultados
        return jsonify({
            "resultados": resultados
        }), 200

    except Error as e:
        return jsonify({"error": f"Error al realizar la búsqueda en la base de datos: {str(e)}"}), 500

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == '__main__':
    app.run(debug=True)
