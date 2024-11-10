from flask import Flask, request, jsonify
from Manejar_Archivos_api import procesar_documento
from Busqueda_Avanzada import buscar_documento
from Reconocer_Patrones import reconocer_patrones
from Entrenar_Modelo import entrenar_modelo

app = Flask(__name__)

# Configuración de la base de datos
db_config = {
    'host': 'localhost',       
    'user': 'admin',      
    'password': '1234',  
    'database': 'Clasificadordocumentos'
}

# Ruta de prueba para verificar la conexión
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Servidor Flask funcionando correctamente!"})

# Ruta para cargar y procesar documentos
@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró ningún archivo"}), 400

    archivo = request.files['file']
    tipo_documento = request.form.get('tipo_documento')  # Tipo de documento (reporte, contrato, factura)

    if archivo.filename == '':
        return jsonify({"error": "El nombre del archivo está vacío"}), 400

    try:
        # Llama a la función de `Manejar_Archivos_api` para procesar y almacenar el documento
        respuesta = procesar_documento(archivo, tipo_documento, db_config)
        return jsonify(respuesta)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ruta para realizar la búsqueda avanzada
@app.route('/buscar', methods=['GET'])
def buscar():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No se proporcionó una consulta de búsqueda"}), 400

    try:
        # Llama a la función de `Busqueda_Avanzada` para buscar documentos
        resultados = buscar_documento(query, db_config)
        return jsonify(resultados)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ruta para reconocer patrones específicos en un documento
@app.route('/reconocer_patrones', methods=['POST'])
def reconocer():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró ningún archivo"}), 400

    archivo = request.files['file']
    try:
        # Extraer contenido del archivo usando el módulo `Reconocer_Patrones`
        patrones = reconocer_patrones(archivo)
        return jsonify({"patrones": patrones})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ruta para entrenar el modelo (solo para desarrollo)
@app.route('/entrenar_modelo', methods=['POST'])
def entrenar():
    try:
        # Llama a la función de `Entrenar_Modelo` para entrenar el modelo de clasificación
        resultado = entrenar_modelo()
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
