import spacy
import re
from spacy.tokens import Span

# Carga el modelo de spaCy y añade el componente de reconocimiento de patrones
nlp = spacy.load("es_core_news_sm")

def reconocer_patrones(texto):
    doc = nlp(texto)
    patrones = {
        "fechas": [],
        "nombres": []
    }

    # Reconocimiento de fechas (expresión regular para capturar formatos comunes de fechas)
    regex_fecha = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}-\d{1,2}-\d{2,4}\b'
    patrones["fechas"] = re.findall(regex_fecha, texto)

    # Reconocimiento de nombres propios (usando entidades de spaCy)
    for entidad in doc.ents:
        if entidad.label_ == "PER":  # "PER" es la etiqueta de persona en el modelo de spaCy
            patrones["nombres"].append(entidad.text)

    return patrones

# Ejemplo de texto para prueba
texto = "El contrato fue firmado el 12/08/2023 por Juan Pérez."
print(reconocer_patrones(texto))
