import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example

# Datos de entrenamiento (debes reemplazarlos con tus propios ejemplos)
TRAINING_DATA = [
    # Reportes
    ("Informe de resultados trimestrales del segundo trimestre.", {"cats": {"reporte": 1, "contrato": 0, "factura": 0}}),
    ("Reporte de desempeño del equipo de ventas anual.", {"cats": {"reporte": 1, "contrato": 0, "factura": 0}}),
    ("Este documento contiene un análisis del mercado actual.", {"cats": {"reporte": 1, "contrato": 0, "factura": 0}}),
    ("Informe de investigación sobre nuevas tecnologías en el sector.", {"cats": {"reporte": 1, "contrato": 0, "factura": 0}}),
    ("Resumen del proyecto de mejora de procesos para el mes de enero.", {"cats": {"reporte": 1, "contrato": 0, "factura": 0}}),
    ("Este reporte muestra las métricas de satisfacción del cliente.", {"cats": {"reporte": 1, "contrato": 0, "factura": 0}}),
    ("Informe de auditoría interna realizado en el último trimestre.", {"cats": {"reporte": 1, "contrato": 0, "factura": 0}}),
    ("Reporte sobre el crecimiento de ventas en el año fiscal 2023.", {"cats": {"reporte": 1, "contrato": 0, "factura": 0}}),
    ("Análisis financiero del desempeño anual de la compañía.", {"cats": {"reporte": 1, "contrato": 0, "factura": 0}}),
    ("Reporte financiero anual.", {"cats": {"reporte": 1, "contrato": 0, "factura": 0}}),

    # Contratos
    ("Contrato de arrendamiento entre el arrendador y el arrendatario.", {"cats": {"reporte": 0, "contrato": 1, "factura": 0}}),
    ("Acuerdo de confidencialidad firmado entre ambas partes.", {"cats": {"reporte": 0, "contrato": 1, "factura": 0}}),
    ("Este documento contiene los términos de un contrato de servicios.", {"cats": {"reporte": 0, "contrato": 1, "factura": 0}}),
    ("Acuerdo de prestación de servicios profesionales.", {"cats": {"reporte": 0, "contrato": 1, "factura": 0}}),
    ("Contrato de trabajo a tiempo parcial entre la empresa y el empleado.", {"cats": {"reporte": 0, "contrato": 1, "factura": 0}}),
    ("Este es un contrato de venta entre el comprador y el vendedor.", {"cats": {"reporte": 0, "contrato": 1, "factura": 0}}),
    ("Contrato de distribución para la comercialización de productos.", {"cats": {"reporte": 0, "contrato": 1, "factura": 0}}),
    ("Documento de acuerdo de licencia de uso de software.", {"cats": {"reporte": 0, "contrato": 1, "factura": 0}}),
    ("Acuerdo de colaboración entre las dos empresas para el proyecto.", {"cats": {"reporte": 0, "contrato": 1, "factura": 0}}),
    ("Contrato de servicios de tecnología entre las partes.", {"cats": {"reporte": 0, "contrato": 1, "factura": 0}}),

    # Facturas
    ("Factura número 7890 emitida el 15/09/2023.", {"cats": {"reporte": 0, "contrato": 0, "factura": 1}}),
    ("Total a pagar según factura: $4,500 USD.", {"cats": {"reporte": 0, "contrato": 0, "factura": 1}}),
    ("Documento correspondiente a la factura por servicios de consultoría.", {"cats": {"reporte": 0, "contrato": 0, "factura": 1}}),
    ("Factura de compra emitida por el proveedor.", {"cats": {"reporte": 0, "contrato": 0, "factura": 1}}),
    ("Factura de venta de productos realizada el 01/07/2023.", {"cats": {"reporte": 0, "contrato": 0, "factura": 1}}),
    ("Documento de pago por la factura emitida a nombre de la empresa.", {"cats": {"reporte": 0, "contrato": 0, "factura": 1}}),
    ("Factura de reparación de equipo número 2345.", {"cats": {"reporte": 0, "contrato": 0, "factura": 1}}),
    ("Total de la factura pendiente: $2,200.", {"cats": {"reporte": 0, "contrato": 0, "factura": 1}}),
    ("Factura generada por servicios de mantenimiento.", {"cats": {"reporte": 0, "contrato": 0, "factura": 1}})
    ("Factura número 12345 por servicios prestados.", {"cats": {"reporte": 0, "contrato": 0, "factura": 1}}),
]

# Carga el modelo base de spaCy y añade el componente de clasificación de texto
nlp = spacy.blank("es")
textcat = nlp.add_pipe("textcat")

# Añade etiquetas para cada tipo de documento
textcat.add_label("reporte")
textcat.add_label("contrato")
textcat.add_label("factura")

# Entrenamiento del modelo
optimizer = nlp.begin_training()
for i in range(10):  # Aumenta el número de épocas para mayor precisión
    losses = {}
    batches = minibatch(TRAINING_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optimizer, losses=losses)
    print(f"Epoch {i+1}, Loss: {losses['textcat']}")

# Guarda el modelo entrenado
nlp.to_disk("modelo_clasificacion")
print("Modelo de clasificación entrenado y guardado.")
