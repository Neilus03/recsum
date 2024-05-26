
from dataclasses import dataclass, field
from typing import List
from src.tasks.utils_typing import Generic as Template

"""
Definiciones de entidades
"""

@dataclass
class Medicación:
    """Se refiere a un fármaco o sustancia utilizada para diagnosticar, curar, tratar o prevenir enfermedades.
    Las medicaciones pueden administrarse en diversas formas y dosis y son cruciales 
    para el manejo de las condiciones de salud del paciente. Pueden clasificarse 
    según su uso terapéutico, mecanismo de acción o características químicas."""
    
    mención: str  # El nombre de la medicación. Ejemplos: "Aspirina" / "Aspirina"
    dosis: str  # La cantidad y frecuencia con la que se prescribe la medicación. Ejemplos: "100 mg al día" / "100 mg al dia"
    vía: str  # El método de administración de la medicación. Ejemplos: "oral" / "oral"
    propósito: List[str]  # Lista de razones o condiciones para las que se prescribe la medicación. Ejemplos: ["dolor", "inflamación"] / ["dolor", "inflamació"]
    fecha_inicio: str  # La fecha en la que se comenzó a tomar la medicación. Ejemplos: "01-01-2023" / "01-01-2023"
    fecha_fin: str  # La fecha en la que se dejó de tomar la medicación, si aplica. Ejemplos: "31-01-2023" / "31-01-2023"

@dataclass
class Enfermedad:
    """Se refiere a una condición de salud o enfermedad que afecta el funcionamiento normal del cuerpo.
    Las enfermedades pueden ser causadas por diversos factores, como infecciones, trastornos genéticos,
    elecciones de estilo de vida o factores ambientales. Pueden afectar diferentes sistemas del cuerpo
    y tener distintos grados de severidad."""
    
    mención: str  # El nombre de la enfermedad o condición de salud. Ejemplos: "Diabetes mellitus" / "Diabetis mellitus"
    síntomas: List[str]  # Lista de signos o síntomas asociados con la enfermedad. Ejemplos: ["sed excesiva", "orinar con frecuencia"] / ["set excessiva", "orinar amb freqüència"]
    tratamiento: List[str]  # Lista de tratamientos o intervenciones utilizadas para manejar la enfermedad. Ejemplos: ["insulina", "dieta"] / ["insulina", "dieta"]
    fecha_diagnostico: str  # La fecha en la que se diagnosticó la enfermedad. Ejemplos: "15-05-2018" / "15-05-2018"
    severidad: str  # El nivel de severidad de la enfermedad. Ejemplos: "crónica" / "crònica"

@dataclass
class ProcedimientoMedico:
    """Se refiere a las intervenciones médicas realizadas para diagnosticar o tratar enfermedades.
    Esto puede incluir cirugías, pruebas diagnósticas y otros tratamientos especializados."""
    
    mención: str  # El nombre del procedimiento médico. Ejemplos: "angioplastia" / "angioplàstia"
    fecha: str  # La fecha en la que se realizó el procedimiento. Ejemplos: "10-02-2023" / "10-02-2023"
    resultado: str  # El resultado o conclusión del procedimiento. Ejemplos: "éxito sin complicaciones" / "èxit sense complicacions"

@dataclass
class DatosHospitalización:
    """Se refiere a la información relacionada con la hospitalización de un paciente, incluida la
    fecha de ingreso, fecha de alta y motivo de la hospitalización. Los datos de hospitalización
    son esenciales para rastrear el estado de salud del paciente, el progreso del tratamiento y
    la utilización de recursos sanitarios."""
    
    fecha_ingreso: str  # La fecha en la que el paciente fue ingresado en el hospital. Ejemplos: "03-04-2024" / "03-04-2024"
    fecha_alta: str  # La fecha en la que el paciente fue dado de alta del hospital. Ejemplos: "10-04-2024" / "10-04-2024"
    motivo: str  # El motivo o causa de la hospitalización del paciente. Ejemplos: "infarto agudo de miocardio" / "infart agut de miocardi"
    unidad: str  # La unidad o departamento del hospital en el que el paciente estuvo internado. Ejemplos: "Unidad de Cuidados Intensivos" / "Unitat de Cures Intensives"
    medico_responsable: str  # El nombre del médico responsable del paciente durante la hospitalización. Ejemplos: "Dr. García" / "Dr. García"

@dataclass
class DatosPaciente:
    """Se refiere a la información relacionada con el historial médico de un paciente, incluido
    el nombre, la edad o la urgencia. Los datos del paciente son esenciales para que los proveedores
    de atención médica brinden el cuidado adecuado y tomen decisiones informadas sobre el manejo del paciente."""
    
    nombre: str  # El nombre del paciente. Ejemplos: "Juan López Martínez" / "Joan López Martínez"
    edad: int  # La edad del paciente. Ejemplos: 60 / 60
    urgencia: str  # El nivel de urgencia de la condición del paciente. Ejemplos: "dolor torácico agudo" / "dolor toràcic agut"
    sexo: str  # El sexo del paciente. Ejemplos: "masculino" / "masculí"
    fecha_nacimiento: str  # La fecha de nacimiento del paciente. Ejemplos: "01-01-1964" / "01-01-1964"
    antecedentes_personales: List[str]  # Lista de antecedentes médicos personales relevantes. Ejemplos: ["hipertensión", "diabetes"] / ["hipertensió", "diabetis"]
    antecedentes_familiares: List[str]  # Lista de antecedentes médicos familiares relevantes. Ejemplos: ["infarto de miocardio en el padre a los 70 años"] / ["infart de miocardi en el pare als 70 anys"]

@dataclass
class SignosVitales:
    """Se refiere a las mediciones de las funciones básicas del cuerpo que son esenciales para la vida.
    Los signos vitales incluyen la temperatura corporal, frecuencia cardíaca, presión arterial, 
    frecuencia respiratoria y saturación de oxígeno."""
    
    temperatura: float  # La temperatura corporal del paciente. Ejemplos: 36.5 / 36.5
    frecuencia_cardiaca: int  # La cantidad de latidos del corazón por minuto. Ejemplos: 72 / 72
    presion_arterial_sistolica: int  # La presión en las arterias cuando el corazón late. Ejemplos: 120 / 120
    presion_arterial_diastolica: int  # La presión en las arterias entre los latidos del corazón. Ejemplos: 80 / 80
    frecuencia_respiratoria: int  # La cantidad de respiraciones por minuto. Ejemplos: 16 / 16
    saturacion_oxigeno: float  # El porcentaje de oxígeno en la sangre. Ejemplos: 98.0 / 98.0

@dataclass
class ResultadosLaboratorio:
    """Se refiere a los resultados de las pruebas de laboratorio realizadas durante la hospitalización 
    del paciente. Estas pruebas pueden incluir análisis de sangre, análisis de orina y otros estudios clínicos."""
    
    tipo_prueba: str  # El tipo de prueba de laboratorio realizada. Ejemplos: "análisis de sangre" / "anàlisi de sang"
    resultados: List[str]  # Los resultados específicos de la prueba. Ejemplos: ["glucosa: 90 mg/dL", "creatinina: 1.2 mg/dL"] / ["glucosa: 90 mg/dL", "creatinina: 1.2 mg/dL"]
    fecha: str  # La fecha en la que se realizaron las pruebas. Ejemplos: "01-06-2023" / "01-06-2023"

@dataclass
class ImagenDiagnostica:
    """Se refiere a los estudios de imagen realizados para diagnosticar o monitorear condiciones de salud.
    Estos estudios pueden incluir radiografías, tomografías, resonancias magnéticas, entre otros."""
    
    tipo_imagen: str  # El tipo de estudio de imagen. Ejemplos: "radiografía de tórax" / "radiografia de tòrax"
    hallazgos: str  # Los hallazgos o conclusiones del estudio de imagen. Ejemplos: "elevación del hemidiafragma izquierdo" / "elevació del hemidiafragma esquerre"
    fecha: str  # La fecha en la que se realizó el estudio de imagen. Ejemplos: "05-06-2023" / "05-06-2023"

@dataclass
class Recomendaciones:
    """Se refiere a las sugerencias y pautas proporcionadas al paciente al momento del alta para
    mejorar su salud y prevenir futuros episodios. Esto puede incluir cambios en el estilo de vida,
    medicamentos, y citas de seguimiento."""
    
    indicaciones: List[str] 

 # Lista de recomendaciones proporcionadas al paciente. Ejemplos: ["dieta baja en sal", "ejercicio moderado"] / ["dieta baixa en sal", "exercici moderat"]
    citas_seguimiento: List[str]  # Lista de citas de seguimiento programadas para el paciente. Ejemplos: ["cita con cardiólogo en 1 mes"] / ["cita amb el cardiòleg en 1 mes"]
    contactos_importantes: List[str]  # Información de contacto para consultas adicionales. Ejemplos: ["Dr. Pérez: 555-1234"] / ["Dr. Pérez: 555-1234"]

DEFINICIONES_ENTIDAD: List[Template] = [
    Medicación,
    Enfermedad,
    ProcedimientoMedico,
    DatosHospitalización,
    DatosPaciente,
    SignosVitales,
    ResultadosLaboratorio,
    ImagenDiagnostica,
    Recomendaciones,
]
