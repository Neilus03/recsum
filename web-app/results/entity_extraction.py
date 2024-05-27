import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jinja2 import Template as jinja2Template
import rich
import logging
from typing import List
from rich import print
import inspect

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
print(f'Project root: {project_root}')
gollie_med_path = os.path.join(project_root, "GoLLIE_MED")
sys.path.append(gollie_med_path)
print(f'GoLLIE_MED path: {gollie_med_path}')

from GoLLIE_MED.src.model.load_model import load_model
from GoLLIE_MED.src.tasks.utils_typing import AnnotationList, dataclass, Template 

# Configure logging
logging.basicConfig(level=logging.INFO)

# Check available CUDA devices and set the appropriate device
available_devices = torch.cuda.device_count()
print(f"Available CUDA devices: {available_devices}")

if available_devices > 0:
    device = torch.device(f"cuda:{available_devices - 1}")  # Use the last available CUDA device
    torch.cuda.set_device(device)
    print(f"Using CUDA device: {device}")
else:
    device = torch.device("cpu")
    print("No CUDA device available. Using CPU.")

# The rest of your code remains the same
model_path = "gollie_model.pt"
tokenizer_path = "gollie_tokenizer"

# Load or initialize model and tokenizer
if os.path.exists(model_path) and os.path.exists(tokenizer_path):
    logging.info(f"Loading model from {model_path}")
    model = torch.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
else:
    logging.info("Loading model from HiTZ/GoLLIE-7B")
    model, tokenizer = load_model(
        inference=True,
        model_weights_name_or_path="HiTZ/GoLLIE-7B",
        quantization=None,
        use_lora=False,
        force_auto_device_map=True,
        use_flash_attention=True,
        torch_dtype="bfloat16"
    )
    torch.save(model, model_path)
    tokenizer.save_pretrained(tokenizer_path)

# Ensure the model is on the correct device
model.to(device)



@dataclass
class Medicación:
    """Se refiere a un fármaco o sustancia utilizada para diagnosticar, curar, tratar o prevenir enfermedades.
    Las medicaciones pueden administrarse en diversas formas y dosis y son cruciales 
    para el manejo de las condiciones de salud del paciente. Pueden clasificarse 
    según su uso terapéutico, mecanismo de acción o características químicas."""
    
    mención: str  # El nombre de la medicación. Ejemplos: "Aspirina"
    dosis: str  # La cantidad y frecuencia con la que se prescribe la medicación. Ejemplos: "100 mg al día"
    vía: str  # El método de administración de la medicación. Ejemplos: "oral"
    propósito: List[str]  # Lista de razones o condiciones para las que se prescribe la medicación. Ejemplos: ["dolor", "inflamación"]
    fecha_inicio: str  # La fecha en la que se comenzó a tomar la medicación. Ejemplos: "01-01-2023"
    fecha_fin: str  # La fecha en la que se dejó de tomar la medicación, si aplica. Ejemplos: "31-01-2023"

@dataclass
class Enfermedad:
    """Se refiere a una condición de salud o enfermedad que afecta el funcionamiento normal del cuerpo.
    Las enfermedades pueden ser causadas por diversos factores, como infecciones, trastornos genéticos,
    elecciones de estilo de vida o factores ambientales. Pueden afectar diferentes sistemas del cuerpo
    y tener distintos grados de severidad."""
    
    mención: str  # El nombre de la enfermedad o condición de salud. Ejemplos: "Diabetes mellitus"
    síntomas: List[str]  # Lista de signos o síntomas asociados con la enfermedad. Ejemplos: ["sed excesiva", "orinar con frecuencia"]
    tratamiento: List[str]  # Lista de tratamientos o intervenciones utilizadas para manejar la enfermedad. Ejemplos: ["insulina", "dieta"]
    fecha_diagnostico: str  # La fecha en la que se diagnosticó la enfermedad. Ejemplos: "15-05-2018"
    severidad: str  # El nivel de severidad de la enfermedad. Ejemplos: "crónica"

@dataclass
class ProcedimientoMedico:
    """Se refiere a las intervenciones médicas realizadas para diagnosticar o tratar enfermedades.
    Esto puede incluir cirugías, pruebas diagnósticas y otros tratamientos especializados."""
    
    mención: str  # El nombre del procedimiento médico. Ejemplos: "angioplastia"
    fecha: str  # La fecha en la que se realizó el procedimiento. Ejemplos: "10-02-2023"
    resultado: str  # El resultado o conclusión del procedimiento. Ejemplos: "éxito sin complicaciones"

@dataclass
class DatosHospitalización:
    """Se refiere a la información relacionada con la hospitalización de un paciente, incluida la
    fecha de ingreso, fecha de alta y motivo de la hospitalización. Los datos de hospitalización
    son esenciales para rastrear el estado de salud del paciente, el progreso del tratamiento y
    la utilización de recursos sanitarios."""
    
    fecha_ingreso: str  # La fecha en la que el paciente fue ingresado en el hospital. Ejemplos: "03-04-2024"
    fecha_alta: str  # La fecha en la que el paciente fue dado de alta del hospital. Ejemplos: "10-04-2024"
    motivo: str  # El motivo o causa de la hospitalización del paciente. Ejemplos: "infarto agudo de miocardio"
    unidad: str  # La unidad o departamento del hospital en el que el paciente estuvo internado. Ejemplos: "Unidad de Cuidados Intensivos"
    medico_responsable: str  # El nombre del médico responsable del paciente durante la hospitalización. Ejemplos: "Dr. García"

@dataclass
class DatosPaciente:
    """Se refiere a la información relacionada con el historial médico de un paciente, incluido
    el nombre, la edad o la urgencia. Los datos del paciente son esenciales para que los proveedores
    de atención médica brinden el cuidado adecuado y tomen decisiones informadas sobre el manejo del paciente."""
    
    nombre: str  # El nombre del paciente. Ejemplos: "Juan López Martínez"
    edad: int  # La edad del paciente. Ejemplos: 60
    urgencia: str  # El nivel de urgencia de la condición del paciente. Ejemplos: "dolor torácico agudo"
    sexo: str  # El sexo del paciente. Ejemplos: "masculino"
    fecha_nacimiento: str  # La fecha de nacimiento del paciente. Ejemplos: "01-01-1964"
    antecedentes_personales: List[str]  # Lista de antecedentes médicos personales relevantes. Ejemplos: ["hipertensión", "diabetes"]
    antecedentes_familiares: List[str]  # Lista de antecedentes médicos familiares relevantes. Ejemplos: ["infarto de miocardio en el padre a los 70 años"]

@dataclass
class SignosVitales:
    """Se refiere a las mediciones de las funciones básicas del cuerpo que son esenciales para la vida.
    Los signos vitales incluyen la temperatura corporal, frecuencia cardíaca, presión arterial, 
    frecuencia respiratoria y saturación de oxígeno."""
    
    temperatura: float  # La temperatura corporal del paciente. Ejemplos: 36.5
    frecuencia_cardiaca: int  # La cantidad de latidos del corazón por minuto. Ejemplos: 72
    presion_arterial_sistolica: int  # La presión en las arterias cuando el corazón late. Ejemplos: 120
    presion_arterial_diastolica: int  # La presión en las arterias entre los latidos del corazón. Ejemplos: 80
    frecuencia_respiratoria: int  # La cantidad de respiraciones por minuto. Ejemplos: 16
    saturacion_oxigeno: float  # El porcentaje de oxígeno en la sangre. Ejemplos: 98.0

@dataclass
class ResultadosLaboratorio:
    """Se refiere a los resultados de las pruebas de laboratorio realizadas durante la hospitalización 
    del paciente. Estas pruebas pueden incluir análisis de sangre, análisis de orina y otros estudios clínicos."""
    
    tipo_prueba: str  # El tipo de prueba de laboratorio realizada. Ejemplos: "análisis de sangre"
    resultados: List[str]  # Los resultados específicos de la prueba. Ejemplos: ["glucosa: 90 mg/dL", "creatinina: 1.2 mg/dL"]
    fecha: str  # La fecha en la que se realizaron las pruebas. Ejemplos: "01-06-2023"

@dataclass
class ImagenDiagnostica:
    """Se refiere a los estudios de imagen realizados para diagnosticar o monitorear condiciones de salud.
    Estos estudios pueden incluir radiografías, tomografías, resonancias magnéticas, entre otros."""
    
    tipo_imagen: str  # El tipo de estudio de imagen. Ejemplos: "radiografía de tórax"
    hallazgos: str  # Los hallazgos o conclusiones del estudio de imagen. Ejemplos: "elevación del hemidiafragma izquierdo"
    fecha: str  # La fecha en la que se realizó el estudio de imagen. Ejemplos: "05-06-2023"

@dataclass
class Recomendaciones:
    """Se refiere a las sugerencias y pautas proporcionadas al paciente al momento del alta para
    mejorar su salud y prevenir futuros episodios. Esto puede incluir cambios en el estilo de vida,
    medicamentos, y citas de seguimiento."""
    
    indicaciones: List[str]  # Lista de recomendaciones proporcionadas al paciente. Ejemplos: ["dieta baja en sal", "ejercicio moderado"]
    citas_seguimiento: List[str]  # Lista de citas de seguimiento programadas para el paciente. Ejemplos: ["cita con cardiólogo en 1 mes"]
    contactos_importantes: List[str]  # Información de contacto para consultas adicionales. Ejemplos: ["Dr. Pérez: 555-1234"]

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

def get_gollie_entities(text: str) -> str:
    # Use inspect.getsource to get the guidelines as a string
    guidelines = [inspect.getsource(definition) for definition in DEFINICIONES_ENTIDAD]

    template_txt = (
        """# The following lines describe the task definition
    {%- for definition in guidelines %}
    {{ definition }}
    {%- endfor %}

    # This is the text to analyze
    text = {{ text.__repr__() }}

    # The annotation instances that take place in the text above are listed here
    result = [
    {%- for ann in annotations %}
        {{ ann }},
    {%- endfor %}
    ]
    """)

    template = jinja2Template(template_txt)
    gold = []
    formatted_text = template.render(guidelines=guidelines, text=text, annotations=gold, gold=gold)

    # Remove black formatting as it causes issues
    # black_mode = black.Mode()
    # formatted_text = black.format_str(formatted_text, mode=black_mode)
    rich.print(formatted_text)

    prompt, _ = formatted_text.split("result =")
    prompt = prompt + "result ="

    model_input = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    model_input["input_ids"] = model_input["input_ids"][:, :-1]
    model_input["attention_mask"] = model_input["attention_mask"][:, :-1]

    with torch.inference_mode():
        model_output = model.generate(
            **model_input.to(model.device),
            max_new_tokens=2048,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
        )

    for y, x in enumerate(model_output):
        print(f"Answer {y}")
        rich.print(tokenizer.decode(x, skip_special_tokens=True).split("result = ")[-1])

    result = AnnotationList.from_output(
        tokenizer.decode(model_output[0], skip_special_tokens=True).split("result = ")[-1],
        #task_module="main.guidelines"
        task_module = "guidelines" 
    )
    rich.print(result)

    with open("gollie_output.txt", "a") as f:
        f.write(str(result))

    return str(result)
