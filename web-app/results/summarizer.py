from groq import Groq
import os

# Initialize the Groq client with the API key from environment variables
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def merch_text_keywords(text, keywords):
    """Formats the input text and keywords into a single string."""
    merch = f"""Summarize the following text taking into account the keywords:
    Text: {text}
    Keywords: {keywords}
    """
    return merch

def get_model_response(text, keywords=None, schematic=False):
    """Makes an API call to Groq to get the summary of the text based on the keywords."""
    messages = []

    if keywords != None:
        # System prompt to guide the summarization process
        messages.append({
            "role": "system",
            "content": """
            Eres un modelo de resumen que recibe textos médicos en castellano o catalán con algunas palabras clave. Tu tarea es resumir el texto manteniendo el contexto de las palabras clave y redactando en español. Por ejemplo, dado el siguiente texto y palabras clave: 
            
            Texto: diagnostic alta codi icd-10 descripcio diagnostic k70.3 cirrosi hepatica alcoholica r18.8 altres tipus d'ascites dades informe motivo de ingreso: mal estado general antecedentes personales: no ram. hta. disnea a peque os esfuerzos y epoc gold ii en seguimiento en neumologia en tratamiento broncodilatador y oxigenoterapia en concentrador portatil a 3lx' para la deambulacion desde enero 2023. elevacion de hemidiafragma izquierdo y adenopatia hiliar derecha con captacion en pet-tc estudiado con ebus + fbs sin lesiones endobronquiales y con ap negativa para malignidad del bas y adenopatias puncionadas. hemorragia digestiva baja secundaria a colitis isquemica en setiembre 2022 (cordoba). colelitiasis. - cirrosis hepatica enolica child c en seguimiento en consulta ch intensivo (dra. martin) descompensacion edemo-ascitica en junio 2022 con empeoramiento tras ingreso hospitalario por colitis isquemica (setiembre 2022), en tratamiento diuretico con espironolactona 200mg/dia + furosemida 80mg/dia desde el 18.05.23 encefalopatia hepatica en agosto 2022. en tratamiento con lactulosa + rifaximina desde noviembre 2022 varices esofagicas peque as con signos de riesgo (fgs en setiembre - cordoba) en tratamiento con carvedilol 6.25mg/12horas ultima ecografia en marzo 2023: no loes hepaticas, porta permeable,discreta cantidad de liquido libre perihepatico y en pelvis ex enolismo cronico, abstinente desde setiembre 2022. fumador 5-6 cig/dia tratamiento habitual: acfol 5mg/dia, carvedilol 6.25mg/12horas, espironolactona 200mg/dia, furosemida 80mg/dia, lactulosa, spiolto 2ing/12horas, iabvd. vive con su madre. enfermedad actual traido por familiares a urgencias por deterioro del estado general, somnolencia, astenia y aumento de edemas en miembros inferiores a pesar de aumento de diureticos. refiere tambien que hace una semana cursa con deposiciones blandas (3-4 deposiciones/dia) sin productos patologicos motivo por el cual suspendio lactulosa. no fiebre, no sintomas urinarios, no dolor abdominal. hoy empeoramiento de su disnea habitual. exploracion fisica urgencias: t 35.9, tas 150, tad 65, sato2 bas 94, fr 16, fc 57 estado general conservado, piel y mucosas hidratadas levemente ictericas. nrl: consciente, orientado en tiempo, espacio y persona, discurso fluido y coherente, no flapping. ac: ruidos cardiacos ritmicos sin soplos. ar: crepitantes basales bilaterales e hipofonesis basal izquierda.  pacient cip data naix. 12.07.1962 edat 60 sexe home nass adre a cp poblacio tel.  admissio 31.05.2023 12:12 alta 16.06.2023 11:27 servei diguohmb digestologia unitat u05uhhbl data i hora d'impressio: 20.06.2023 02:53:22    pagina 1 de 4 informe alta hospitalitzacio abdomen globuloso, blando, depresible, no palpo masas ni megalias, rha conservados, no dolor a la palpacion, sin reaccion peritoneal, edema de pared. eeii: presenta edemas en eeii hasta muslos, fovea ++. planta digestivo ta 143/74 fc72 afebril. sato2 basal 90% ap: hipofonesis en base izquierda. abd: globuloso, no clara semiologia ascitica. no irritacion peritoneal. rha + ext: edemas ++/+++ hasta muslos sn: consciente y orientado. no flapping pruebas complementarias: + analitica: leucocitos 5740 (n 3220, l 1600), hb 123 g/l, plaquetas 140000, tp 49%, inr 1.64, ttpa 1.55, glucosa 48mg/dl, creatinina 1.7 mg/dl, fg 43.98 ml/min, urea 22.1 mg/dl, sodio 139.0 mmol/l, potasio 4.0 mmol/l, alt 19.8 ui/l, albumina 1.5 g/dl, pcr 2.79 mg/l, + eab (arterial): ph 7.366, co2 24.6 mmhg, o2 65.8 mmhg, bic 13.8 mmol/l, be -9.1, sat. 89.8%. + sedimento urinario: eritrocitos 20-50/c, leucocitos 3-5/c. + rx torax : elevacion de hemidiafragma izquierdo, borramiento de angulo costofrenico izquierdo. - paracentesis x3 : blanca - analitica 01.06: leucocitos 3910, hb 10.9 g/dl, hto 30%, vcm 92, plaquetas 108000, tp 43%, glucosa 133 mg/dl, na 143, k 4.2, br 3.77 mg/dl, ast/alt 37/21, fa/ggt 82/15, pcr 1. - hemocultivo: negativo - coprocultivo: negativo - adenovirus y rotavirus en heces: negativo - gdh c. difficile en heces: negativo - pcr covid: negativo - ecocardiograma : fevi conservado. hallazgos compatibles con cardiopatia hipertensiva. - urocultivo: negativo - gammagrafia : no hay imagenes sugestivas de corresponder con tep en la gpp obtenida y, en todo caso, valoramos el estudio indeterminado en localizacion de la base pulmonar izquierda donde hay una hipoperfusion moderada no segmentaria que coincide con alteracion radiologica (atelectasia/colapso basal sin descartar sobreinfeccion asociada?? y elevacion de este hemidiafragma). - cateterismo hepatico y cardiopulmonar: cateterismo de venas suprahepaticas por via yugular. obtencion de presiones hepaticas mediante cateter balon. mediciones repetidas de la presion suprahepatica enclavada (pse) y del gradiente de presion venosa hepatica (gpvh). resultados (mmhg) pse:31.0 psl:15.5 gpvh: 15.5 pvci: 15.5 pad: 13.0 2. cateterismo cardiopulmonar. pap: 32.0 mm hg pcp: 28.0 mm hg pad: 13.0 mm hg gc: 10.1 l/min pam: 115 mm hg fc: 74 latidos/min ic: 4.9 l/min/m2 sv: 136 ml/latido rvs: 808 dynassegcm-5 irvs: 1665 dynassegcm-5 rvp: 32 dynassegcm-5 irvp: 65 dynassegcm-5 pap al final de la espiracion: 35.0 mm hg  pacient cip data naix. 12.07.1962 edat 60 sexe home nass adre a cp poblacio tel.  admissio 31.05.2023 12:12 alta 16.06.2023 11:27 servei diguohmb digestologia unitat u05uhhbl data i hora d'impressio: 20.06.2023 02:53:22    pagina 2 de 4 informe alta hospitalitzacio conclusiones 1.- cateterismo cardiopulmonar con presiones cardiopulmonares elevadas (pap 32 mmhg, pcp 28 mmhg, ad 13mmhg). gasto cardiaco de 10,1 l/min). 2.- hipertension portal sinusoidal clinicamente significativa con gpvh de 15,5mmhg, no se observan comunicantes veno-venosas. - orina de 24horas: total diuresis: 1.4 l/dia. albumina 5125 mg/dia proteinuria 7.39 g/24horas orientacion diagnostica - insuficiencia renal aguda en paciente con enfermedad renal cronica en estudio paciente que en analitica de ingreso tenia cr de 1.7 mg/dl, fg 43 ml/min cuando previamente tenia filtrados habituales de 70 ml/min, cr 1.1 mg/dl. se oriento como de origen prerrenal en contexto de aumento de la dosis de diureticos (estaba con espironolactona 200mg/dia + furosemida 80mg/dia ). recibio expansion con albumina 1 g/kg (80gr) el dia 31.05 y posteriormente 2 ampollas/ 12 h ev sin mayor mejoria de la funcion renal a pesar de reexpansion con albumina por lo que se suspendio albumina que se vuelve a reiniciar el dia 8/06 y se mantiene durante otras 72h. en contexto de sobrecarga de volumen el paciente cursa con empeoramiento de disnea habitual por lo que se retira albumina y se inicia tratamiento con furosemida iv 40mg/12horas dado que podria tratarse de una erc establecida se solicita estudio de orina de 24horas en la que se observa proteinuria en rangos nefroticos por lo que ha sido valorado por nefrologia, se inicia tratamiento con espironolactona 100mg/dia y furosemida 80mg/dia via oral. al alta con creat 1.8 y fg 39 - descompensacion edemo-ascitica . cirrosis hepatica enolica child c10 meld 21 puntos al ingreso con edemas que empeoran durante la hospitalizacion debido a que se suspendieron diureticos por insuficiencia renal. se reinicia diureticos debido a sobrecarga de volumen con mejoria de los edemas en miembros inferiores y disminucion del peso. se ha intentado en 3 ocasiones realizar paracentesis diagnosticas que han sido blancas. al alta con espironolactona 100mg/dia + furosemida 80mg/dia y peso de 87kg. - insuficiencia respiratoria cronica mixta : sobrecarga de volumen + restrictivo. epoc gold ii. al ingreso disnea al reposo y desaturacion por lo que ha recibido tratamiento con broncodilatadores inhalados y o2 por vmk 26%. valorado por neumologia, el grado de epoc que presenta no justifica la disnea a minimos esfuerzos y la desaturacion. en este ingreso, se solicita gammagrafia v/ p que descarta tep cronico. se ha descartado sd hepato-pulmonar por ecocardiograma con contraste burbujas que no visualiza shunts. cateterismo cardio pulmonar que descarta sde portopulmonar . elevacion hemidiafragma izquierdo que provoca atelectasia pulmonar 2 . pdte de informe de tac toracico (06.06). tras inicio de furosemida iv el paciente presenta franca mejoria de disnea y de saturacion. al alta sat o2 93-94% al aire ambiental. seguira controles en consulta de neumologia. - fiebre sin foco. resuelto el 8/06 presento pico febril asociado a desaturacion puntual. se solicitan hemocultivos, urocultivo y se realiza paracentesis que resulta blanca. se inicia cobertura atb con ceftriaxona 1gr/dia durante 5 dias. recomendaciones al alta - dieta baja en sal  pacient cip data naix. 12.07.1962 edat 60 sexe home nass adre a cp poblacio tel.  admissio 31.05.2023 12:12 alta 16.06.2023 11:27 servei diguohmb digestologia unitat u05uhhbl data i hora d'impressio: 20.06.2023 02:53:22    pagina 3 de 4 informe alta hospitalitzacio - abstinencia absoluta del alcohol - control de peso diario - furosemida 40mg: 1-1-0 - espironolactona 100mg: 1-0-0 - lactulosa: 1 sobre al dia. suspender si diarrea - continuar con oxigeno a 3lx' para la deambulacion - resto de medicacion habitual - control y seguimiento en consulta ch intensivo (dra. martin) y en especialistas habituales tipus d'ingres: urgent motiu d'alta: alt.med.domicil metge adjunt: ,   servei: diguohmb digestologia data informe: 16.06.2023  pacient cip data naix. 12.07.1962 edat 60 sexe home nass adre a cp poblacio tel.  admissio 31.05.2023 12:12 alta 16.06.2023 11:27 servei diguohmb digestologia unitat u05uhhbl data i hora d'impressio: 20.06.2023 02:53:22    pagina 4 de 4 informe alta hospitalitzacio.
            
            Entidades extraídas [
                Diagnosis(mention="Cáncer colorrectal en estadio IV", symptoms=["fatiga extrema", "pérdida de peso significativa", "dolor abdominal recurrente"], treatment=["Quimioterapia sistémica", "Terapia dirigida con anticuerpos monoclonales", "Cuidados paliativos", "Soporte nutricional"]),
                Procedure(mention="Endoscopia", purpose=["detección de masa en el colon"]),
                Procedure(mention="Biopsia", purpose=["confirmación de adenocarcinoma"]),
                Procedure(mention="Tomografía computarizada (TC)", purpose=["detección de metástasis"]),
                Medication(mention="Quimioterapia sistémica", purpose=["reducir el tamaño del tumor y controlar la enfermedad"]),
                Medication(mention="Terapia dirigida con anticuerpos monoclonales", purpose=["bloquear el crecimiento del cáncer"]),
                LifestyleChange(mention="Soporte nutricional", purpose=["manejar la pérdida de peso y la anemia"]),
                SpecialistReferral(mention="Derivación a oncólogo", purpose=["gestión integral de la enfermedad"])
            ]
            
            Genera un resumen como este: 
            
            Paciente, de 60 años, fue ingresado el 31 de mayo de 2023 por deterioro general, somnolencia, astenia y edemas en los miembros inferiores. Diagnosticado con cirrosis hepática Child C y EPOC GOLD II, tratada con broncodilatadores y oxigenoterapia. Los análisis de sangre revelaron anemia ferropénica severa y niveles elevados de marcadores tumorales. Una endoscopia y biopsia confirmaron un adenocarcinoma colorrectal, con metástasis hepáticas y pulmonares. Se recomendó quimioterapia sistémica y terapia dirigida con anticuerpos monoclonales, además de un programa de soporte nutricional intensivo. Durante la hospitalización, desarrolló fiebre sin foco claro, tratada con ceftriaxona. Al alta, tenía una saturación de oxígeno del 93-94% y un peso de 87 kg. Se reiniciaron diuréticos y se estableció un plan de seguimiento con controles en neumología y digestología. Se recomendó una dieta baja en sal, abstinencia absoluta de alcohol, control de peso diario y continuar con oxigenoterapia durante la deambulación.
            """,
        })
    
        # Format the user input
        input_text = merch_text_keywords(text, keywords)
    
    else:
        # System prompt to guide the summarization process
        messages.append({
            "role": "system",
            "content": "Eres un modelo de resumen que recibe textos médicos en castellano o catalán. Tu tarea es resumir el texto manteniendo el contexto y redactando en español."
        })
        
        # If no keywords are provided, use the text as input
        input_text = text
        
    if schematic:
        print("Schematic summary", '*' * 10)
        # System prompt to make the summary schematic
        messages.append({
            "role": "system",
            "content": "Haz el resumen esquemático, agregando viñetas o un formato de tabla para que sea más fácil de leer. Recuerda hacerlo en español"
        })
        
    else:
        # System prompt to make the summary detailed
        messages.append({
            "role": "system",
            "content": "Haz el resumen detallado pero compacto y en un parrafo. Recuerda hacerlo en español"
        })
        
    # Append the user input to the message history
    messages.append({
        "role": "user",
        "content": input_text,
    })
    
    # Make the API call to Groq
    chat_completion = client.chat.completions.create(
        messages=messages,
        #model="llama3-70b-8192",
        model = "Gemma-7b-It"
    )
    
    # Return the response from the model
    return chat_completion.choices[0].message.content

def summarize_text(text, keywords=None, schematic=False):
    """Summarizes the text based on the provided keywords using the Groq API."""
    return get_model_response(text, keywords, schematic)

if __name__ == "__main__":
    # Test the function with a sample text and keywords
    text = """ El paciente Joan López Martínez, de 60 años, ingresó el 03-04-2024 en la Unitat de Cures Intensives del hospital, bajo la atención del Dr. García, debido a un infart agut de miocardi. Los antecedentes personales del paciente incluyen hipertensió y diabetis, mientras que en sus antecedentes familiares se destaca un infart de miocardi en el pare als 70 anys. Durante la hospitalización, se administró Aspirina con una dosis de 100 mg al dia por vía oral para el dolor y la inflamació, comenzando el 03-04-2024 y finalizando el 10-04-2024.
               El diagnóstico de Joan fue de Diabetis mellitus, con síntomas como set excessiva y orinar amb freqüència. El tratamiento incluyó insulina y una dieta específica desde el 15-05-2018. Además, se realizó un procedimiento de angioplàstia el 10-02-2023, el cual resultó en un èxit sense complicacions. Los signos vitales registrados mostraron una temperatura de 36.5°C, una frecuencia cardíaca de 72 latidos por minuto, una presión arterial de 120/80 mmHg, una frecuencia respiratoria de 16 respiraciones por minuto y una saturación de oxígeno del 98%.
               En cuanto a los resultados de laboratorio, se realizaron diversos análisis de sang el 01-06-2023, con resultados de glucosa en 90 mg/dL y creatinina en 1.2 mg/dL. También se realizó una radiografia de tòrax el 05-06-2023, la cual reveló una elevació del hemidiafragma esquerre. Al momento del alta, el 10-04-2024, se dieron recomendaciones al paciente, incluyendo una dieta baixa en sal y exercici moderat. Se programó una cita de seguimiento con el cardiòleg en 1 mes y se proporcionó el contacto del Dr. Pérez para consultas adicionales: 555-1234."""
    
    keywords = """
    ProcedimientoMedico(mención='angiopl stia', fecha='10-02-2023', resultado='complicacions'), 
    SignosVitales(temperatura=36.5, frecuencia_cardiaca=72, presion_arterial_sistolica=120, presion_arterial_diastolica=80, frecuencia_respiratoria=16, saturacion_oxigeno=98.0), 
    ResultadosLaboratorio(tipo_prueba='an lisis de sang el', resultados=['glucosa en 90 mg/dL y creatinina en 1.2 mg/dL'], fecha='01-06-2023'), 
    ImagenDiagnostica(tipo_imagen='radiografia de t rax', hallazgos='elevaci del hemidiafragma esquerre', fecha='05-06-2023'), 
    Recomendaciones(indicaciones=['dieta baixa en sal', 'exercici moderat'], citas_seguimiento=[], contactos_importantes=[])
    """

    response = get_model_response(text.replace('*', ''), keywords)
    print(response)
