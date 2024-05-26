
from dataclasses import dataclass, field
from typing import List
from GoLLIE_MED.src.tasks.utils_typing import Generic as Template

"""
ENTITY DEFINITION
"""

@dataclass
class Medication:
    """Refers to a drug or substance used to diagnose, cure, treat, or prevent diseases.
    Medications can be administered in various forms and doses and are crucial for managing
    patients' health conditions. They can be classified based on their therapeutic use,
    mechanism of action, or chemical characteristics."""
    
    mention: str  # The name of the medication. Examples: "Aspirin"
    dose: str  # The amount and frequency of the prescribed medication. Examples: "100 mg daily"
    route: str  # The method of administering the medication. Examples: "oral"
    purpose: List[str]  # List of reasons or conditions for which the medication is prescribed. Examples: ["pain", "inflammation"]
    start_date: str  # The date when the medication was started. Examples: "01-01-2023"
    end_date: str  # The date when the medication was discontinued, if applicable. Examples: "31-01-2023"

@dataclass
class Disease:
    """Refers to a health condition or illness that affects the normal functioning of the body.
    Diseases can be caused by various factors, such as infections, genetic disorders, lifestyle choices,
    or environmental factors. They can affect different body systems and have varying degrees of severity."""
    
    mention: str  # The name of the disease or health condition. Examples: "Diabetes mellitus"
    symptoms: List[str]  # List of signs or symptoms associated with the disease. Examples: ["excessive thirst", "frequent urination"]
    treatment: List[str]  # List of treatments or interventions used to manage the disease. Examples: ["insulin", "diet"]
    diagnosis_date: str  # The date when the disease was diagnosed. Examples: "15-05-2018"
    severity: str  # The severity level of the disease. Examples: "chronic"

@dataclass
class MedicalProcedure:
    """Refers to medical interventions performed to diagnose or treat diseases.
    This can include surgeries, diagnostic tests, and other specialized treatments."""
    
    mention: str  # The name of the medical procedure. Examples: "angioplasty"
    date: str  # The date when the procedure was performed. Examples: "10-02-2023"
    outcome: str  # The result or conclusion of the procedure. Examples: "successful without complications"

@dataclass
class HospitalizationData:
    """Refers to information related to a patient's hospitalization, including the admission date,
    discharge date, and reason for hospitalization. Hospitalization data is essential for tracking
    the patient's health status, treatment progress, and healthcare resource utilization."""
    
    admission_date: str  # The date when the patient was admitted to the hospital. Examples: "03-04-2024"
    discharge_date: str  # The date when the patient was discharged from the hospital. Examples: "10-04-2024"
    reason: str  # The reason or cause of the patient's hospitalization. Examples: "acute myocardial infarction"
    unit: str  # The hospital unit or department where the patient was admitted. Examples: "Intensive Care Unit"
    responsible_physician: str  # The name of the physician responsible for the patient during hospitalization. Examples: "Dr. Garcia"

@dataclass
class PatientData:
    """Refers to information related to a patient's medical history, including name, age, and urgency.
    Patient data is essential for healthcare providers to deliver appropriate care and make informed
    decisions about patient management."""
    
    name: str  # The patient's name. Examples: "Juan Lopez Martinez"
    age: int  # The patient's age. Examples: 60
    urgency: str  # The urgency level of the patient's condition. Examples: "acute chest pain"
    sex: str  # The patient's sex. Examples: "male"
    birth_date: str  # The patient's birth date. Examples: "01-01-1964"
    personal_history: List[str]  # List of relevant personal medical history. Examples: ["hypertension", "diabetes"]
    family_history: List[str]  # List of relevant family medical history. Examples: ["father had myocardial infarction at 70"]

@dataclass
class VitalSigns:
    """Refers to measurements of the body's basic functions that are essential for life.
    Vital signs include body temperature, heart rate, blood pressure, respiratory rate, and oxygen saturation."""
    
    temperature: float  # The patient's body temperature. Examples: 36.5
    heart_rate: int  # The number of heartbeats per minute. Examples: 72
    systolic_bp: int  # The blood pressure in the arteries when the heart beats. Examples: 120
    diastolic_bp: int  # The blood pressure in the arteries between heartbeats. Examples: 80
    respiratory_rate: int  # The number of breaths per minute. Examples: 16
    oxygen_saturation: float  # The percentage of oxygen in the blood. Examples: 98.0

@dataclass
class LaboratoryResults:
    """Refers to the results of laboratory tests performed during the patient's hospitalization.
    These tests can include blood tests, urine tests, and other clinical studies."""
    
    test_type: str  # The type of laboratory test performed. Examples: "blood test"
    results: List[str]  # Specific results of the test. Examples: ["glucose: 90 mg/dL", "creatinine: 1.2 mg/dL"]
    date: str  # The date when the tests were performed. Examples: "01-06-2023"

@dataclass
class DiagnosticImaging:
    """Refers to imaging studies performed to diagnose or monitor health conditions.
    These studies can include X-rays, CT scans, MRIs, among others."""
    
    image_type: str  # The type of imaging study. Examples: "chest X-ray"
    findings: str  # The findings or conclusions of the imaging study. Examples: "elevation of left hemidiaphragm"
    date: str  # The date when the imaging study was performed. Examples: "05-06-2023"

@dataclass
class Recommendations:
    """Refers to the suggestions and guidelines provided to the patient upon discharge to improve their
    health and prevent future episodes. This can include lifestyle changes, medications, and follow-up appointments."""
    
    instructions: List[str]  # List of recommendations provided to the patient. Examples: ["low-salt diet", "moderate exercise"]
    follow_up_appointments: List[str]  # List of scheduled follow-up appointments for the patient. Examples: ["appointment with cardiologist in 1 month"]
  
ENTITY_DEFINITIONS: List[type] = [
    Medication,
    Disease,
    MedicalProcedure,
    HospitalizationData,
    PatientData,
    VitalSigns,
    LaboratoryResults,
    DiagnosticImaging,
    Recommendations,
]
