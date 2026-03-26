#!/usr/bin/env python3
"""Generate bulk seed bank records — 500 realistic training pairs."""
import random
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from harvest import save_seed, count_seeds

NAMES_M = ['Ahmed','Mohammad','Abdullah','Khaled','Fahad','Yousef','Ali','Hussein','Hassan','Omar','Ibrahim','Salman','Nasser','Jaber','Mubarak','Faisal','Salem','Nawaf','Turki','Saad','Majed','Waleed','Hani','Rashid','Mansour','Hamad','Jasem','Adel','Zaid','Essa','Anas','Ziad','Mazen','Hesham','Mustafa','Khalil','Sami','Sultan','Nayef','Tariq','Qasim','Akram','Hazem','Nabeel','Kareem','Ghassan','Bahaa','Bassem','Safwan','Fuad','Hilal','Zaki','Jawad','Hamza','Soliman']
NAMES_F = ['Fatima','Noura','Mariam','Sara','Haya','Dalal','Munira','Aisha','Reem','Dana','Lulwa','Latifa','Sheikha','Jawaher','Amira','Amal','Hind','Manal','Samira','Nadia','Salwa','Haifa','Zainab','Khadija','Mona','Hessa','Abeer','Ghada','Shahd','Malak','Layan','Yara','Nouf','Noor','Maha','Hala','Yasmin','Asma','Eman','Hanan','Dina','Rana','Lamia','Razan','Hadeel','Lujain','Raghad','Tamara','Huda']
FAMILY = ['Al-Mutairi','Al-Enezi','Al-Shammari','Al-Rashidi','Al-Ajmi','Al-Dosari','Al-Kandari','Al-Otaibi','Al-Harbi','Al-Hajri','Al-Fadli','Al-Bloushi','Al-Saleh','Al-Ghanem','Al-Roumi','Al-Badr','Al-Khaled','Al-Mubarak','Al-Jassem','Al-Ibrahim','Al-Sabah','Al-Hamad','Al-Fahad','Al-Salem','Al-Awadhi','Al-Qatami','Behbehani','Al-Tabtabaei','Al-Marzouk','Al-Sarraf','Al-Turki','Al-Mansour','Al-Zaid','Al-Rashed','Al-Sahli']

DX_COMBOS = [
    'NSTEMI, DM2, HTN','STEMI, DM2','ACS, HTN','AF, CHF, CKD3','ADHF, AF, DM2',
    'DVT, PE','CAD, DM2, HTN, CKD3','CAP, COPD','AECOPD, DM2','Pneumonia, Sepsis',
    'Chest infection','ARDS, Sepsis','AKI, Sepsis, DM2','CKD5, HTN','Hypernatremia, AKI',
    'Urosepsis, AKI','UTI, DM2','CVA, AF, HTN','CVA left MCA occlusion, AF',
    'TIA, HTN, DM2','Seizure, DM2','Meningitis','UGIB, Liver cirrhosis',
    'SBO','Acute pancreatitis, DM2','Cholangitis, Sepsis','DKA, T1DM','HHS, DM2',
    'Sepsis, UTI, DM2','Cellulitis, DM2','COVID, Pneumonia','Post-op Hip fracture',
    'Polytrauma','Burns 30% BSA','Chest infection and UTI','LVF exacerbation, CKD4',
    'Fall, Hip fracture','Fever, PUO','Syncope, AF','Dehydration, AKI',
    'COPD, CAP','GI bleed, CKD3','Hepatic encephalopathy, CLD',
    'SBP, Ascites, CLD','Febrile neutropenia','DIC, Sepsis','Fracture R femur',
    'Status epilepticus','SAH','ICH','Cholecystitis','Appendicitis',
    'Anaphylaxis','Hyperkalemia, CKD5','Hyponatremia, CHF',
    'Rhabdomyolysis, AKI','Necrotizing fasciitis','PE, DVT','Pyelonephritis',
    'Pneumonia, COPD, DM2','Septic shock, Urosepsis','NSTEMI, CKD4, DM2',
    'CVA, DM2, HTN, AF','Chest pain, ?ACS','Acute kidney injury on CKD3',
    'Bilateral pneumonia, COVID','Upper GI bleed, Liver cirrhosis',
    'Diabetic foot, Cellulitis, DM2','Pressure ulcer, Malnutrition',
    'Smoke inhalation, Burns','Blast injury, Polytrauma',
    'Bronchiolitis','Croup','Febrile seizure','Kawasaki disease',
    'Pyloric stenosis','NEC','RDS','BPD','PDA',
    'Glaucoma','Retinal detachment','Tonsillitis','Epistaxis',
    'BPH, Urinary retention','Testicular torsion','Placenta previa',
    'Pre-eclampsia','Ectopic pregnancy','Ovarian torsion',
    'SCC, Post-op','Melanoma, Mets','Psoriasis flare',
    'ORIF R tibia','Lap chole, Gallstones','TURP, BPH',
    'Craniotomy, SDH','Tracheostomy, Prolonged intubation',
]

MEDS_POOL = [
    'Aspirin','Clopidogrel','Ticagrelor','Enoxaparin','Heparin','Warfarin','Apixaban',
    'Metformin','Insulin Glargine','Insulin Aspart','Amlodipine','Ramipril','Losartan',
    'Bisoprolol','Carvedilol','Furosemide','Spironolactone','Atorvastatin','Omeprazole',
    'Pantoprazole','Ceftriaxone','Meropenem','Vancomycin','Piperacillin-Tazobactam',
    'Azithromycin','Ciprofloxacin','Metronidazole','Paracetamol','Morphine','Fentanyl',
    'Salbutamol','Prednisolone','Dexamethasone','Levetiracetam','Lactulose','Ondansetron',
    'Noradrenaline','Empagliflozin','NS 0.9%','KCl','Albumin','Diazepam','Midazolam',
    'Propofol','Rocuronium','Dopamine','Dobutamine','Amiodarone','Digoxin',
    'Cefuroxime','Fluconazole','Acyclovir','Linezolid','Colistin',
    'Hydrocortisone','Methylprednisolone','Budesonide','Ipratropium',
    'Tramadol','Gabapentin','Pregabalin','Ketamine','Naloxone',
    'Levothyroxine','Semaglutide','Sitagliptin','Gliclazide',
    'Rosuvastatin','Ezetimibe','Fenofibrate',
]

BEDS = ([f'E-M-{i:02d}' for i in range(1,26)] +
        [f'E-F-{i:02d}' for i in range(1,26)] +
        [f'{i}-{j}' for i in range(1,21) for j in range(1,5)] +
        [f'ICU-{i}' for i in range(1,13)] +
        [f'CCU-{i}' for i in range(1,7)] +
        [f'A{i}' for i in range(1,21)] +
        [f'B{i}' for i in range(1,21)])

WARDS = ['Med-1','Med-2','Med-3','Med-4','Surg-1','Surg-2','ICU','CCU','HDU','ER',
         'Ward 19','Ward 20','Ward 21','Ward 22','Ward 23','NICU','PICU','Ortho','Neuro']
TRIAGE = ['RED','YELLOW','GREEN','GRAY','BLACK']
MOBILITY = ['AMBULATORY','WHEELCHAIR','STRETCHER','CRITICAL_TRANSPORT']
O2 = ['NONE','NASAL_CANNULA','FACE_MASK','NON_REBREATHER','BIPAP','VENTILATOR']
ISO = ['NONE','CONTACT','DROPLET','AIRBORNE']
CODE = ['FULL','DNR','COMFORT']
BLOOD = ['A+','A-','B+','B-','AB+','AB-','O+','O-']
DOCTORS = ['Dr. Nasser','Dr. Fahad','Dr. Salem','Dr. Hani','Dr. Ibrahim','Dr. Rashid',
           'Dr. Ali','Dr. Turki','Dr. Noura','Dr. Sara','Bader','Noura','Saleh','Zahra']
STATUS = ['New','Active','Chronic','Stable','Improving','Deteriorating','For discharge',
          'NBM','Day 1','Day 2','Day 3','Day 5','Day 7','ICU discharge','Transfer',
          'For OT','For ERCP','Pending','Awaiting']
ALLERGIES = ['NKDA','Penicillin','Sulfa','NSAID','Codeine','Morphine','Aspirin',
             'Iodine','Latex','Vancomycin','PCN','Ciprofloxacin']

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 500

print(f'Generating {TARGET} seed bank records...')
for i in range(TARGET):
    g = random.choice(['M', 'F'])
    first = random.choice(NAMES_M if g == 'M' else NAMES_F)
    family = random.choice(FAMILY)
    name = f'{first} {family}'
    age = random.randint(1, 98)
    bed = random.choice(BEDS)
    ward = random.choice(WARDS)
    dx = random.choice(DX_COMBOS)
    meds = ', '.join(random.sample(MEDS_POOL, random.randint(1, 5)))
    triage = random.choices(TRIAGE, weights=[15, 30, 35, 10, 10])[0]
    mob = random.choice(MOBILITY)
    o2 = random.choice(O2)
    iso = random.choice(ISO)
    code = random.choice(CODE)
    blood = random.choice(BLOOD)
    doc = random.choice(DOCTORS)
    status = random.choice(STATUS)
    allergy = random.choice(ALLERGIES)

    input_text = f'{bed}\t{name}\t{age}/{g}\t{dx}\t{meds}'
    structured = {
        'fullName': name, 'bed': bed, 'age': age, 'gender': g,
        'dx': dx, 'meds': meds, 'triage': triage, 'mobility': mob,
        'o2': o2, 'iso': iso, 'code': code, 'bloodType': blood,
        'ward': ward, 'assignedDoctor': doc, 'sheetStatus': status,
        'allergies': allergy,
    }
    save_seed(input_text, structured, {'source': 'synthetic', 'batch': f'bulk_{TARGET}'})

    if (i + 1) % 100 == 0:
        print(f'  {i + 1}/{TARGET} records generated')

print(f'Done. Seed bank now has {count_seeds()} total records.')
