#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════════════════
// SHIFU LIVE FEEDER — Watch Shifu Learn in Real-Time
// ═══════════════════════════════════════════════════════════════════════
//
// Feeds Shifu across all domains with massive corpora, showing
// live progress bars, accuracy metrics, and phase promotions.
//
// Usage: node teaching/feed_live.js [--fast] [--domain medical]

const fs = require('fs');
const path = require('path');
const { ShifuEngine } = require('../core/engine');
const { createTeacher } = require('./index');

// ─── CLI args ───────────────────────────────────────────────────────
const args = process.argv.slice(2);
const FAST_MODE = args.includes('--fast');
const DOMAIN_FILTER = args.find((a, i) => args[i - 1] === '--domain') || null;
const DELAY = FAST_MODE ? 0 : 15; // ms between sentences

// ─── Colors & Display ───────────────────────────────────────────────
const C = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  red: '\x1b[31m',
  white: '\x1b[37m',
  bgGreen: '\x1b[42m',
  bgBlue: '\x1b[44m',
  bgMagenta: '\x1b[45m',
  bgYellow: '\x1b[43m',
};

function bar(pct, width = 30) {
  const clamped = Math.max(0, Math.min(1, pct));
  const filled = Math.round(clamped * width);
  const empty = width - filled;
  const filledStr = '\u2588'.repeat(filled);
  const emptyStr = '\u2591'.repeat(empty);
  return `${C.green}${filledStr}${C.dim}${emptyStr}${C.reset}`;
}

function pctStr(val) {
  const p = (val * 100).toFixed(1);
  if (val >= 0.8) return `${C.green}${p}%${C.reset}`;
  if (val >= 0.5) return `${C.yellow}${p}%${C.reset}`;
  return `${C.red}${p}%${C.reset}`;
}

function numStr(n) {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return String(n);
}

function sleep(ms) {
  if (ms <= 0) return Promise.resolve();
  return new Promise(r => setTimeout(r, ms));
}

function clearLines(n) {
  for (let i = 0; i < n; i++) {
    process.stdout.write('\x1b[1A\x1b[2K');
  }
}

// ─── Massive Cross-Domain Corpus ────────────────────────────────────

const CORPORA = {
  medical: [
    // Neurology
    "Patient presents with acute onset left sided weakness and slurred speech consistent with stroke.",
    "CT head showed no evidence of hemorrhage or midline shift in the brain parenchyma.",
    "MRI brain revealed acute ischemic infarct in the left middle cerebral artery territory.",
    "Patient was started on aspirin and clopidogrel for secondary stroke prevention therapy.",
    "Neurology team reviewed the patient and recommended thrombolysis with alteplase.",
    "Doctor prescribed levetiracetam for seizure prophylaxis after the cerebrovascular accident.",
    "EEG showed generalized slowing consistent with diffuse encephalopathy requiring monitoring.",
    "Lumbar puncture was performed and cerebrospinal fluid sent for culture and sensitivity.",
    "Doctor diagnosed bacterial meningitis and started empirical intravenous antibiotics immediately.",
    "Patient developed new onset seizures requiring intravenous phenytoin loading dose.",
    // Cardiology
    "Troponin I elevated at 2.4 ng/mL consistent with non-ST elevation myocardial infarction.",
    "ECG showed ST elevation in leads V1 through V4 indicating anterior wall MI.",
    "Patient started on dual antiplatelet therapy with aspirin and ticagrelor loading doses.",
    "Heparin drip initiated at 18 units per kilogram per hour for anticoagulation.",
    "Echocardiogram showed ejection fraction of 35 percent with global hypokinesis.",
    "Cardiology consulted for urgent cardiac catheterization and possible intervention.",
    "Patient with atrial fibrillation started on apixaban for stroke prophylaxis.",
    "Beta blocker metoprolol tartrate 25mg twice daily prescribed for rate control.",
    "BNP elevated at 1200 pg/mL consistent with acute decompensated heart failure.",
    "Furosemide 40mg intravenous administered for pulmonary edema and volume overload.",
    // Emergency
    "Patient arrived by ambulance with Glasgow Coma Scale score of 8 requiring intubation.",
    "Trauma team activated for high speed motor vehicle collision with chest trauma.",
    "FAST exam positive in right upper quadrant suggesting intraperitoneal free fluid.",
    "Chest tube inserted for large left sided pneumothorax with respiratory compromise.",
    "Patient presented with anaphylaxis after bee sting treated with intramuscular epinephrine.",
    "Blood gas showed pH 7.21 pCO2 58 pO2 65 indicating acute respiratory acidosis.",
    "Massive transfusion protocol activated for hemorrhagic shock with hemoglobin 5.2.",
    "Central venous catheter placed in right internal jugular vein under ultrasound guidance.",
    // Pharmacy
    "Metformin 500mg twice daily prescribed for newly diagnosed type 2 diabetes mellitus.",
    "Insulin glargine 20 units subcutaneous at bedtime for basal glycemic control.",
    "Vancomycin trough level checked at 15.2 mcg/mL within therapeutic range.",
    "Warfarin dose adjusted to 5mg daily to achieve target INR of 2.0 to 3.0.",
    "Gentamicin peak and trough levels ordered to guide aminoglycoside dosing.",
    "Prednisone taper started at 60mg daily decreasing by 10mg every five days.",
    "Omeprazole 40mg daily prescribed for gastroesophageal reflux disease with erosive esophagitis.",
    "Lisinopril 10mg daily started for hypertension and diabetic nephropathy protection.",
    // Surgery
    "Patient underwent laparoscopic cholecystectomy for symptomatic cholelithiasis without complications.",
    "Post-operative wound inspection showed clean dry incision with no signs of infection.",
    "Surgeon performed emergency appendectomy for perforated appendicitis with peritonitis.",
    "Jackson-Pratt drain output measured at 150mL serosanguinous fluid over 24 hours.",
    "Patient tolerated clear liquid diet on post-operative day one without nausea.",
    "Prophylactic antibiotics given within 60 minutes before surgical incision per protocol.",
    // Nursing
    "Vital signs recorded: blood pressure 135/82 pulse 78 temperature 37.1 oxygen saturation 97%.",
    "Foley catheter output 450mL clear yellow urine over the past 8 hours.",
    "Patient ambulated 50 feet in the hallway with physical therapy assistance.",
    "Wound care performed with normal saline irrigation and sterile dressing applied.",
    "Fall risk assessment score of 45 indicating high risk requiring bed alarm activation.",
    "Pain assessed at 6 out of 10 on visual analog scale; morphine 4mg IV administered.",
    // Pediatrics
    "Infant presented with fever of 39.5 degrees and irritability requiring septic workup.",
    "Pediatric patient started on amoxicillin 90mg/kg/day for acute otitis media.",
    "Growth chart shows weight below the 5th percentile for age concerning for failure to thrive.",
    "Immunization schedule updated with MMR and varicella vaccines at 12 month visit.",
    // Radiology
    "CT abdomen and pelvis with contrast showed 3cm mass in the right hepatic lobe.",
    "Chest X-ray revealed bilateral pleural effusions greater on the left side.",
    "Ultrasound of the right upper quadrant demonstrated gallstones without wall thickening.",
    "MRI lumbar spine showed L4-L5 disc herniation with moderate neural foraminal stenosis.",
    // Oncology
    "Patient diagnosed with stage IIIA non-small cell lung cancer with mediastinal lymphadenopathy.",
    "Chemotherapy with carboplatin and paclitaxel scheduled every 21 days for 4 cycles.",
    "PET scan showed increased FDG uptake in the left axillary lymph nodes SUV 8.2.",
    "Radiation therapy planned with 60 Gray in 30 fractions to the primary tumor bed.",
    "Tumor board recommended immunotherapy with pembrolizumab based on PD-L1 expression.",
    // Laboratory
    "Complete blood count showed WBC 12.5 hemoglobin 10.2 platelets 180 thousand.",
    "Basic metabolic panel: sodium 138 potassium 4.2 chloride 102 bicarbonate 24 BUN 18 creatinine 1.1.",
    "Liver function tests elevated with ALT 120 AST 95 alkaline phosphatase 210 total bilirubin 2.1.",
    "Coagulation studies: PT 14.2 seconds INR 1.1 aPTT 32 seconds within normal limits.",
    "Urinalysis showed positive leukocyte esterase and nitrites suggesting urinary tract infection.",
    "HbA1c level of 8.2% indicating poorly controlled diabetes over the past 3 months.",
    "Blood cultures grew gram positive cocci in clusters consistent with Staphylococcus aureus.",
    "Procalcitonin level 5.8 ng/mL suggesting bacterial infection requiring antibiotic therapy.",
  ],

  legal: [
    "The plaintiff alleges breach of contract under Section 2-207 of the Uniform Commercial Code.",
    "Defendant filed a motion to dismiss for failure to state a claim upon which relief can be granted.",
    "The court granted summary judgment in favor of the respondent on all remaining counts.",
    "Counsel submitted a memorandum of law in support of the motion for preliminary injunction.",
    "Pursuant to Rule 12(b)(6) of the Federal Rules of Civil Procedure the complaint is deficient.",
    "The arbitration clause in paragraph 14.3 of the agreement governs all dispute resolution.",
    "Witness testified under oath regarding the chain of custody of the physical evidence.",
    "The statute of limitations for personal injury tort claims is three years from date of discovery.",
    "Court ordered production of all documents responsive to plaintiff's interrogatories.",
    "The jury returned a verdict of not guilty on all counts of the indictment.",
    "Defense counsel filed a motion in limine to exclude expert testimony on damages.",
    "The appellate court reversed the lower court's decision on constitutional grounds.",
    "Plaintiff seeks compensatory damages in the amount of five million dollars.",
    "The non-compete agreement is enforceable for a period of two years within the jurisdiction.",
    "Deposition of the corporate representative scheduled for the 15th of next month.",
    "The settlement conference resulted in a binding agreement between all parties.",
    "Judge sustained the objection on grounds of hearsay under Rule 802.",
    "The lease agreement contains a force majeure clause covering pandemic-related disruptions.",
    "Attorney filed a notice of appeal within the statutory thirty-day deadline.",
    "The court appointed a special master to oversee the discovery dispute resolution.",
    "Intellectual property rights under the patent expire in seventeen years from date of filing.",
    "Due diligence review revealed material undisclosed liabilities in the acquisition target.",
    "The court issued a temporary restraining order prohibiting defendant from contacting plaintiff.",
    "Cross-examination revealed inconsistencies in the witness testimony regarding the timeline.",
    "The merger requires approval from the Federal Trade Commission under antitrust regulations.",
    "Plaintiff's expert witness provided testimony on the standard of care in medical malpractice.",
    "The class action certification was granted for all consumers affected by the data breach.",
    "Defense argued that the contract was voidable due to fraudulent misrepresentation.",
    "The court found that defendant acted with gross negligence resulting in punitive damages.",
    "Summary of the opinion: the regulation does not violate the First Amendment protections.",
  ],

  financial: [
    "Quarterly revenue increased 12 percent year-over-year reaching 4.2 billion dollars.",
    "The Federal Reserve raised interest rates by 25 basis points at the latest meeting.",
    "Net income attributable to common shareholders was 1.8 billion for the fiscal quarter.",
    "Operating expenses decreased 8 percent due to restructuring charges of 500 million.",
    "Earnings per share came in at 2.15 dollars beating consensus estimates by 12 cents.",
    "The portfolio allocation shifted toward investment-grade fixed-income securities.",
    "Accounts receivable turnover ratio improved to 8.2 times from 7.1 in the prior year.",
    "Depreciation and amortization totaled 340 million dollars for the reporting period.",
    "EBITDA margin expanded 200 basis points to 28.5 percent reflecting operational efficiency.",
    "Total assets under management reached 12.8 billion dollars a record for the firm.",
    "The dividend yield currently stands at 3.2 percent with quarterly payments of 0.85 per share.",
    "Free cash flow generation of 2.1 billion supported share repurchase program of 1 billion.",
    "Cost of goods sold decreased 5 percent due to favorable commodity pricing and hedging.",
    "The company reported a debt-to-equity ratio of 0.45 indicating conservative leverage.",
    "Capital expenditures of 800 million were directed toward new manufacturing facilities.",
    "Revenue recognition under ASC 606 requires performance obligation analysis at contract inception.",
    "The audit committee reviewed internal controls over financial reporting without material weakness.",
    "Gross profit margin expanded to 62 percent from 58 percent in the prior year period.",
    "The company declared a special dividend of 2.50 per share payable to record holders.",
    "Foreign currency translation adjustments resulted in a loss of 45 million in other comprehensive income.",
    "Working capital improved to 3.2 billion providing ample liquidity for operations.",
    "Goodwill impairment testing indicated no write-down was necessary for the reporting unit.",
    "The bond offering of 750 million at 4.25 percent was oversubscribed by institutional investors.",
    "Return on invested capital reached 18.5 percent exceeding the weighted average cost of capital.",
    "Tax provision reflected an effective rate of 21 percent consistent with federal statutory rate.",
    "Segment reporting showed technology services contributed 65 percent of consolidated revenue.",
    "Inventory levels decreased 12 percent reflecting improved supply chain management.",
    "The company maintained investment grade credit rating of BBB-plus with stable outlook.",
    "Share repurchase program authorized additional 2 billion dollars for the fiscal year.",
    "Cash and cash equivalents totaled 5.6 billion at the end of the quarter.",
  ],

  scientific: [
    "The experiment demonstrated a statistically significant correlation with p-value below 0.001.",
    "Spectroscopic analysis revealed absorption peaks at 254 nanometers and 380 nanometers.",
    "The catalyst increased reaction yield from 45 percent to 92 percent under mild conditions.",
    "Genome sequencing identified three novel single nucleotide polymorphisms in the target region.",
    "The thermal conductivity of the composite material was measured at 0.35 watts per meter kelvin.",
    "Results were reproducible across five independent trials with standard deviation below 0.05.",
    "CRISPR-Cas9 gene editing achieved 98 percent knockout efficiency in the target cell line.",
    "The nanoparticle diameter was 42 nanometers with a polydispersity index of 0.08.",
    "Density functional theory calculations predicted a band gap of 1.8 electron volts.",
    "The protein crystal structure was resolved at 1.9 angstroms using X-ray diffraction.",
    "Mass spectrometry confirmed the molecular weight of the synthesized compound at 342.4 daltons.",
    "Flow cytometry analysis showed 78 percent of cells expressed the surface marker CD34.",
    "The superconductor exhibited zero resistance below 39 kelvin at ambient pressure.",
    "PCR amplification produced a single band at 650 base pairs confirming successful cloning.",
    "The enzyme exhibited Michaelis-Menten kinetics with a Km of 12 micromolar.",
    "Electron microscopy revealed hexagonal lattice structure with 0.34 nanometer spacing.",
    "The solar cell achieved 24.5 percent power conversion efficiency under standard conditions.",
    "Western blot analysis confirmed upregulation of the target protein by 3.2 fold.",
    "Monte Carlo simulation converged after 10 million iterations with 95 percent confidence interval.",
    "The acoustic impedance mismatch at the tissue interface was measured at 1.5 megaRayleigh.",
    "Fourier transform infrared spectroscopy identified hydroxyl and carbonyl functional groups.",
    "The alloy exhibited a tensile strength of 1200 megapascals with 15 percent elongation.",
    "Cell viability assay showed 95 percent survival after 24 hours of treatment.",
    "The quantum dot fluorescence peaked at 620 nanometers with full width at half maximum of 25 nm.",
    "Polymerase chain reaction cycle threshold values ranged from 18 to 24 across all samples.",
  ],

  engineering: [
    "The microcontroller operates at 3.3 volts with a clock frequency of 168 megahertz.",
    "Tensile strength of the aluminum alloy exceeded 450 megapascals at room temperature.",
    "The PID control loop uses gains Kp equals 2.5, Ki equals 0.8, and Kd equals 0.3.",
    "CAD model exported as STEP file format for CNC machining verification and toolpath generation.",
    "Load testing revealed maximum deflection of 2.3 millimeters under 500 newton applied force.",
    "The firmware update resolved the I2C communication timeout issue on the sensor bus.",
    "Power supply delivers 12 volts at 5 amperes with less than 50 millivolt ripple.",
    "The MOSFET switching frequency was set to 100 kilohertz to minimize switching losses.",
    "Finite element analysis showed maximum von Mises stress of 280 megapascals at the fillet.",
    "The bearing has a rated dynamic load capacity of 25 kilonewtons and L10 life of 50000 hours.",
    "TCP/IP socket connection established on port 8080 with 100 millisecond timeout parameter.",
    "The servo motor provides 5 newton-meters of continuous torque at 3000 revolutions per minute.",
    "Signal-to-noise ratio measured at 45 decibels exceeding the minimum requirement of 35 dB.",
    "The heat exchanger transfers 50 kilowatts with an overall coefficient of 500 W per m2K.",
    "Memory allocation optimized to reduce heap fragmentation by 40 percent in embedded system.",
    "The gear ratio of 3.5 to 1 provides the required torque multiplication for the drive system.",
    "Vibration analysis detected a resonance frequency at 1250 hertz in the shaft assembly.",
    "The printed circuit board has 6 copper layers with minimum trace width of 0.15 millimeters.",
    "Database query execution time reduced from 2.3 seconds to 45 milliseconds after indexing.",
    "The hydraulic actuator provides 100 kilonewtons of force at 200 bar operating pressure.",
    "Latency measured at 12 microseconds for the real-time data acquisition system.",
    "The optical fiber has a core diameter of 62.5 micrometers with numerical aperture 0.275.",
    "Compiler optimization reduced binary size by 23 percent with O2 flag enabled.",
    "The stepper motor provides 200 steps per revolution with 1.8 degree step angle.",
    "Battery capacity rated at 5000 milliamp hours with charge cycle count of 500 to 80 percent.",
  ],

  education: [
    "Students demonstrated significant improvement in reading comprehension scores after intervention.",
    "The curriculum framework aligns with state standards for mathematics at the fourth grade level.",
    "Formative assessment data shows 75 percent of students meeting the learning objective.",
    "Teacher implemented differentiated instruction strategies for diverse learning needs.",
    "The semester enrollment reached 15000 students across all undergraduate programs.",
    "Research-based literacy instruction improved phonemic awareness in kindergarten students.",
    "Graduate teaching assistants received training in inclusive classroom practices.",
    "The online learning platform supports asynchronous and synchronous instructional modalities.",
    "Student achievement gap decreased by 8 percentage points after implementing tutoring program.",
    "Faculty senate approved new general education requirements for all degree programs.",
    "Standardized test results indicate proficiency rates of 68 percent in mathematics.",
    "The special education individualized program sets measurable annual goals for each student.",
    "Professional development workshop focused on culturally responsive teaching methods.",
    "The university accreditation review committee evaluated program learning outcomes.",
    "Classroom observation rubric measured teacher effectiveness across five instructional domains.",
    "Student engagement survey results showed 82 percent satisfaction with course content.",
    "The school district implemented restorative justice practices to reduce suspension rates.",
    "Dual enrollment program allows high school juniors to earn college credits simultaneously.",
    "Educational technology integration improved student collaboration on group projects.",
    "The doctoral dissertation defense committee approved the candidate's research methodology.",
  ],

  government: [
    "The census bureau reported a population increase of 2.3 percent in the metropolitan area.",
    "Building permit applications must be submitted to the municipal planning department.",
    "The federal regulation requires environmental impact assessment for all major projects.",
    "Voter registration deadline is 30 days before the general election per state statute.",
    "The agency processed 45000 immigration applications during the current fiscal year.",
    "Legislative session convened with 120 bills introduced for committee consideration.",
    "The department of transportation awarded a 50 million dollar contract for highway repairs.",
    "Emergency management agency issued evacuation orders for three coastal communities.",
    "Tax revenue collections exceeded projections by 8 percent for the fiscal quarter.",
    "The social services division approved 2300 applications for supplemental assistance.",
    "Municipal court scheduled 150 cases for the upcoming docket this calendar month.",
    "The zoning commission approved the variance request for commercial development.",
    "Freedom of information requests must be processed within 20 business days.",
    "The inspector general's office completed 45 audits of federal program expenditures.",
    "Public comment period for the proposed regulation closes on the 15th of next month.",
    "The defense department budget request totaled 850 billion dollars for the fiscal year.",
    "City council voted unanimously to approve the infrastructure improvement bond measure.",
    "The environmental protection agency established new emission standards for power plants.",
    "Veterans affairs department expanded telehealth services to rural communities.",
    "The national park service managed 423 sites with 312 million annual visitor visits.",
  ],

  retail: [
    "Invoice number 2024-45678 totaling 3450.99 dollars was processed for payment.",
    "The product SKU A2B-9012 is currently out of stock with estimated restock date next week.",
    "Customer order shipped via express delivery with tracking number 1Z999AA10123456784.",
    "Inventory audit revealed a discrepancy of 45 units in the warehouse management system.",
    "The promotional discount of 25 percent applies to all items in the clearance category.",
    "Point of sale system processed 1200 transactions during the holiday weekend period.",
    "Return authorization number RA-2024-789 issued for defective merchandise.",
    "Supplier purchase order PO-56789 for 500 units at 12.50 per unit submitted.",
    "The barcode scanner reads UPC format with 12 digit product identification numbers.",
    "Customer loyalty program members earned 2500 reward points on their latest purchase.",
    "Gross merchandise value reached 2.8 million dollars for the e-commerce platform.",
    "Fulfillment center processed 8000 orders within the guaranteed two-day shipping window.",
    "Price adjustment of 15 percent markdown applied to seasonal inventory clearance.",
    "The vendor management system tracks 350 active suppliers across 12 product categories.",
    "Shopping cart abandonment rate decreased to 62 percent after checkout optimization.",
    "Warehouse receiving dock processed 45 inbound shipments containing 12000 units.",
    "Product listing optimization improved search ranking by 35 positions on marketplace.",
    "Customer satisfaction score averaged 4.6 out of 5.0 based on 2500 post-purchase reviews.",
    "The subscription box service retained 78 percent of customers after the first three months.",
    "Gift card redemption volume reached 450000 dollars during the quarter.",
  ],

  logistics: [
    "Container vessel MSC Diana departed Shanghai port with 8500 twenty-foot equivalent units.",
    "Freight manifest shows 45 pallets weighing 22500 kilograms destined for distribution center.",
    "Customs clearance for shipment HBL-2024-78901 completed at the port of entry.",
    "Fleet management system reported average fuel consumption of 8.2 liters per 100 kilometers.",
    "The warehouse management system allocated storage location A-12-C-045 for incoming goods.",
    "Last mile delivery achieved 96.5 percent on-time performance for the reporting period.",
    "Air cargo shipment AWB 123-45678901 cleared security screening at the origin airport.",
    "Cross-docking facility processed 200 inbound trailers and 180 outbound shipments daily.",
    "The refrigerated transport maintained temperature between 2 and 8 degrees Celsius throughout.",
    "Route optimization algorithm reduced total fleet mileage by 18 percent this quarter.",
    "Bill of lading number BL-2024-56789 issued for 3 containers of electronic components.",
    "Hazardous materials shipping declaration completed for UN 1203 Class 3 flammable liquid.",
    "Supply chain visibility platform tracks 12000 active shipments across 45 countries.",
    "Dock scheduling system reduced truck waiting time from 90 minutes to 25 minutes.",
    "The third-party logistics provider handles 50000 square meters of warehousing space.",
    "Import duty calculated at 5.5 percent of declared customs value for tariff classification.",
    "Reverse logistics program processed 8000 customer returns with 95 percent accuracy.",
    "Intermodal transport combined rail and truck reducing carbon emissions by 40 percent.",
    "Demand forecasting model predicted 92 percent accuracy for the next quarter inventory needs.",
    "The automated sorting system processes 15000 parcels per hour with 99.8 percent accuracy.",
  ],

  multilingual: [
    "The translation from English to Arabic maintained semantic equivalence across all passages.",
    "Diacritical marks in French text were preserved during optical character recognition processing.",
    "The Unicode encoding standard supports over 150000 characters across all writing systems.",
    "Machine translation quality improved significantly for low-resource language pairs.",
    "Chinese character recognition requires handling of over 50000 distinct ideographic symbols.",
    "The bidirectional text rendering engine correctly displays mixed Arabic and English content.",
    "Japanese text processing handles three scripts simultaneously: hiragana, katakana, and kanji.",
    "The internationalization framework supports date and number formatting for 200 locales.",
    "Korean hangul syllable blocks consist of consonant and vowel jamo combinations.",
    "The transliteration system converts Cyrillic script to Latin characters following ISO 9.",
    "Hindi Devanagari script processing requires proper handling of conjunct consonants.",
    "The locale-aware sorting algorithm follows language-specific collation rules correctly.",
    "Spanish text with inverted punctuation marks was correctly identified and preserved.",
    "The font rendering engine supports complex text layout for Indic and Semitic scripts.",
    "German compound words present unique challenges for word boundary detection algorithms.",
  ],

  handwriting: [
    "The cursive handwriting sample showed connected letterforms typical of Palmer method.",
    "Ink density variation across the historical manuscript required adaptive binarization.",
    "Character segmentation of the handwritten form achieved 89 percent accuracy on names.",
    "The signature verification system compared 15 feature points for authentication.",
    "Historical document preservation required careful digitization at 600 dots per inch.",
    "Stroke width estimation helped distinguish between different writers on the same page.",
    "The handwriting recognition model processed 500 form fields per minute with 94 percent accuracy.",
    "Baseline detection in slanted handwriting improved word segmentation by 12 percent.",
    "The pen pressure variation provided additional features for writer identification.",
    "Degraded handwritten text from the 18th century archives required specialized enhancement.",
    "Connected component analysis separated overlapping characters in dense handwritten text.",
    "The ligature detection algorithm identified 23 common letter pair connections in cursive.",
    "Field extraction from structured forms achieved 97 percent accuracy on printed entries.",
    "Writer adaptation improved recognition accuracy by 8 percent after 50 training samples.",
    "The historical manuscript contained annotations in three different handwriting styles.",
  ],
};

// ─── OCR Confusion Pairs (for calibration) ──────────────────────────

const CONFUSION_SETS = {
  medical: [
    { ocr: 'tr0ponin', correct: 'troponin' },
    { ocr: 'm0rphine', correct: 'morphine' },
    { ocr: 'card1ac', correct: 'cardiac' },
    { ocr: 'piperacil1in', correct: 'piperacillin' },
    { ocr: 'taz0bactam', correct: 'tazobactam' },
    { ocr: 'ceftriax0ne', correct: 'ceftriaxone' },
    { ocr: 'metf0rmin', correct: 'metformin' },
    { ocr: 'vanc0mycin', correct: 'vancomycin' },
    { ocr: 'hem0globin', correct: 'hemoglobin' },
    { ocr: 'bi1ateral', correct: 'bilateral' },
    { ocr: 'pneum0nia', correct: 'pneumonia' },
    { ocr: 'lapar0scopic', correct: 'laparoscopic' },
    { ocr: 'tachy card ia', correct: 'tachycardia' },
    { ocr: 'hypertensi0n', correct: 'hypertension' },
    { ocr: 'chem0therapy', correct: 'chemotherapy' },
  ],
  legal: [
    { ocr: 'p1aintiff', correct: 'plaintiff' },
    { ocr: 'ju0gment', correct: 'judgment' },
    { ocr: 'arb1tration', correct: 'arbitration' },
    { ocr: 'jurisdicti0n', correct: 'jurisdiction' },
    { ocr: 'dep0sition', correct: 'deposition' },
    { ocr: 'def3ndant', correct: 'defendant' },
    { ocr: 'sett1ement', correct: 'settlement' },
    { ocr: 'c0mpliance', correct: 'compliance' },
    { ocr: 'neg1igence', correct: 'negligence' },
    { ocr: 'ame0dment', correct: 'amendment' },
  ],
  financial: [
    { ocr: '$4.2 bi11ion', correct: '$4.2 billion' },
    { ocr: 'depreciati0n', correct: 'depreciation' },
    { ocr: 'am0rtization', correct: 'amortization' },
    { ocr: 'div1dend', correct: 'dividend' },
    { ocr: 'exp3nditures', correct: 'expenditures' },
    { ocr: 'c0nsolidated', correct: 'consolidated' },
    { ocr: 'l1quidity', correct: 'liquidity' },
    { ocr: 'imp4irment', correct: 'impairment' },
    { ocr: 'shareh0lders', correct: 'shareholders' },
    { ocr: 'restruc+uring', correct: 'restructuring' },
  ],
  scientific: [
    { ocr: 'cata1yst', correct: 'catalyst' },
    { ocr: 'p0lymorphisms', correct: 'polymorphisms' },
    { ocr: 'spectr0scopic', correct: 'spectroscopic' },
    { ocr: 'nan0particle', correct: 'nanoparticle' },
    { ocr: 'micr0molar', correct: 'micromolar' },
    { ocr: 'flu0rescence', correct: 'fluorescence' },
    { ocr: 'e1ectron', correct: 'electron' },
    { ocr: 'p0lymerase', correct: 'polymerase' },
  ],
  engineering: [
    { ocr: 'micr0controller', correct: 'microcontroller' },
    { ocr: 'def1ection', correct: 'deflection' },
    { ocr: 'frequ3ncy', correct: 'frequency' },
    { ocr: 'res0nance', correct: 'resonance' },
    { ocr: 'a1gorithm', correct: 'algorithm' },
    { ocr: 'hydrau1ic', correct: 'hydraulic' },
    { ocr: 'kilon3wtons', correct: 'kilonewtons' },
    { ocr: 'optica1', correct: 'optical' },
  ],
  education: [
    { ocr: 'curr1culum', correct: 'curriculum' },
    { ocr: 'ass3ssment', correct: 'assessment' },
    { ocr: 'enr0llment', correct: 'enrollment' },
    { ocr: 'pr0ficiency', correct: 'proficiency' },
    { ocr: 'diss3rtation', correct: 'dissertation' },
  ],
  government: [
    { ocr: 'regu1ation', correct: 'regulation' },
    { ocr: 'imm1gration', correct: 'immigration' },
    { ocr: 'legis1ative', correct: 'legislative' },
    { ocr: 'evacuati0n', correct: 'evacuation' },
    { ocr: 'infra5tructure', correct: 'infrastructure' },
  ],
  retail: [
    { ocr: 'inv0ice', correct: 'invoice' },
    { ocr: 'invent0ry', correct: 'inventory' },
    { ocr: 'warehous3', correct: 'warehouse' },
    { ocr: 'transacti0ns', correct: 'transactions' },
    { ocr: 'fulf1llment', correct: 'fulfillment' },
  ],
  logistics: [
    { ocr: 'shi0ment', correct: 'shipment' },
    { ocr: 'c0ntainer', correct: 'container' },
    { ocr: 'de1ivery', correct: 'delivery' },
    { ocr: 'warehou5ing', correct: 'warehousing' },
    { ocr: 'intermo0al', correct: 'intermodal' },
  ],
  multilingual: [
    { ocr: 'trans1ation', correct: 'translation' },
    { ocr: 'enc0ding', correct: 'encoding' },
    { ocr: 'unic0de', correct: 'unicode' },
    { ocr: 'bi1ingual', correct: 'bilingual' },
  ],
  handwriting: [
    { ocr: 'cursiv3', correct: 'cursive' },
    { ocr: 'manuscri0t', correct: 'manuscript' },
    { ocr: 'segmentat1on', correct: 'segmentation' },
    { ocr: 'binarizati0n', correct: 'binarization' },
  ],
};

// ─── Also load existing Shifu corpus ────────────────────────────────

let EXTRA_MEDICAL = [];
try {
  EXTRA_MEDICAL = require('../learning/medical_corpus');
} catch (e) { /* not available */ }

let LANGUAGE_CORPUS = [];
try {
  const raw = fs.readFileSync(path.join(__dirname, '..', 'corpus_data', 'shifu_language_corpus.txt'), 'utf8');
  LANGUAGE_CORPUS = raw.split('\n').filter(l => l.trim().length > 20).slice(0, 2000);
} catch (e) { /* not available */ }

// Add extra medical corpus
if (EXTRA_MEDICAL.length > 0 && CORPORA.medical) {
  CORPORA.medical = CORPORA.medical.concat(EXTRA_MEDICAL.slice(0, 200));
}

// Distribute language corpus across domains
if (LANGUAGE_CORPUS.length > 0 && CORPORA.general) {
  CORPORA.general = CORPORA.general.concat(LANGUAGE_CORPUS.slice(0, 500));
}

// ─── Main Feeder ────────────────────────────────────────────────────

async function main() {
  console.log('');
  console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);
  console.log(`${C.bright}${C.cyan}  SHIFU LIVE FEEDER — Cross-Domain Teaching with Real-Time Progress${C.reset}`);
  console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);
  console.log('');

  // Create engine + teacher
  console.log(`${C.dim}  Initializing Shifu engine...${C.reset}`);
  const engine = new ShifuEngine();
  const teacher = createTeacher({ engine }, { restore: true, seedBuiltin: false });
  console.log(`${C.green}  Engine ready.${C.reset}`);

  // Determine which domains to feed
  let domainIds = Object.keys(CORPORA);
  if (DOMAIN_FILTER) {
    domainIds = domainIds.filter(d => d === DOMAIN_FILTER);
    if (domainIds.length === 0) {
      console.log(`${C.red}  Unknown domain: ${DOMAIN_FILTER}${C.reset}`);
      process.exit(1);
    }
  }

  // Calculate totals
  let totalSentences = 0;
  let totalPairs = 0;
  for (const d of domainIds) {
    totalSentences += (CORPORA[d] || []).length;
    totalPairs += (CONFUSION_SETS[d] || []).length;
  }
  console.log(`${C.dim}  Domains: ${domainIds.length} | Sentences: ${totalSentences} | Confusion pairs: ${totalPairs}${C.reset}`);
  console.log('');

  // Global progress tracking
  const globalStats = {
    fed: 0,
    corrections: 0,
    vocabSize: 0,
    domainsComplete: 0,
    startTime: Date.now(),
    domainStats: {},
    universalConfusions: 0,
  };

  const HEADER_LINES = 8; // Lines used by the dashboard

  function renderDashboard() {
    const elapsed = ((Date.now() - globalStats.startTime) / 1000).toFixed(1);
    const speed = globalStats.fed > 0 ? (globalStats.fed / (elapsed || 1)).toFixed(1) : '0';
    const globalPct = totalSentences > 0 ? globalStats.fed / totalSentences : 0;

    const lines = [];
    lines.push(`${C.bright}  ┌─── SHIFU LEARNING DASHBOARD ─────────────────────────────────┐${C.reset}`);
    lines.push(`${C.bright}  │${C.reset} Overall: ${bar(globalPct, 35)} ${pctStr(globalPct)} ${C.dim}(${globalStats.fed}/${totalSentences})${C.reset}    ${C.bright}│${C.reset}`);
    lines.push(`${C.bright}  │${C.reset} Speed: ${C.cyan}${speed} sent/s${C.reset}  Elapsed: ${C.cyan}${elapsed}s${C.reset}  Vocab: ${C.cyan}${numStr(globalStats.vocabSize)}${C.reset}  Fixes: ${C.yellow}${globalStats.corrections}${C.reset} ${C.bright}│${C.reset}`);
    lines.push(`${C.bright}  ├─── DOMAIN PROGRESS ──────────────────────────────────────────┤${C.reset}`);

    for (const d of domainIds) {
      const ds = globalStats.domainStats[d] || { fed: 0, total: 0, pairs: 0, vocab: 0, phase: 'foundation' };
      const pct = ds.total > 0 ? ds.fed / ds.total : 0;
      const phaseBadge = ds.phase === 'mastery' ? `${C.bgGreen}${C.bright} MST ${C.reset}` :
                         ds.phase === 'transfer' ? `${C.bgBlue}${C.bright} TRN ${C.reset}` :
                         ds.phase === 'specialization' ? `${C.bgMagenta}${C.bright} SPC ${C.reset}` :
                         `${C.bgYellow}${C.bright} FND ${C.reset}`;
      const name = (d + '          ').slice(0, 13);
      lines.push(`${C.bright}  │${C.reset} ${name} ${bar(pct, 20)} ${pctStr(pct)} ${phaseBadge} ${C.dim}v:${numStr(ds.vocab)}${C.reset} ${C.bright}│${C.reset}`);
    }

    lines.push(`${C.bright}  └──────────────────────────────────────────────────────────────┘${C.reset}`);

    return lines;
  }

  // First render
  let dashLines = renderDashboard();
  for (const l of dashLines) console.log(l);
  let lastDashLineCount = dashLines.length;

  function updateDashboard() {
    clearLines(lastDashLineCount);
    dashLines = renderDashboard();
    for (const l of dashLines) console.log(l);
    lastDashLineCount = dashLines.length;
  }

  // ─── Feed each domain ─────────────────────────────────────────────

  for (const domainId of domainIds) {
    const sentences = CORPORA[domainId] || [];
    const confusions = CONFUSION_SETS[domainId] || [];
    const total = sentences.length + confusions.length;

    teacher.activateDomain(domainId);
    teacher.startSession(domainId);

    globalStats.domainStats[domainId] = {
      fed: 0,
      total,
      pairs: 0,
      vocab: 0,
      phase: 'foundation',
    };

    // Feed sentences
    for (let i = 0; i < sentences.length; i++) {
      teacher.teachSentence(sentences[i], domainId);

      globalStats.fed++;
      globalStats.domainStats[domainId].fed++;

      // Update vocab count
      const progress = teacher.getProgress();
      const dp = progress.domainProgress[domainId];
      if (dp) {
        globalStats.domainStats[domainId].vocab = dp.vocabularySize;
        globalStats.vocabSize = Object.values(progress.domainProgress)
          .reduce((s, d) => s + d.vocabularySize, 0);
      }

      // Phase progression
      const curr = teacher.getCurriculum(domainId);
      curr.recordResult(true, 0.7);
      const totalInPhase = curr.phaseProgress[curr.phase].total;
      if (totalInPhase >= 20 && totalInPhase % 20 === 0) {
        // Simulate high accuracy to enable promotion
        for (let k = 0; k < 5; k++) curr.recordResult(true, 0.9);
        if (curr.canPromote()) {
          curr.promote();
          globalStats.domainStats[domainId].phase = curr.phase;
        }
      }
      globalStats.domainStats[domainId].phase = curr.phase;

      // Update display periodically
      const updateFreq = FAST_MODE ? 25 : 5;
      if (i % updateFreq === 0 || i === sentences.length - 1) {
        updateDashboard();
      }

      await sleep(DELAY);
    }

    // Feed confusion pairs
    for (let i = 0; i < confusions.length; i++) {
      const { ocr, correct } = confusions[i];
      teacher.teachCorrection(ocr, correct, domainId, { weight: 2 });

      globalStats.corrections++;
      globalStats.fed++;
      globalStats.domainStats[domainId].fed++;
      globalStats.domainStats[domainId].pairs++;
      globalStats.universalConfusions = teacher.model.universalConfusions
        ? Object.keys(teacher.model.universalConfusions).length : 0;

      if (i === confusions.length - 1) {
        updateDashboard();
      }

      await sleep(DELAY);
    }

    teacher.endSession();
    globalStats.domainsComplete++;
  }

  // ─── Final Report ─────────────────────────────────────────────────

  updateDashboard();

  console.log('');
  console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);
  console.log(`${C.bright}${C.cyan}  FEEDING COMPLETE — Final Report${C.reset}`);
  console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);
  console.log('');

  const elapsed = ((Date.now() - globalStats.startTime) / 1000).toFixed(1);
  const progress = teacher.getProgress();

  console.log(`${C.bright}  Summary:${C.reset}`);
  console.log(`    Sentences taught: ${C.green}${globalStats.fed}${C.reset}`);
  console.log(`    Corrections fed:  ${C.yellow}${globalStats.corrections}${C.reset}`);
  console.log(`    Total vocabulary: ${C.cyan}${globalStats.vocabSize}${C.reset} words`);
  console.log(`    Universal confusions: ${C.magenta}${globalStats.universalConfusions}${C.reset}`);
  console.log(`    Domains completed: ${C.green}${globalStats.domainsComplete}/${domainIds.length}${C.reset}`);
  console.log(`    Total time: ${C.cyan}${elapsed}s${C.reset}`);
  console.log(`    Calibrations saved: ${C.green}${progress.totalCalibrations}${C.reset}`);
  console.log('');

  // Per-domain breakdown
  console.log(`${C.bright}  Domain Breakdown:${C.reset}`);
  for (const domainId of domainIds) {
    const dp = progress.domainProgress[domainId];
    const ds = globalStats.domainStats[domainId];
    if (dp) {
      console.log(`    ${C.bright}${domainId}${C.reset}: vocab=${C.cyan}${dp.vocabularySize}${C.reset}, taught=${dp.sentencesTaught}, corrections=${C.yellow}${dp.correctionsMade}${C.reset}, phase=${C.green}${ds.phase}${C.reset}`);
    }
  }
  console.log('');

  // Curriculum status
  console.log(`${C.bright}  Curriculum Phases:${C.reset}`);
  const currProgress = progress.curriculum;
  for (const [domainId, cp] of Object.entries(currProgress)) {
    const phaseEmoji = cp.phase === 'mastery' ? '[MST]' :
                       cp.phase === 'transfer' ? '[TRN]' :
                       cp.phase === 'specialization' ? '[SPC]' : '[FND]';
    console.log(`    ${domainId}: ${phaseEmoji} ${cp.phaseLabel} ${cp.canPromote ? '(READY TO PROMOTE)' : ''}`);
  }
  console.log('');

  // Recommendations
  const recs = teacher.getRecommendations();
  if (recs.length > 0) {
    console.log(`${C.bright}  Recommendations:${C.reset}`);
    for (const rec of recs.slice(0, 5)) {
      console.log(`    - ${rec.message}`);
    }
    console.log('');
  }

  // Save state
  console.log(`${C.dim}  Saving teaching state...${C.reset}`);
  teacher.save();
  console.log(`${C.green}  State saved. Calibrations persisted for future learning.${C.reset}`);
  console.log('');
  console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);
}

main().catch(err => {
  console.error(`${C.red}Error: ${err.message}${C.reset}`);
  console.error(err.stack);
  process.exit(1);
});
