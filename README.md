# 🛰️ Verity Intelligence Command
**AI-Powered OSINT & Disinformation Intelligence Platform**

Verity Intelligence Command is a multimodal Open Source Intelligence (OSINT) platform designed to detect, analyze, and map coordinated disinformation campaigns. By combining NLP, image forensics, relational intelligence, and active learning pipelines, Verity serves as a powerful decision-support system for analysts, researchers, cybersecurity teams, and misinformation investigators.

> ⚠️ **Disclaimer:** Verity provides probabilistic intelligence assessments and forensic indicators. It is designed to assist and accelerate investigations, not to replace human verification or serve as the sole authority for factual accuracy.

---

## ✨ Core Capabilities

### 🔎 Multimodal Intelligence Engine
Verity combines NLP-based text forensics with AI image and deepfake analysis to generate unified credibility scoring.

**Text Intelligence**
*   **Embeddings:** DistilBERT-powered analysis.
*   **Deception Analysis:** Stylistic and sentence-level suspiciousness scoring.
*   **Verification:** Claim verification using NLI cross-encoders and entity validation via Wikipedia/Wikidata.

**Vision Intelligence**
*   **Detection Models:** SigLIP-2 AI image detection and deepfake classification.
*   **Forensics:** Error Level Analysis (ELA), EXIF metadata inspection, biological consistency checks, and face extraction pipelines.

### 🧠 Intelligence Graph (Neo4j)
Verity automatically builds a relational intelligence graph that maps the ecosystem of a narrative. It links articles, domains, authors, entities, and narrative clusters to enable cross-domain disinformation tracking, coordinated narrative detection, repeated entity propagation analysis, and domain risk profiling.

### 🤖 Proactive Threat Hunting
The platform includes an autonomous threat-hunting agent that continuously monitors the web. It searches for emerging misinformation narratives, scrapes suspicious articles, runs automated forensic analysis, stages uncertain content for analyst review, and optionally pushes intelligence directly into Neo4j. 

### 🔄 Human-in-the-Loop Active Learning
Verity features a retraining pipeline for continuous model improvement. Analysts can review uncertain scans, label real or fake content, trigger retraining, and fine-tune the vision model. The pipeline includes checkpoint recovery, structured training logs, ONNX INT8 quantization, transfer learning, early stopping, and validation tracking.

### 📊 Explainable AI
Instead of relying on black-box scoring, the platform exposes its forensic reasoning. Explainability features include sentence-level suspicion scoring, feature attribution, claim evidence inspection, entity verification summaries, and comprehensive confidence breakdowns.

---

## 🧱 System Architecture

```text
                    ┌─────────────────────────┐
                    │  Threat Hunter Agent    │
                    │  Web Scraping Pipeline  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌────────────────────┐
                     │  Intelligence Core │
                     └────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼

┌────────────────┐    ┌────────────────────┐    ┌─────────────────┐
│ Text Forensics │    │ Vision Forensics   │    │ Entity & Claims │
│                │    │                    │    │ Verification    │
│ • DistilBERT   │    │ • SigLIP-2         │    │ • Wikipedia API │
│ • Style Engine │    │ • Deepfake Model   │    │ • Wikidata      │
│ • NLP Signals  │    │ • ELA Detection    │    │ • NLI CrossEnc  │
└────────────────┘    └────────────────────┘    └─────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Credibility Scoring     │
                    │ + Explainability Layer  │
                    └─────────────────────────┘
                                 │
                 ┌───────────────┼────────────────┐
                 ▼                                ▼

       ┌──────────────────┐            ┌───────────────────┐
       │ SQLite History   │            │ Neo4j Graph Intel │
       │ Persistent Store │            │ Narrative Mapping │
       └──────────────────┘            └───────────────────┘
       
## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Backend** | Python |
| **NLP** | DistilBERT, spaCy, sentence-transformers |
| **Vision Models** | SigLIP-2, EfficientNet, facenet-pytorch |
| **Machine Learning**| scikit-learn, PyTorch |
| **Graph Intel** | Neo4j |
| **Database** | SQLite |
| **Optimization** | ONNX Runtime INT8 |
| **Visualization** | Plotly, streamlit-agraph |
| **PDF Reporting** | fpdf2 |

---

## 📂 Project Structure

```text
verity/
├── app.py                     # Main Streamlit intelligence dashboard
├── train_model.py             # Text intelligence training pipeline
├── auto_train.py              # Vision retraining pipeline
├── threat_hunter.py           # Autonomous OSINT hunter
├── image_forensics.py         # AI image + deepfake analysis
├── claim_verifier.py          # NLI-based claim verification
├── entity_checker.py          # Wikipedia/Wikidata validation
├── Sentence_scorer.py         # Sentence suspiciousness scoring
├── features.py                # Shared stylistic feature engine
├── neo4j_graph.py             # Graph intelligence layer
├── database.py                # Persistent SQLite storage
├── report_generator.py        # PDF intelligence reports
├── model/                     # Trained models + ONNX exports
├── training_data/             # Human-labeled retraining data
├── staging_data/              # Threat hunter staging queue
├── data/                      # Training datasets
└── requirements.txt
🚀 Installation
1. Clone the Repository

Bash
git clone [https://github.com/yourusername/verity-intelligence-command.git](https://github.com/yourusername/verity-intelligence-command.git)
cd verity-intelligence-command
2. Create a Virtual Environment

Windows:

DOS
python -m venv venv
venv\Scripts\activate
Linux / macOS:

Bash
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies

Bash
pip install -r requirements.txt
4. Install spaCy Model

Bash
python -m spacy download en_core_web_sm
▶️ Running The Platform
Launch the Dashboard
The default local URL will be http://localhost:8501.

Bash
streamlit run app.py
Threat Hunter
Bash
# Run a one-time scan
python threat_hunter.py --once

# Run continuous monitoring (e.g., every 30 minutes)
python threat_hunter.py --interval 30
Model Training
Bash
# Train the text intelligence model
python train_model.py

# Fine-tune the vision intelligence model
python auto_train.py
📑 Intelligence Reports
Verity generates structured, exportable PDF forensic reports containing:

Credibility scores

Forensic reasoning

Deepfake indicators

Claim verification evidence

Entity validation results

Metadata analysis

🔐 Key Engineering Features
Thread-safe SQLite architecture

ONNX INT8 quantization pipeline

Checkpoint recovery training

Modular ML pipelines

Structured JSON logging

Cached inference models

Transfer learning workflows

Graph-based intelligence correlation

📈 Future Roadmap
RAG-powered intelligence assistant

Real-time social media ingestion

Multi-language misinformation detection

Temporal narrative evolution tracking

Distributed threat intelligence feeds

Kubernetes deployment

Analyst collaboration system

Vector database integration

📜 License & Usage
License: MIT License

⚠️ Usage Scope: This project is strictly intended for educational research, cybersecurity experimentation, OSINT investigations, and media forensics research.