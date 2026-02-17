### Hybrid OSINT-Driven Cyber Threat Intelligence (CTI) Framework
  Overview

This repository presents a Hybrid Cyber Threat Intelligence (CTI) Framework that integrates unstructured Open-Source Intelligence (OSINT) with structured threat knowledge from the MITRE ATT&CK framework. The system combines Natural Language Processing (NLP), transformer-based contextual embeddings, deep learning classifiers, knowledge graph modeling, and ensemble validation to produce automated, contextual, and high-confidence cyber threat intelligence.

## The framework is designed for:
>> Security researchers
>> Threat intelligence analysts
>> SOC teams
>> Academic research
>> AI-driven cybersecurity experimentation



### Key Capabilities

>> Automated OSINT ingestion and preprocessing
>> Indicator of Compromise (IOC) extraction (IPs, Domains, CVEs, Hashes)
>> Cybersecurity-specific Named Entity Recognition (NER)
>> Hybrid feature engineering (TF-IDF + BERT embeddings)
>> CNN-based structured feature learning
>> Transformer-based semantic threat classification
>> Knowledge graph construction using NetworkX
>> Ensemble learning with confidence scoring
>> Mapping threats to MITRE ATT&CK tactics and techniques
>> Evaluation using Accuracy, Precision, Recall, F1-score



### System Architecture

OSINT Sources
     ↓
Text Preprocessing & IOC Extraction
     ↓
Feature Engineering
 (TF-IDF + BERT Embeddings)
     ↓
CNN Classifier + BERT Classifier
     ↓
Knowledge Graph Construction
     ↓
Ensemble Validation Layer
     ↓
Contextualized CTI Output



### Project Structure


Hybrid-CTI-Framework/
│
├── data/                     # Sample dataset (user-provided data required)
├── models/                   # Saved trained models
├── results/                  # Predictions and evaluation metrics
├── src/                      # Modular source code
├── Fully-updated-framework-code.py
├── requirements.txt
└── README.md



### Installation

1. Clone Repository
    $ git clone https://github.com/ifyouknowyou/A-hybrid-CTI-Framework.git
    $ cd A-hybrid-CTI-Framework

2. Create Virtual Environment
    $ python3 -m venv venv
    $ source venv/bin/activate        # Linux / macOS
    $ venv\Scripts\activate           # Windows

3. Install Dependencies
    $ pip install -r requirements.txt

4. Download spaCy Model
    $ python -m spacy download en_core_web_sm




### Dataset Information

>> The framework supports both unstructured and structured threat data.
>> Unstructured OSINT Sources
>> Cybersecurity news articles
>> Technical blogs
>> Threat intelligence reports
>> Vulnerability disclosures
>> Security forum discussions


### Expected dataset format (CSV):

>> text	label
>> Threat report content	malware
>> Phishing attack analysis	phishing

### Place dataset inside:

>> /kaggle/input/big-twitter-dataset/raw_tweets_text.csv
>> /kaggle/input/text-based-cyber-threat-detection/cyber-threat-intelligence_all.csv
>> MITRE ATT&CK techniques
>> CVE mappings
>> /kaggle/input/cyber-bert/CyberBERT.csv

Public vulnerability references
Note: Only sample datasets are included. Users must provide their own OSINT data.


### Running the Framework

Option 1: Direct Script Execution
    $ python Fully-updated-framework-code.py

Option 2: Modular Execution
    $ python src/main.py



### Output

The system generates:
>> Predicted threat category
>> Extracted IOCs
>> Associated ATT&CK techniques
>> Knowledge graph relationships
>> Model confidence score
>> Performance metrics


### Results are saved in:

  $ /results/


### Model Components

>> Feature Engineering
>> TF-IDF statistical representation
>> Transformer-based contextual embeddings (BERT)
>> Classification Layer
>> Convolutional Neural Network (CNN)
>> Transformer-based classifier
>> Knowledge Graph
>> Entity-relation modeling using NetworkX
>> Threat actor → Malware → Technique → CVE relationships
>> Ensemble Validation
>> Majority voting
>> Confidence-based aggregation
>> False positive reduction
>> Performance


### The hybrid architecture improves:

>> Contextual threat interpretation
>> Robustness against noisy OSINT
>> False positive reduction
>> Structured-to-unstructured data alignment


### Hardware Requirements
Minimum:
>> Python 3.10+
>> 8GB RAM

Recommended:
>> GPU (CUDA-enabled) for BERT training
>> 16GB RAM for large-scale OSINT datasets
>> Reproducibility

To reproduce results:
>> Use provided requirements.txt
>> Maintain consistent dataset splits
>> Fix random seeds during training
>> Use identical preprocessing steps


### Security Notice

This framework is intended for:

>> Research
>> Educational use
>> Defensive cybersecurity applications

It must not be used for offensive or malicious purposes.

Thank You !!!!!
