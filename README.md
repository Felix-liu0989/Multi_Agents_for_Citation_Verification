The core implementation of the Multi-Agent Collaborative Generation (MACG) framework for generating high-quality related work sections with citation verification and correction capabilities.


 Environment Setup for MACG Framework
![Uploading image.png…]()

## 🐍 Python Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Create conda environment
conda create -n taxo python=3.10 -y
conda activate taxo

# Install dependencies
pip install -r requirements.txt
```

## 📦 Core Dependencies

## 🔧 Environment Variables

Create a `.env` file in the project root:

```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
DEEPSEEK_API_KEY=your_deepseek_key

# Vector Database
MILVUS_URI=your_milvus_uri
MILVUS_TOKEN=your_milvus_token

# Model Paths
SENTENCE_TRANSFORMER_MODEL=/path/to/sentence/transformer
BERTOPIC_MODEL=/path/to/bertopic/model
```

## 🚀 Quick Start

1. **Activate Environment**
   ```bash
   conda activate taxo
   ```

2. **Set Environment Variables**
   ```bash
   export OPENAI_API_KEY="your_key"
   export MILVUS_URI="your_uri"
   ```

3. **Basic Usage**
```python
from citegeist.generator import Generator
import os
import json,jsonlines
import time
from citegeist.utils.infer import load_processed_ids

generator = Generator(
   llm_provider="gemini",  # Choice of: "azure" (OpenAI Studio), "anthropic", "gemini", "mistral", and "openai"
   api_key=os.environ.get("OPENROUTER_API_KEY"), # Here, you will need to set the respective API key
   model_name="google/gemini-2.5-flash", # Choose the model that the provider supports
   database_uri=os.environ.get("MILVUS_URI", ""),  # Set the path (local) / url (remote) for the Milvus DB connection
   database_token=os.environ.get("MILVUS_TOKEN", ""),  # Optionally, also set the access token (you DON'T need to set this when using the locally hosted Milvus Database)
)

abstract = ""
result = generator.generate_related_work_MACG(abstract, 10, 2, 0.0)
```


### Core Components

#### `citegeist/` - Main Framework
- **`generator.py`** - Core generator class with citation validation and correction
- **`utils/`** - Utility functions and helper modules
- **`database/`** - Database interaction modules
- **`__init__.py`** - Package initialization

#### `baselines/` - Baseline Implementations
- **`baseline_perplexity_deep_research.py`** - Perplexity Deep Research baseline
- **`baseline_naive_rag_gpt.py`** - Naive RAG with GPT baseline
- **`baseline_vallina_gpt.py`** - Vanilla GPT baseline

#### `multi_dims/` - Multi-dimensional Analysis
- Multi-dimensional literature analysis and visualization tools


## 🚀 Key Features

### 1. Citation Verification & Correction
- **Dual-model validation** using Gemini and DeepSeek
- **Error classification** into 5 categories:
  - Direct Contradiction
  - Information Not Present / Unsubstantiated
  - Misrepresentation / Imprecise Wording
  - Incorrect Attribution
  - Other

### 2. Multi-Agent Framework
- **Summarizer Agent** - Generates concise literature summaries
- **Structurer Agent** - Identifies and groups research themes
- **Integrator Agent** - Synthesizes outputs into coherent narrative
- **FactCheck Agent** - Verifies fidelity to source material

### 3. Evaluation Metrics
- **Claim Precision** - Accuracy of individual claims
- **Citation Precision** - Accuracy of citations
- **Reference Precision** - Accuracy of source attribution
- **Citation Density** - Citation frequency per sentence
- **Average Citations per Sentence** - Citation distribution






---
