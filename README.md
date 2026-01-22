# Statistical Tests - Hospital Data Analysis

A professional Python project for performing statistical analysis on hospital datasets, comparing door-to-needle times between two hospitals using normality tests, t-tests, and Wilcoxon tests.

## Table of Contents

- [Project Overview](#project-overview)
- [Why Multiple Versions?](#why-multiple-versions)
- [How It Works](#how-it-works)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)

## Project Overview

This project implements a complete statistical analysis pipeline to compare door-to-needle times between two hospitals. The pipeline automatically:

1. Expands frequency-based datasets into single-column arrays
2. Tests data normality using Shapiro-Wilk tests
3. Applies appropriate comparison tests:
   - **t-test** if both datasets are normally distributed (p > 0.05)
   - **Wilcoxon rank-sum test** if one or both datasets are non-normal (p ≤ 0.05)

## Why Multiple Versions?

This project provides **two different implementations** to suit different use cases:

### V1 - Modular Architecture (Recommended for Development)

**Location**: `/V1/`

**Best for**: Learning, testing, and projects requiring flexibility

**Features**:
- Clean separation of concerns with dedicated modules:
  - `data_generator.py` - Generate synthetic hospital datasets
  - `data_processor.py` - Expand frequency-based data
  - `statistical_tests.py` - Perform statistical tests
- Comprehensive unit test coverage with pytest
- Easy to extend and modify individual components
- Professional project structure following Python best practices

**Use when**:
- You need to understand or modify the internal workings
- You want comprehensive testing and debugging capabilities
- You're building a larger project and need modular components

### V2 - Monolithic Module (Recommended for Integration)

**Location**: `/V2/`

**Best for**: Integration into other projects (Rasa chatbots, APIs, web services)

**Features**:
- Single file (`hospital_statistics.py`) containing all functionality
- Minimal dependencies
- Simple API with convenience functions
- Easy to copy/paste into other projects
- Ideal for chatbots and microservices

**Use when**:
- You need to integrate statistical analysis into a chatbot (e.g., Rasa)
- You want to quickly add hospital statistics to an existing project
- You prefer a simple, self-contained module
- You need JSON/dictionary output for APIs

**Quick comparison**:

| Feature | V1 (Modular) | V2 (Monolithic) |
|---------|--------------|-----------------|
| Files | Multiple modules | Single file |
| Testing | Comprehensive pytest suite | Simple test script |
| Integration | Requires package structure | Copy single file |
| Best for | Development & learning | Production integration |
| Flexibility | High - modify individual parts | Medium - all in one |

## How It Works

### The Statistical Pipeline Explained

The program follows a systematic approach to ensure valid statistical comparisons:

#### Step 1: Data Generation or Input
- **Synthetic data**: Generate realistic hospital datasets with configurable parameters
- **Real data**: Load your own CSV files with `door_to_needle` times and frequency counts

#### Step 2: Data Expansion
Input format (frequency table):
```
door_to_needle    n
     45          3
     60          2
```

Expanded format (individual observations):
```
[45, 45, 45, 60, 60]
```

Each measurement is repeated according to its frequency count, creating a complete dataset for statistical analysis.

#### Step 3: Normality Testing (Shapiro-Wilk)
The program tests whether each hospital's data follows a normal distribution:
- **Null hypothesis (H₀)**: Data is normally distributed
- **Significance level**: α = 0.05
- **Decision rule**:
  - p-value > 0.05 → Data is normal
  - p-value ≤ 0.05 → Data is non-normal

#### Step 4: Choosing the Right Statistical Test
Based on normality test results, the program automatically selects:

**Option A: Independent t-test** (when both datasets are normal)
- Parametric test comparing means
- More powerful when assumptions are met
- Returns: t-statistic and p-value

**Option B: Wilcoxon Rank-Sum Test** (when one or both datasets are non-normal)
- Non-parametric alternative (Mann-Whitney U test)
- No normality assumption required
- Compares distributions by ranking values
- Returns: U-statistic and p-value

#### Step 5: Interpretation
- **p-value < 0.05**: Significant difference between hospitals
- **p-value ≥ 0.05**: No significant difference detected

The program outputs formatted results with clear interpretation guidance.

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project**:
```bash
cd /path/to/Statistical_Tests
```

2. **Create a virtual environment** (recommended):
```bash
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

3. **Install dependencies**:
```bash
# For V1 (modular version)
pip install -r V1/requirements.txt

# For V2 (monolithic version)
pip install -r V2/requirements.txt
```

### Quick Test

**Test V1**:
```bash
# Navigate to V1
cd V1

# Run main analysis
python main.py

# Run tests
pytest
```

**Test V2**:
```bash
# Navigate to V2
cd V2

# Run test module
python test_module.py

# Run examples
python examples.py
```

### Troubleshooting Setup

**Issue**: `ModuleNotFoundError`
- **Solution**: Make sure virtual environment is activated and dependencies are installed

**Issue**: `pytest: command not found`
- **Solution**: Install pytest: `pip install pytest`

**Issue**: Import errors in V1
- **Solution**: Run scripts from the V1 directory: `cd V1 && python main.py`

## Project Structure

### Overall Structure

```
Statistical_Tests/
├── .venv/                       # Virtual environment (created during setup)
├── V1/                          # Modular version
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_generator.py    # Dataset generation
│   │   ├── data_processor.py    # Dataset expansion
│   │   └── statistical_tests.py # Statistical testing
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_data_generator.py
│   │   ├── test_data_processor.py
│   │   └── test_statistical_tests.py
│   ├── results/                 # Analysis results (auto-generated)
│   ├── main.py                  # Main execution script
│   ├── examples.py              # Usage examples
│   └── requirements.txt         # Dependencies
├── V2/                          # Monolithic version
│   ├── hospital_statistics.py   # All-in-one module
│   ├── test_module.py           # Simple test script
│   ├── examples.py              # Usage examples
│   ├── results/                 # Analysis results (auto-generated)
│   ├── README.md                # V2-specific documentation
│   └── requirements.txt         # Dependencies
├── .README.md                   # This file (main documentation)
└── README.md                    # Project overview
```

### V1 Module Descriptions

- **`data_generator.py`**: Functions to generate synthetic hospital datasets with configurable parameters
- **`data_processor.py`**: Expands frequency-based data into individual observations
- **`statistical_tests.py`**: Performs Shapiro-Wilk tests, t-tests, and Wilcoxon tests
- **`main.py`**: Example script running the complete analysis pipeline
- **`tests/`**: Comprehensive unit tests using pytest

### V2 Module Description

- **`hospital_statistics.py`**: Self-contained module with all functionality
  - Data generation
  - Data expansion
  - Statistical tests
  - Result formatting
  - File I/O
- **`examples.py`**: Multiple usage examples including API integration
- **`test_module.py`**: Simple functional test

## Usage

### V1 - Modular Version

#### Quick Start

Run the complete analysis pipeline with default parameters:

```bash
cd V1
python main.py
```

#### Python API Usage

```python
from src.data_generator import generate_both_hospital_datasets
from src.data_processor import expand_both_datasets
from src.statistical_tests import run_statistical_pipeline

# Generate datasets
df_hospital_1, df_hospital_2 = generate_both_hospital_datasets(
    n_rows=50,
    random_state=42
)

# Expand datasets
expanded_1, expanded_2 = expand_both_datasets(df_hospital_1, df_hospital_2)

# Run statistical analysis
result = run_statistical_pipeline(expanded_1, expanded_2)

# Access results
print(f"Test type: {result.test_type}")
print(f"P-value: {result.test_result.p_value}")
```

#### Run Examples

```bash
cd V1
python examples.py
```

### V2 - Monolithic Version

#### Quick Start

```bash
cd V2
python examples.py
```

#### Simple Usage

```python
from hospital_statistics import quick_analysis

# Run quick analysis with synthetic data
result = quick_analysis(n_rows=50, random_state=42, save=True)
print(result.get_summary())
```

#### Using Your Own Data

```python
import pandas as pd
from hospital_statistics import analyze_dataframes

# Load your data
df1 = pd.read_csv('hospital1.csv')  # Columns: door_to_needle, n
df2 = pd.read_csv('hospital2.csv')

# Analyze
result = analyze_dataframes(df1, df2)
print(result.get_summary())

# Get results as dictionary for APIs
result_dict = result.to_dict()
```

#### Integration Example (Rasa Chatbot)

```python
from rasa_sdk import Action
from hospital_statistics import HospitalStatistics

class ActionAnalyzeHospitals(Action):
    def name(self) -> str:
        return "action_analyze_hospitals"
    
    def run(self, dispatcher, tracker, domain):
        stats = HospitalStatistics()
        result = stats.run_analysis(n_rows=50, random_state=42)
        
        message = (
            f"Difference {'SIGNIFICANT' if result.is_significant() else 'NOT significant'} "
            f"(p={result.test_result.p_value:.4f}, test: {result.test_type})"
        )
        
        dispatcher.utter_message(text=message)
        return []
```

## Data Format

### Input Format

Datasets contain two columns:
- **door_to_needle**: Integer values between 30 and 120 (time in minutes)
- **n**: Integer frequency counts between 1 and 50

Example:
```
door_to_needle    n
     45          3
     60          2
     75          5
```

### Expanded Format

After expansion, each `door_to_needle` value is repeated `n` times:
```
[45, 45, 45, 60, 60, 75, 75, 75, 75, 75]
```

## Statistical Pipeline

### Step 1: Shapiro-Wilk Normality Test

Tests whether each dataset follows a normal distribution:
- **Null hypothesis**: Data is normally distributed
- **Significance level**: α = 0.05
- **Decision**: p > 0.05 → normal; p ≤ 0.05 → non-normal

### Step 2a: Independent t-test (if both datasets are normal)

Compares means of two normally distributed samples:
- Returns: t-statistic and p-value
- **Null hypothesis**: No difference in means between hospitals

### Step 2b: Wilcoxon Rank-Sum Test (if either dataset is non-normal)

Non-parametric alternative to t-test:
- Returns: U-statistic and p-value
- **Null hypothesis**: No difference in distributions between hospitals

## Testing

### V1 - Comprehensive Testing

Run the complete test suite:

```bash
cd V1
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run specific test file:

```bash
pytest tests/test_statistical_tests.py
```

Run with coverage report:

```bash
pytest --cov=src tests/
```

### V2 - Simple Testing

```bash
cd V2
python test_module.py
```

## Data Format

### Input Format

Datasets contain two columns:
- **door_to_needle**: Integer values between 30 and 120 (time in minutes)
- **n**: Integer frequency counts between 1 and 50

Example CSV:
```csv
door_to_needle,n
45,3
60,2
75,5
```

### Expanded Format

After expansion, each `door_to_needle` value is repeated `n` times:
```
[45, 45, 45, 60, 60, 75, 75, 75, 75, 75]
```

## Example Output

```
======================================================================
STATISTICAL ANALYSIS RESULTS: HOSPITAL COMPARISON
======================================================================

1. SHAPIRO-WILK NORMALITY TESTS
----------------------------------------------------------------------
Hospital 1:
  Statistic (W): 0.9954
  P-value:       0.2346
  Normal:        True

Hospital 2:
  Statistic (W): 0.9961
  P-value:       0.3457
  Normal:        True

2. COMPARISON TEST: T-TEST
----------------------------------------------------------------------
  T-statistic:   -2.3457
  P-value:       0.0199

3. INTERPRETATION
----------------------------------------------------------------------
  Result: SIGNIFICANT difference (p < 0.05)
  The two hospitals show statistically different door-to-needle times.
======================================================================
```

## Key Features

- **Professional Code Quality**: Fully documented with comprehensive docstrings
- **Type Hints**: All functions include type annotations
- **Two Architectures**: Choose modular or monolithic based on your needs
- **Comprehensive Testing**: Full unit test coverage (V1) and functional tests (V2)
- **Reproducible**: Random state control for consistent results
- **Configurable**: Easy parameter customization
- **Easy Integration**: V2 designed for chatbots, APIs, and microservices

## Dependencies

- **numpy** (≥1.24.0): Numerical computing
- **scipy** (≥1.10.0): Statistical functions (Shapiro-Wilk, t-test, Mann-Whitney U)
- **pandas** (≥2.0.0): Data manipulation and DataFrame operations
- **pytest** (≥7.4.0): Testing framework (V1 only)

## Technical Notes

### Wilcoxon Test Implementation

This project uses `scipy.stats.mannwhitneyu`, which implements the Wilcoxon rank-sum test for independent samples (also known as the Mann-Whitney U test). The statistic returned is the U-statistic.

### Type Hints

The project uses Python 3.10+ union syntax (`|`) for type hints. For Python < 3.10, use `Union` from `typing`.

### When to Use Each Version

**Use V1 if**:
- You're learning statistical analysis
- You need to modify the algorithm
- You want comprehensive testing
- You're building a larger application with complex requirements

**Use V2 if**:
- You need to integrate into a chatbot (Rasa, Dialogflow, etc.)
- You're building a REST API or microservice
- You want a quick drop-in solution
- You need JSON output for web applications
- You prefer minimal file dependencies

## Contributing

To extend this project:

1. Choose the appropriate version (V1 for modular, V2 for integration)
2. Add new statistical tests or features
3. Create corresponding unit tests (V1) or update test_module.py (V2)
4. Update documentation with usage examples
5. Run tests to ensure everything works

## Author

Mathieu

## Date

January 2026


