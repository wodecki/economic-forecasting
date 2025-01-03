# Economic Forecasting with LLMs

Supplementary Code and Data for "Combining generative AI and expert knowledge in economic forecasting
---

## Overview

This repository provides the supplementary code, data, and instructions associated with the paper titled "Combining generative AI and expert knowledge in economic forecasting". This study investigates the integration of Large Language Models (LLMs) with expert knowledge to extract and evaluate narrative economic forecasts from sell-side analyst reports.

Here, we present a simplified version of the codebase used in our paper. The key differences are as follows:

​	1.	This repository includes only one batch of ten economic reports, whereas the paper analyzed 10,000 such reports.

​	2.	All reports (reports.csv), code (including prompts in main.py), and examples (examples.csv) have been translated into English.

​	3.	Batch generation and handling functionalities, essential for processing large text corpora in the production code, have been removed for simplicity.



The repository includes:

- Python scripts for replicating the analyses.
- Datasets used in the study.
- Generated reports and examples.

## Repository Structure

```
github/
│
├── main.py             # Primary script for executing analyses
├── env.py              # Environment configuration
├── requirements.txt    # List of dependencies
├── .env                # A file with OpenAI API KEY - to be filled in by the user
├── data/               # Folder containing datasets
│   ├── forecasts.csv   # Forecast data derived from economic reports with LLMs
│   ├── examples.csv    # Example inputs/outputs
│   └── reports.csv     # Human-generated economic reports
```

## Requirements

### Using `venv`

To isolate dependencies and ensure compatibility, it's recommended to use a virtual environment:

1. **Create a virtual environment named `venv`:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Deactivate the virtual environment (when finished):**
   ```bash
   deactivate
   ```

### Alternative: Without `venv`

If you prefer to install the dependencies globally (not recommended), use:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/wodecki/economic-forecasting.git
   cd economic-forecasting
   ```

2. Set up the OPENAI_API_KEY within a `.env` file (you can obtain it from https://platform.openai.com/):
   ```
   OPENAI_API_KEY="sk-proj-vW...r0A"
   ```

3. Run the primary script:
   ```bash
   python main.py
   ```

## Data Description

### reports.csv

Includes a sample of daily economic reports from Polish financial institutions, which serve as a source for forecast extraction. This is an input to `main.py`

### examples.csv

Provides a carefully crafted set of examples to LLM as pairs of a sentence from economic report and a correct forecast. 

### forecasts.csv

Contain forecasts extracted from economic reports using LLM. This is the output of `main.py`



## Citation

If you use this repository in your research, please cite:
```
@article{Rybinski2025,
  title={Combining generative AI and expert knowledge in economic forecasting},
  author={Krzysztof Rybinski, Andrzej Wodecki},
  journal={Journal Name},
  year={2025},
  volume={xx},
  pages={xx--xx},
  doi={xx.xxxx/xxxxxx}
}
```

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the LICENSE file for details.

---
