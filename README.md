# GroupX_dataAnalysis
Data Analysis And Visualization group project


## Overview
This repository contains code for analyzing two South African datasets from the World Bank to explore [Poverty headcount ratio at national poverty lines (% of population),
Employment to population ratio, 15+, total (%) (national estimate)]. The aim is to explore the relationship between employment levels and poverty reduction using World Bank indicators.

## Relationship Between the Indicators
* Employment and poverty are closely connected.
* Higher employment-to-population ratios usually mean more people are earning an income, which reduces the poverty headcount ratio.
* Conversely, low employment rates often increase poverty levels, especially in economies with limited social protection.
* Analyzing these indicators together helps us understand how labor markets influence poverty outcomes.

## Setup Instructions
1. Clone the repo: `git clone https://github.com/your-username/GroupX_DataAnalysis.git`
2. Install dependencies: `pip install -r requirements.txt` (requires Python 3.8+ installed from python.org).
3. Download datasets and place in `data/` (see below for links).
4. Run the main script: `python scripts/data_pipeline.py`
5. Open notebook: `jupyter notebook notebooks/analysis.ipynb` (install Jupyter if needed: `pip install jupyter`).

## Datasets
- Dataset 1: [Poverty headcount ratio at national poverty lines (% of population)] from https://data360.worldbank.org/en/indicator/WB_WDI_SI_POV_NAHC
  * Measures the share of the working-age population (15 years and older) that is employed. Employment includes wage jobs and self-employment, even part-time work.
- Dataset 2: [Employment to population ratio, 15+, total (%) (national estimate)] from https://data360.worldbank.org/en/indicator/WB_WDI_SL_EMP_TOTL_SP_NE_ZS
  *Percentage of the population living below the national poverty line, as defined by each country. Based on household consumption or income surveys.

## Notes
- Ensure unique datasets to avoid 15% penalty.
