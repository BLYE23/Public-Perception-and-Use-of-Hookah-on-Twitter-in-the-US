# Public Perception and Use of Hookah on Twitter in the US

## Overview

This project investigates public perceptions and behaviors surrounding hookah use in the United States, as reflected on Twitter. By leveraging advanced natural language processing (NLP) techniques, we provide actionable insights into trends, attitudes, and promotion patterns that can guide future tobacco prevention campaigns and regulatory policies.

## Team

- **Yiwei Han**
- **Puhua Ye**
- **Mengwei Wu**
- **Yuka Shimazaki**

### Sponsorship and Guidance

- **Sponsor**: URMC CTSI  
- **Academic Advisor**: Professor Ajay Anand  

---

## Project Objectives

1. Analyze public perception and usage trends of hookah on Twitter.
2. Identify key topics and attitudes toward hookah usage and promotions.
3. Provide guidance for potential regulatory measures and tobacco prevention strategies.

---

## Methods and Tools

### Data Collection and Preprocessing

- **Source**: Twitter API  
- **Scope**: Over 927,000 US-based tweets, filtered to 323,347 hookah-related tweets.  
- **Techniques**:
  - Keyword filtering.
  - Human labeling to categorize tweets by sentiment (positive, neutral, negative) and user type (user/non-user).
  - Data augmentation (e.g., synonym replacement, back translation).

### Modeling Approaches

- **RoBERTa**: Fine-tuned transformer model for sentiment classification and user
- **Llama 2**: Advanced language model fine-tuned using PEFT (LoRA) and NEFTune for enhanced performance.
- **Topic Modeling**:
  - **BERTopic** and **LDA** to identify themes in commercial and non-commercial tweets.

### Visualization and Insights

- Time-series analysis to track hookah-related tweet trends.
- Geographical analysis to map public attitudes and hookah usage across the US.

---

## Key Results

- **Commercial Trends**: The top hookah brands mentioned in tweets were Adalya and Starbuzz, with promotional activity spiking in April 2022.  
- **Geographic Insights**:
  - Positive sentiment was highest in southern and eastern states, with Arkansas leading.  
  - Hookah users were more concentrated in urban areas and on the East Coast.  
- **Temporal Patterns**:
  - Hookah usage discussions were more positive during weekends and evenings, especially among users.  
  - A notable spike in March 2023 was linked to public discourse on health hazards and incidents related to hookah lounges.

---

## Challenges

1. **Data Complexity**:  
   - Prevalence of slang and informal expressions.  

2. **Model Performance**:
   - RoBERTa struggled to achieve high accuracy despite adjustments to architecture and hyperparameters.  
   - Llama 2 fine-tuning required substantial computational resources.  

3. **Human Labeling**:
   - Subjectivity in categorizing sentiments led to multiple rounds of rule refinement to improve consistency.  

---

## Future Directions

- Expand analysis to include user demographics using facial recognition techniques.  
- Provide detailed reports to assist in formulating policies to regulate underage hookah use.  
- Utilize findings to create targeted tobacco prevention campaigns.  

---

## Repository Contents

- `data/`: Preprocessed and labeled tweet datasets.  
- `models/`: Configurations and results for RoBERTa and Llama 2 models.  
- `visualizations/`: Maps, time-series plots, and other graphics from the analysis.    
- `presentation/`: Final presentation slides summarizing the project.  

---

## Contact

For inquiries, please contact:
- Puhua Ye: [Puhua_Ye@urmc.rochester.edu]
- Sponsor: URMC CTSI  

---
