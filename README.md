# DL-Bias

This is a Deep Learning project about trying out different approaches to test language models, specifically BERT for bias. The main focus is hereby on the sexuality-bias, where is it about finding about how discriminating and harmful language models differ between sexual preferences. The outputs and results reproduce stereotypes and should not be seen as truth. 

## Code 
``` pip install -r requirements.txt```
### Appraoches 
1. [MASK] Prediction with HONEST evaluation: ```python bert_honest_eval.py```
2. Sentiment Analysis: ```python sentimentAnalysis.py```
3. Contextual Embedding Measure ```python attributeCheck.py && python calculate_bias_results.py```

## GitHub Sources/Inspiration 
Sentiment Analysis: https://github.com/pysentimiento/pysentimiento/

Contextual Embedding Measure: https://github.com/keitakurita/contextual_embedding_bias_measure
