# Towards More Fine-grained and Reliable NLP Performance Prediction
This repository provides examples of fine-grained performance prediction for different NLP tasks (machine translation, Part-of-Speech, Named Entity Recognition, and Chinese Word segmentation). We also provide the code we used to perform reliability analysis for performance prediction methods through confidence intervals and calibration.


## Scripts
### Fine-grained Performance Prediction

Compare performances of performance prediction of gradient-boosting models and tensor regression models with cross validation:
```python
performance_prediction.ipynb

compare_models(data_dataframe, data_tensor, missing_values, tensor_mapping, num_folds)

```

### Reliability Analysis for Performance Prediction

Perform calibration analysis on performance prediction models through bootstrapping and reconstructing synthetic datasets: 
```python
boosting models: 
calibration_boosting_models.ipynb

tensor regression models:
calibration_tensor_models.ipynb

bootstrap_reconstruct(dataset, tensor_mapping, num_iter=100, task='tsfmt', model='pca')

```
