# Natural Language Processing by training a supervised BERT model using PyTorch & Tensorflow on AWS
## Predict star rating of labeled Amazon reviews

Predict star rating of labeled Amazon reviews belonging to 3 categories (reducing size of dataset to decrease computation time). The predictors used will be word embeddings derived based on BERT model. Both, transfer learning vs fine tuning options of BERT are explored. REST end point is created for predictions.

### AWS Services used
- SageMaker Jupyter notebooks: Used for everything from model training, optimization, prediction & deployment
- S3: Buckets for storing data & models
- EC2
- CloudWatch
- IAM
- Glue
- Athena

### Libraries & Packages used
