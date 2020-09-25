# Natural Language Processing using BERT, PyTorch, Tensorflow, Spark on AWS
## Predict star rating of labeled Amazon reviews

Predict star rating of labeled Amazon reviews belonging to 3 categories (reducing size of dataset to decrease computation time) using supervised learning. The predictors used will be word embeddings derived based on BERT model. Both, transfer learning vs fine tuning options of BERT are explored. Spark is used for feature transformation & model training on multiple EC2 nodes. REST end point is created for predictions.

### AWS Services used
- SageMaker Jupyter notebooks: Used for everything from model training, optimization, prediction & deployment
- S3: Buckets for storing data & models
- Step Functions: To be able to create an automated pipeline of the entire process from training, testing to predicting from an end-point
- EC2
- Athena: To be able to query S3 data
- IAM: For setting up roles & AWS Services access
- CloudWatch

### Libraries & Packages used
- Python 3.6
- [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html#:~:text=Boto%20is%20the%20Amazon%20Web,level%20access%20to%20AWS%20services.): The AWS SDK for Python
- HuggingFace library [Transformers](https://huggingface.co/transformers/): for BERT
- Tensorflow, PyTorch, Scikit learn
- PySpark, [Deequ](https://github.com/awslabs/deequ): Deequ is Spark library specific for performing data quality checks on AWS
- Pandas, Numpy, Seaborn

### Navigating this repository
- 01_setup
  This folder sets up all the dependencies, creates the s3 bucket, attaches IAM roles & policies, and tests SageMaker jupyter notebook instance
- 02_ingest
  - Copy TSV data to S3: Copies data from public S3 bucket to private S3 bucket.
  - Create Athena Database: This just creates a database and sets up the foundation for converting our otherwise flat tsv files in S3 to queryable using SQL
  - Register S3 TSV with Athena: Attach schema to tsv files in S3. TSV files can now be queried using PyAthena
  - Convert S3 TSV to parquet with Athena: This is just for learning purposes. We will not be using parquet files for training models
  - Query Data with AWS Wrangler: This is just for learning purposes - exploration of an AWS service
- 03_explore
  - Visualize reviews dataset: Perform exploratory data analysis. Use PyAthena to query data in SQL & get results in dataframe, use Seaborn library to visualize results.
  - Analyze Data quality processing job Spark: Spark library deequ is used to process & analyze dataset while preparing it before modeling. Constraints checks on fields are performed.
- 04_prepare
  - Prepare Dataset BERT Adhoc: Performing BERT-specific tokenization & feature transformation in notebook on small dataset before converting it into script format.
  - 


### Data Source
[AWS public S3 bucket](https://s3.amazonaws.com/amazon-reviews-pds/readme.html)
