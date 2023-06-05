# Supported data connectors and format

## Data connectors

H2O LLM Studio supports the following data connectors to access or upload external data sources.

- **Upload**: Upload a local dataset from your machine. 
- **Local**: Specify the file location of the dataset on your machine. 
- **AWS S3 (Amazon AWS S3)**: Connect to an Amazon AWS S3 data bucket. 
- **Kaggle**: Connect to a Kaggle dataset. 

## Data format 

- Each data connector requires either a single `.csv` or `.pq` file, or the data to be in a `.zip` file for a successful import. 

- H2O LLM studio requires a `.csv` file with a minimum of two columns, where one contains the instructions and the other has the model’s expected output. You can also include an additional validation dataframe in the same format or allow for an automatic train/validation split to assess the model’s performance.

- Optionally, a **Parent Id** can be used for training nested data prompts that are linked to a parent question. 