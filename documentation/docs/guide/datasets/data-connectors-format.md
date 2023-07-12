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

- During an experiment you can adapt the data representation with the following settings:
    - **Prompt Column:** The column in the dataset containing the user prompt.
    - **Answer Column:** The column in the dataset containing the expected output.
    - **Parent Id Column:** An optional column specifying the parent id to be used for chained conversations. The value of this column needs to match an additional column with the name `id`. If provided, the prompt will be concatenated after preceeding parent rows.

## Example data

H2O LLM Studio provides a sample dataset (converted dataset from [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1))
that can be downloaded [here](https://www.kaggle.com/code/philippsinger/openassistant-conversations-dataset-oasst1?scriptVersionId=127047926). It is recommended to use `train_full.csv` for training. This dataset is also downloaded and prepared by default when first starting the GUI. Multiple dataframes can be uploaded into a single dataset by uploading a `.zip` archive.