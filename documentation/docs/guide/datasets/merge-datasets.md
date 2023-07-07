import Icon from "@material-ui/core/Icon";

# Merge datasets

H2O LLM Studio enables you to merge imported datasets into one main dataset. This functionality can be used to merge training and validation data together into one dataset or extend your existing dataset with more data and increase your dataset size. 

:::info
H2O LLM Studio does not merge dataset files in the sense that rows are combined, and duplicate rows are removed. "Merge", in this case, refers to bringing the dataset files a dataset might have to a single dataset (another dataset), continuing other dataset files already.
:::

Generally, you might want to merge datasets in H2O LLM Studio to have both the training data .csv and validation data .csv in one final dataset. 

1. On the H2O LLM Studio left-navigation pane, click **View datasets**.  
2. Click the <Icon>more_vert</Icon> Kebab menu of the dataset you want to merge with. 
3. Click **Edit dataset**. 
4. Click **Merge with existing dataset**.
5. Select the dataset you want that you want to merge with. 
    ![merge-datasets](merge-datasets.png)
6. Click **Merge**.
7. Adjust the dataset configuration if needed. For more information about the configurations, see [Configure dataset](./import-dataset#configure-dataset). 
8. Click **Continue**.
9. Review the text to ensure that the input and output is as intended, and then click **Continue**.

Your datasets are now merged. 

:::info
Alternatively, you can also merge datasets at the point of [importing a dataset](./import-dataset) or combine both datasets (.csv files) into a `.zip` file before uploading it as a whole dataset. 
:::



