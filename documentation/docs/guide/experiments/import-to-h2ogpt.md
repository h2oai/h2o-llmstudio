import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Import a model to h2oGPT

Once the model has been fine-tuned using H2O LLM Studio, you can then use [h2oGPT](https://github.com/h2oai/h2ogpt/blob/main/README.md) to query, summarize, and chat with your model. 

The most common method to get the model from H2O LLM Studio over to h2oGPT, is to import it into h2oGPT via HuggingFace. However, if your data is sensitive, you can also choose to download the model locally to your machine, and then import it directly into h2oGPT. 

You can use any of the following methods: 

- Publish the model to HuggingFace and import the model from HuggingFace
- Download the model and import it to h2oGPT by specifying the local folder path
- Download the model and upload it to h2oGPT using the file upload option on the UI
- Pull a model from a Github repository or a resolved web link

## Steps

1. [Publish the model to HuggingFace](export-trained-model.md) or [download the model locally](export-trained-model.md#download-a-model). 

2. If you opt to download the model, make sure you extract the downloaded .zip file. 

3. Use the following command to import it into h2oGPT.
    ```
    python generate.py --base_model=[link_or_path_to_folder]
    ```

    :::note Examples
    <Tabs className="unique-tabs">
    <TabItem value="Example1" label="From HuggingFace">
    <pre><code>python generate.py --base_model=HuggingFaceH4/zephyr-7b-beta</code></pre>
    </TabItem>
    <TabItem value="Example2" label="From a Local File">
    <pre><code>python generate.py --base_model=zephyr-7b-beta.Q5_K_M.gguf</code></pre>
    </TabItem>
    <TabItem value="Example3" label="From a Repository">
    <pre><code>python generate.py --base_model=TheBloke/zephyr-7B-beta-AWQ</code></pre>
    </TabItem>
    </Tabs>
    :::

:::info
For more information, see the [h2oGPT documentation](https://github.com/h2oai/h2ogpt/blob/main/docs/FAQ.md#adding-models). 