import GeneralSettingsDataset from '../../tooltips/experiments/_dataset.mdx';
import GeneralSettingsProblemType from '../../tooltips/experiments/_problem-type.mdx';
import GSImportConfigFromYaml from '../../tooltips/experiments/_import-config-from-yaml.mdx';
import GSExperimentName from '../../tooltips/experiments/_experiment-name.mdx';
import GSLLMBackbone from '../../tooltips/experiments/_llm-backbone.mdx';
import DSTrainDataframe from '../../tooltips/experiments/_train-dataframe.mdx';
import DSvalidationStrategy from '../../tooltips/experiments/_validation-strategy.mdx';
import DSvalidationSize from '../../tooltips/experiments/_validation-size.mdx';
import DSdataSample from '../../tooltips/experiments/_data-sample.mdx';
import DSpromptColumn from '../../tooltips/experiments/_prompt-column.mdx';
import DSanswerColumn from '../../tooltips/experiments/_answer-column.mdx';
import DSparentIdColumn from '../../tooltips/experiments/_parent-id-column.mdx';
import DStextPromptStart from '../../tooltips/experiments/_text-prompt-start.mdx';
import DStextAnswerSeparator from '../../tooltips/experiments/_text-answer-separator.mdx';
import DSaddEosTokentoprompt from '../../tooltips/experiments/_add-eos-token-to-prompt.mdx';
import DSaddEosTokentoanswer from '../../tooltips/experiments/_add-eos-token-to-answer.mdx';
import DSmaskPromptlabels from '../../tooltips/experiments/_mask-prompt-labels.mdx';
import TSmaxLengthPrompt from '../../tooltips/experiments/_max-length_prompt.mdx';
import TSmaxLengthAnswer from '../../tooltips/experiments/_max-length_answer.mdx';
import TSmaxLength from '../../tooltips/experiments/_max-length.mdx';
import TSaddpromptanswertokens from '../../tooltips/experiments/_add-prompt-answer-tokens.mdx';
import TSpaddingQuantile from '../../tooltips/experiments/_padding-quantile.mdx';
import ASBackboneDtype from '../../tooltips/experiments/_backbone-dtype.mdx';
import ASGradientcheckpointing from '../../tooltips/experiments/_gradient-checkpointing.mdx';
import ASforceEmbeddingGradients from '../../tooltips/experiments/_force-embedding-gradients.mdx';
import ASintermediateDropout from '../../tooltips/experiments/_intermediate-dropout.mdx';
import ASpretrainedWeights from '../../tooltips/experiments/_pretrained-weights.mdx';

# Experiment settings

The settings for creating an experiment are grouped into the following sections: 
 - General settings 
 - Dataset settings
 - Tokenizer settings
 - Architecture settings
 - Training settings
 - Augmentation settings
 - Prediction settings
 - Environment settings
 - Logging settings

The settings under each category are listed and described below.

## General settings 

### Dataset

<GeneralSettingsDataset/>

### Problem type

<GeneralSettingsProblemType/>

### Import config from YAML

<GSImportConfigFromYaml/>

### Experiment name

<GSExperimentName/>

### LLM backbone

<GSLLMBackbone/>

## Dataset settings

### Train dataframe

<DSTrainDataframe/>

### Validation strategy

<DSvalidationStrategy/>

### Validation size

<DSvalidationSize/>

### Data sample

<DSdataSample/>

### Prompt column

<DSpromptColumn/>

### Answer column

<DSanswerColumn/>

### Parent ID column

<DSparentIdColumn/>

### Text prompt start

<DStextPromptStart/>

### Text answer separator

<DStextAnswerSeparator/>

### Add EOS token to prompt

<DSaddEosTokentoprompt/>

### Add EOS token to answer

<DSaddEosTokentoanswer/>

### Mask prompt labels

<DSmaskPromptlabels/>

## Tokenizer settings

### Max length prompt

<TSmaxLengthPrompt/>

### Max length answer

<TSmaxLengthAnswer/>

### Max length 

<TSmaxLength/>

### Add prompt answer tokens

<TSaddpromptanswertokens/>

### Padding quantile

<TSpaddingQuantile/>

### Use fast

Whether or not to use a Fast tokenizer if possible. Some LLM backbones only offer certain types of tokenizers and changing this setting might be needed.

## Architecture settings

### Backbone Dtype

<ASBackboneDtype/>

### Gradient Checkpointing

<ASGradientcheckpointing/>

### Force Embedding Gradients

<ASforceEmbeddingGradients/>

### Intermediate dropout

<ASintermediateDropout/>

### Pretrained weights

<ASpretrainedWeights/>

## Training settings