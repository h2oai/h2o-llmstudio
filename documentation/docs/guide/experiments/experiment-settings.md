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
import TSoptimizer from '../../tooltips/experiments/_optimizer.mdx';
import TSlearningRate from '../../tooltips/experiments/_learning-rate.mdx';
import TSbatchSize from '../../tooltips/experiments/_batch-size.mdx';
import TSepochs from '../../tooltips/experiments/_epochs.mdx';
import TSschedule from '../../tooltips/experiments/_schedule.mdx';
import TSwarmupEpochs from '../../tooltips/experiments/_warmup-epochs.mdx';
import TSweightDecay from '../../tooltips/experiments/_weight-decay.mdx';

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

### Optimizer

Defines the algorithm or method (optimizer) to use for model training. The selected algorithm or method defines how the model should change the attributes of the neural network, such as weights and learning rate. Optimizers solve optimization problems and make more accurate updates to attributes to reduce learning losses.


Options
 - **Adadelta**
    -  To learn about Adadelta, see <a href="https://arxiv.org/abs/1212.5701" target="_blank" >ADADELTA: An Adaptive Learning Rate Method</a>. 
 - **Adam**
    - To learn about Adam, see <a href="https://arxiv.org/abs/1412.6980" target="_blank" >Adam: A Method for Stochastic Optimization</a>. 
 - **AdamW**
    - To learn about AdamW, see <a href="https://arxiv.org/abs/1711.05101" target="_blank" >Decoupled Weight Decay Regularization</a>.
 - **AdamW8bit**
    - To learn about AdamW, see <a href="https://arxiv.org/abs/1711.05101" target="_blank" >Decoupled Weight Decay Regularization</a>.
 - **RMSprop** 
    - To learn about RMSprop, see <a href="https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf" target="_blank" >Neural Networks for Machine Learning</a>.
 - **SGD** 
    - H2O LLM Studio uses a stochastic gradient descent optimizer.

### Learning rate

<TSlearningRate/>

### Batch size

<TSbatchSize/>

### Epochs

<TSepochs/>

### Schedule

<TSschedule/>

### Warmup epochs

<TSwarmupEpochs/>

### Weight decay

<TSweightDecay/>

### Gradient clip

### Grad accumulation

### Lora

### Lora R

### Lora Alpha

### Lora dropout

### Lora target modules

### Save best checkpoint

### Evaluation epochs

### Evaluate before training

### Train validation data