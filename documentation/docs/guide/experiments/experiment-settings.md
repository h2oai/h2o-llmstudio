---
description: All the settings needed for creating an experiment are explored in this page.
---
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
import DSsystemColumn from '../../tooltips/experiments/_system-column.mdx';
import DSanswerColumn from '../../tooltips/experiments/_answer-column.mdx';
import DSparentIdColumn from '../../tooltips/experiments/_parent-id-column.mdx';
import DStextPromptStart from '../../tooltips/experiments/_text-prompt-start.mdx';
import DStextAnswerSeparator from '../../tooltips/experiments/_text-answer-separator.mdx';
import DSadaptiveKlControl from '../../tooltips/experiments/_adaptive-kl-control.mdx';
import DSaddEosTokentoprompt from '../../tooltips/experiments/_add-eos-token-to-prompt.mdx';
import DSaddEosTokentoanswer from '../../tooltips/experiments/_add-eos-token-to-answer.mdx';
import DSmaskPromptlabels from '../../tooltips/experiments/_mask-prompt-labels.mdx';
import TSmaxLengthPrompt from '../../tooltips/experiments/_max-length-prompt.mdx';
import TSmaxLengthAnswer from '../../tooltips/experiments/_max-length-answer.mdx';
import TSmaxLength from '../../tooltips/experiments/_max-length.mdx';
import TSaddpromptanswertokens from '../../tooltips/experiments/_add-prompt-answer-tokens.mdx';
import TSpaddingQuantile from '../../tooltips/experiments/_padding-quantile.mdx';
import TSuseFast from '../../tooltips/experiments/_use-fast.mdx';
import ASBackboneDtype from '../../tooltips/experiments/_backbone-dtype.mdx';
import ASGradientcheckpointing from '../../tooltips/experiments/_gradient-checkpointing.mdx';
import ASforceEmbeddingGradients from '../../tooltips/experiments/_force-embedding-gradients.mdx';
import ASintermediateDropout from '../../tooltips/experiments/_intermediate-dropout.mdx';
import ASpretrainedWeights from '../../tooltips/experiments/_pretrained-weights.mdx';
import TSoptimizer from '../../tooltips/experiments/_optimizer.mdx';
import TSlossfunction from '../../tooltips/experiments/_loss-function.mdx';
import TSlearningRate from '../../tooltips/experiments/_learning-rate.mdx';
import TSuseflashattention2 from '../../tooltips/experiments/_use-flash-attention-2.mdx';
import TSbatchSize from '../../tooltips/experiments/_batch-size.mdx';
import TSepochs from '../../tooltips/experiments/_epochs.mdx';
import TSschedule from '../../tooltips/experiments/_schedule.mdx';
import TSwarmupEpochs from '../../tooltips/experiments/_warmup-epochs.mdx';
import TSweightDecay from '../../tooltips/experiments/_weight-decay.mdx';
import TSGradientclip from '../../tooltips/experiments/_gradient-clip.mdx';
import TSgradAccumulation from '../../tooltips/experiments/_grad-accumulation.mdx';
import TSlora from '../../tooltips/experiments/_lora.mdx';
import TSloraR from '../../tooltips/experiments/_lora-r.mdx';
import TSloraAlpha from '../../tooltips/experiments/_lora-alpha.mdx';
import TSloraDropout from '../../tooltips/experiments/_lora-dropout.mdx';
import TSloraTargetModules from '../../tooltips/experiments/_lora-target-modules.mdx';
import TSsavebestcheckpoint from '../../tooltips/experiments/_save-best-checkpoint.mdx';
import TSevaluationepochs from '../../tooltips/experiments/_evaluation-epochs.mdx';
import TSevaluationbeforetraining from '../../tooltips/experiments/_evaluate-before-training.mdx';
import TStrainvalidationdata from '../../tooltips/experiments/_train-validation-data.mdx';
import TSuseRHLF from '../../tooltips/experiments/_use-rlhf.mdx';
import TSrewardModel from '../../tooltips/experiments/_reward-model.mdx';
import TSinitialKlCoefficient from '../../tooltips/experiments/_initial-kl-coefficient.mdx';
import TSklTarget from '../../tooltips/experiments/_kl-target.mdx';
import TSklHorizon from '../../tooltips/experiments/_kl-horizon.mdx';
import TSadvantagesGamma from '../../tooltips/experiments/_advantages-gamma.mdx';
import TSadvantagesLambda from '../../tooltips/experiments/_advantages-lambda.mdx';
import TSppoClipPolicy from '../../tooltips/experiments/_ppo-clip-policy.mdx';
import TSppoClipValue from '../../tooltips/experiments/_ppo-clip-value.mdx';
import TSscalingFactorValueLoss from '../../tooltips/experiments/_scaling-factor-value-loss.mdx';
import TSppoEpochs from '../../tooltips/experiments/_ppo-epochs.mdx';
import TSppoBatchSize from '../../tooltips/experiments/_ppo-batch-size.mdx';
import TSppoGenerateTemp from '../../tooltips/experiments/_ppo-generate-temperature.mdx';
import TSoffloadRewardModel from '../../tooltips/experiments/_offload-reward-model.mdx';
import AStokenmaskprobability from '../../tooltips/experiments/_token-mask-probability.mdx';
import ASskipParentprobability from '../../tooltips/experiments/_skip-parent-probability.mdx';
import ASrandomparentprobability from '../../tooltips/experiments/_random-parent-probability.mdx';
import ASneftunenoisealpha from '../../tooltips/experiments/_neftune_noise_alpha.mdx';
import PSmetric from '../../tooltips/experiments/_metric.mdx';
import PSmetricgptmodel from '../../tooltips/experiments/_metric-gpt-model.mdx';
import PSmetricgpttemplate from '../../tooltips/experiments/_metric-gpt-template.mdx';
import PSminlengthinference from '../../tooltips/experiments/_min-length-inference.mdx';
import PSmaxlengthinference from '../../tooltips/experiments/_max-length-inference.mdx';
import PSbatchsizeinference from '../../tooltips/experiments/_batch-size-inference.mdx';
import PSdosample from '../../tooltips/experiments/_do-sample.mdx';
import PSnumbeams from '../../tooltips/experiments/_num-beams.mdx';
import PStemperature from '../../tooltips/experiments/_temperature.mdx';
import PSrepetitionpenalty from '../../tooltips/experiments/_repetition-penalty.mdx';
import PSstoptokens from '../../tooltips/experiments/_stop-tokens.mdx';
import PStopk from '../../tooltips/experiments/_top-k.mdx';
import PStopp from '../../tooltips/experiments/_top-p.mdx';
import ESgpus from '../../tooltips/experiments/_gpus.mdx';
import ESmixedprecision from '../../tooltips/experiments/_mixed-precision.mdx';
import EScompilemodel from '../../tooltips/experiments/_compile-model.mdx';
import ESfindunusedparameters from '../../tooltips/experiments/_find-unused-parameters.mdx';
import EStrustremotecode from '../../tooltips/experiments/_trust-remote-code.mdx';
import EShuggingfacebranch from '../../tooltips/experiments/_huggingface-branch.mdx';
import ESnumofworkers from '../../tooltips/experiments/_number-of-workers.mdx';
import ESseed from '../../tooltips/experiments/_seed.mdx';
import LSlogger from '../../tooltips/experiments/_logger.mdx';
import LSneptuneproject from '../../tooltips/experiments/_neptune-project.mdx';

# Experiment settings

The settings for creating an experiment are grouped into the following sections: 
 - [General settings](#general-settings) 
 - [Dataset settings](#dataset-settings)
 - [Tokenizer settings](#tokenizer-settings)
 - [Architecture settings](#architecture-settings)
 - [Training settings](#training-settings)
 - [Augmentation settings](#augmentation-settings)
 - [Prediction settings](#prediction-settings)
 - [Environment settings](#environment-settings)
 - [Logging settings](#logging-settings)

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

### System column

<DSsystemColumn/>

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

## Adaptive Kl control

<DSadaptiveKlControl/>

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

<TSuseFast />

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

### Loss function

<TSlossfunction/>

### Optimizer

<TSoptimizer />

### Learning rate

<TSlearningRate/>

### Use Flash Attention 2

<TSuseflashattention2/>

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

<TSGradientclip/>

### Grad accumulation

<TSgradAccumulation/>

### Lora

<TSlora/>

### Lora R

<TSloraR/>

### Lora Alpha

<TSloraAlpha/>

### Lora dropout

<TSloraDropout/>

### Lora target modules

<TSloraTargetModules/>

### Save best checkpoint

<TSsavebestcheckpoint/>

### Evaluation epochs

<TSevaluationepochs/>

### Evaluate before training

<TSevaluationbeforetraining/>

### Train validation data

<TStrainvalidationdata/>

### Use RLHF

<TSuseRHLF/>

### Reward model

<TSrewardModel/>

### Adaptive KL control

<DSadaptiveKlControl/>

### Initial KL coefficient

<TSinitialKlCoefficient/>

### KL target

<TSklTarget/>

### KL Horizon

<TSklHorizon/>

### Advantages gamma

<TSadvantagesGamma/>

### Advantages Lambda

<TSadvantagesLambda/>

### PPO clip policy

<TSppoClipPolicy/>

### PPO clip value

<TSppoClipValue/>

### Scaling factor value loss

<TSscalingFactorValueLoss/>

### PPO epochs

<TSppoEpochs/>

### PPO Batch Size

<TSppoBatchSize/>

### PPO generate temperature

<TSppoGenerateTemp/>

### Offload reward model

<TSoffloadRewardModel/>

## Augmentation settings

### Token mask probability

<AStokenmaskprobability/>

### Skip parent probability

<ASskipParentprobability/>

### Random parent probability

<ASrandomparentprobability/>

### Neftune noise alpha

<ASneftunenoisealpha/>

## Prediction settings

### Metric

<PSmetric/>

### Metric GPT model

<PSmetricgptmodel/>

### Metric GPT template

<PSmetricgpttemplate/>

### Min length inference

<PSminlengthinference/>

### Max length inference

<PSmaxlengthinference/>

### Batch size inference

<PSbatchsizeinference/>

### Do sample

<PSdosample/>

### Num beams

<PSnumbeams/>

### Temperature 

<PStemperature/>

### Repetition penalty

<PSrepetitionpenalty/>

### Stop tokens

<PSstoptokens/>

### Top K

<PStopk />

### Top P

<PStopp />

## Environment settings

### GPUs

<ESgpus/>

### Mixed precision

<ESmixedprecision/>

### Compile model

<EScompilemodel/>

### Find unused parameters

<ESfindunusedparameters/>

### Trust remote code

<EStrustremotecode/>

### Huggingface branch

<EShuggingfacebranch/>

### Number of workers

<ESnumofworkers/>

### Seed

<ESseed/>

## Logging settings

### Logger

<LSlogger/>

### Neptune project

<LSneptuneproject/>

