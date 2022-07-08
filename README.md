# ASR for Dutch Dysarthric Speech
This is a repository presenting the outcome of my thesis for MSc. Voice Technology at the University of Groningen. **The thesis developed the Dutch dysarthric speech recognition with self-supervised learning (SSL) models, wav2vec 2.0 and XLSR-53.** To the best of the author's knowledge, this is the first attempt to apply SSL models to Dutch dysarthric speech recognition. The implementations are done using [Fariseq library](https://github.com/facebookresearch/fairseq). For more details of the research and experiments, please visit [the paper](https://drive.google.com/file/d/13VkVk38DEhn8TQHVvql1Vhi-HzlmlV9O/view?usp=sharing). With this repo, you can reproduce the evaluation experiment following the instructions below.

#### Repo Structure:
- <sup>1</sup>dataset 
	- speaker_independent: containing data for the speaker-independent experiments
		- pp{PATIENT ID}
			- .wav
			- <sup>2</sup>test.trans.txt
	- speaker_dependent: containing data for the speaker-dependent experiments
		- target: data from target speakers
			- pp{PATIENT ID}
				- .wav
				- test.trans.txt
		- dummy: data from dummy speakers
			- pp{PATIENT ID}
				- .wav
				- test.trans.txt
	- dictionary
		- dict.ltr.txt
		- <sup>3</sup>control.dict.ltr.txt

- <sup>4</sup>models
	- fine-tuned: containing the models fine-tuned without control speakers
		- base
		- xlsr
	- fine-tuned_control: containing the models fine-tuned with control speakers
		- base
		- xlsr
	- speaker_dependent: containing the models re-fine-tuned with a target speaker's utterances
		- pp17
		- pp28
		- pp41

- scripts
	- `inference.py`: main script for the evaluation

<sup>1</sup>The data is extracted from the [Domotica database](https://www.esat.kuleuven.be/psi/spraak/downloads/). Download the evaluation dataset from [here](https://drive.google.com/file/d/1sTwuLjvZLWidG__cZbXhPXFztZd17SUr/view?usp=sharing). Please unzip it locally under the root of the repository.\
<sup>2</sup>This is a file containing transcriptions corresponding to each audio file.\
<sup>3</sup>This is a dictionary for the models fine-tuned with control speakers. It can be the same as the one for the models fine-tuned without control speakers since appeared characters in the dataset are the same. However, the different ordered dictionary was used for the fine-tuning. Therefore, please use this to evaluate the models fine-tuned with control speakers.\
<sup>4</sup>The models can be downloaded from [here](https://drive.google.com/file/d/13p8o2pmzeZTEaoTip6a0xxfgQoaoIWiI/view?usp=sharing). Please unzip it locally under the root of the repository.

## Requirements
1. Please install [Fairseq library](https://github.com/facebookresearch/fairseq) beforehand. For the library installation, please follow the original repo instruction.
Once you have installed the library, modify the file `\fairseq\fairseq\tasks\audio_pretraining.py` to avoid the error due to the parameters mismatch. Please add the following scripts to the file.

```
# copied from fine-tuning cfg
from fairseq.dataclass.configs import GenerationConfig
from typing import Optional, Any

# Add the following parameters to class AudioPretrainingConfig(foarseqDataclass)
# copied from fine-tuning cfg
    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_detok: Optional[str] = field(
        default=None,
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); "
            "required if using --eval-bleu; use 'space' to disable "
            "detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: str = field(
        default="{}", metadata={"help": "args for building the tokenizer, if needed"}
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None, metadata={"help": "remove BPE before computing BLEU"}
    )
    eval_bleu_args: str = field(
        default="{}",
        metadata={
            "help": "generation args for BLUE scoring, e.g., "
            '\'{"beam": 4, "lenpen": 0.6}\''
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq models); "
            "adds 'prev_output_tokens' to input and appends eos to target"
        },
    )
```

2. Please install the dependencies by executing the following command.
```
pip install -r requirements.txt

```
## Evaluation
Before starting the evaluation, make sure that you unzip the data.\
First, move to the scripts directory.
```
cd scripts
```
Then, execute the following command under the root directory of this repository. You have to specify proper arguments based on your environment.
```
python inference_beam_search.py \\
	--path_to_cp /PATH/TO/FINE-TUNED/MODELS/CHECKPOINT \\ 
	--wav_dir /PATH/TO/DATASET \\
	--path_to_trans /PATH/TO/TRANSCRIPTION/FILE \\
	--path_to_dict /PATH/TO/DICTIONARY/FILE \\
	--out_dir /PATH/TO/OUTPUT/DIRECTORY/TO/SAVE/RESULTS \\
	--out_name NAME_OF_THE_OUTPUT_FILE \\
	--beam_width 50
```
Make sure to specify the correct dictionary file. For the evaluation of the speaker-independent models fine-tuned with control speakers, you need to use `control.dict.ltr.txt`. For others, you will use `dict.ltr.txt`.

After the execution, you will obtain the output file `OUT_NAME_BEAM_WIDTH`, which contains each audio file name, true transcription, prediciton, and WER. The last row contains the total WER. The file content looks like the below.
```
SegmentNr_152.wav	 Label: ALADIN STAANDE LAMP OP TWEE	 Pred: ALADIN STANDSLAM OP TWEE	 WER: 0.14814814814814814
SegmentNr_119.wav	 Label: ALADIN DEUR SLAAPKAMER DICHT	 Pred: ALADIN DEUR SLAAPKAMER DICHT	 WER: 0.0
SegmentNr_20.wav	 Label: ALADIN LICHT UIT IN SLAAPKAMER	 Pred: ALADIN LICHT UIT IN SLAAPKAMER	 WER: 0.0
SegmentNr_210.wav	 Label: ALADIN STAANDE LAMP OP TWEE	 Pred: ALADIN STAND LAM OP TWEE	 WER: 0.1111111111111111
SegmentNr_150.wav	 Label: ALADIN LICHT UIT IN SLAAPKAMER	 Pred: ALADIN LICHT IT IN SLAAPKAMER	 WER: 0.03333333333333333
⋮
Total WER: 0.15172580879651584 

```
The evaluation might take up to 10 minutes depending on the environment.

## Results of the Speaker-Dependent ASR
Below is the results of the speaker-dependent ASR for Dutch dysarthric speech. The table presents WER for patients in different severity groups.

| Model | Mild: pp17 | Moderate: pp28 | High: pp41 |
|-------|------|-----------|------|
| XLSR-53 | 10.79 |  15.17  | 17.36 |

For more results of other experiments, please refer to the paper. 

## License
The fine-tuned models are MIT-licenced. For data usage, you have to obey the restrictions imposed by the provider. To summarize, it is NOT allowed to make the speakers identifiable and recognizable by derived creations. For more details, please visit [here](https://www.esat.kuleuven.be/psi/spraak/downloads/).


