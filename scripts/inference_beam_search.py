import torch
import torchaudio
#from torchaudio.models.decoder import ctc_decoder
import argparse
import fairseq
import argparse
import glob
import os
import librosa
import numpy as np
from itertools import groupby
import re
from torchmetrics import CharErrorRate
from pyctcdecode import build_ctcdecoder
#from ctc_decoder import beam_search
   
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_cp", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--path_to_dict", required=True)
    parser.add_argument("--path_to_trans", required=True)
    parser.add_argument("--lm", default=None)
    parser.add_argument("--beam_width", default=None, type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_name", required=True)
    args = parser.parse_args()

    # Define the model
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.path_to_cp], arg_overrides={"data": "dict.ltr"})
    model = model[0]
    model.eval()
    
    # Make the token lists 
    tokens_lst = ["<pad>", "<s>", "</s>", "<unk>"]
    with open(args.path_to_dict) as f:
        for line in f.readlines():
            tokens_lst.append(line[0])
    
    # Build CTC decoder
    decoder = build_ctcdecoder(labels=tokens_lst, kenlm_model_path=args.lm)

    # Obtain data samples as a list
    data = glob.glob(os.path.join(args.wav_dir, "*.wav"), recursive=True)
    
    # Make a look-up table from audio to transcript
    audio_to_trans = {}
    with open(args.path_to_trans) as f:
        for line in f.readlines():
            audio_to_trans[line.split("\t")[0]] = line.split("\t")[1].rstrip("\n")
    
    # Evaluation
    with torch.no_grad():
        wer = 0    
        preds = []
        true = []
        
        results = []

        for sample in data:
            audio_name = re.findall("\/([\w\d]*).wav", sample)[0]
            audio_name += ".wav"
            actual_transcript = audio_to_trans[audio_name]
    
            waveform, sr = torchaudio.load(sample)
            emission = model(source=waveform, padding_mask=None)
            logits = torch.squeeze(emission["encoder_out"], 1)
            
            beam_trans = decoder.decode(logits.detach().numpy(), beam_width=args.beam_width)
            beam_wer = torchaudio.functional.edit_distance(actual_transcript, beam_trans) / len(actual_transcript)
            
            wer += beam_wer
            preds.append(beam_trans)
            true.append(actual_transcript)
            
            results.append([audio_name + '\t', f'Label: {actual_transcript}' + '\t', f'Pred: {beam_trans}' + '\t', f'WER: {str(beam_wer)}'])

        wer = str(wer / len(true))
        metric_cer = CharErrorRate()
        cer = str(metric_cer(preds, true))         
        print(f"WER:{wer}")

        results.append([f"Total WER: {wer}", f"Total CER: {cer}"])

        name = args.out_name + "_" + str(args.beam_width) + ".txt"

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
            
        out_file = os.path.join(args.out_dir, name)

        with open(out_file, 'w') as f:
            f.write("\n".join(" ".join(line) for line in results))

if __name__ == '__main__':
    main()


