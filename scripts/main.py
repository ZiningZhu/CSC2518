import argparse
from jiwer import wer
import librosa  
import numpy as np 
from pathlib import Path
import pickle
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.dataio.preprocess import AudioNormalizer
import torch 
import torchaudio
from tqdm import tqdm

from utils import eer


##########################
###  Configurations
##########################

CLIP_TIME = 10  # Use this length of audio. CLIP_TIME=10 means 10 seconds.
SAMPLE_RATE = 16000
base_dir = "../data/LibriSpeech"


##########################
###  Utility functions
##########################

def get_audio_signals_one_utterance_per_speaker(split_name="dev-clean", clip_time=CLIP_TIME):
    # Collects a clip of CLIP_TIME audio per speaker
    # Return: list of fixed-len (CLIP_TIME * SR) torch.tensors
    collected_utterances = []
    for speaker in Path(base_dir, split_name).iterdir():
        speaker_collected = False 
        if not speaker.is_dir():
            continue 

        for session in speaker.iterdir():
            if speaker_collected:
                break 
            if not session.is_dir():
                continue 

            for utt in session.iterdir():
                if not str(utt).endswith("flac"):
                    continue 

                signal, sr = torchaudio.load(str(utt), channels_first=False)
                assert sr == SAMPLE_RATE, f"Expected sample rate is {SAMPLE_RATE}. Got {sr}"
                audio_len = len(signal) / sr 
                if audio_len > CLIP_TIME:
                    s = signal[: CLIP_TIME * sr, 0]
                    collected_utterances.append(s)
                    speaker_collected = True 
                    break

    return collected_utterances


def get_data_per_speaker(split_name="dev-clean"):
    # collected_utterances: A list (len n_speaker) of list (len n_utterances_this_speaker) of torch.tensor
    # collected_transcripts: A list (len n_speaker) of list (len n_utterances_this_speaker) of str
    collected_utterances = []
    collected_transcripts = []
    total_time = 0
    num_utterances = 0
    for speaker in sorted(Path(base_dir, split_name).iterdir()):
        if not speaker.is_dir():
            continue 
        speaker_utterances = []
        speaker_texts = []
        for session in sorted(speaker.iterdir()):
            if not session.is_dir():
                continue 
            for utt in sorted(session.iterdir()):
                if str(utt).endswith("flac"):
                    # Collect audio
                    signal, sr = torchaudio.load(str(utt), channels_first=False)
                    assert sr == SAMPLE_RATE, f"Expected sample rate is {SAMPLE_RATE}. Got {sr}"
                    speaker_utterances.append(signal)

                    audio_len = len(signal) / sr 
                    total_time += audio_len 
                    num_utterances += 1
                elif str(utt).endswith(".txt"):
                    # Collect transcript
                    for line in utt.read_text().split("\n"):
                        if len(line) == 0:
                            continue 
                        line = " ".join(line.split()[1:]).upper()
                        speaker_texts.append(line)
        assert len(speaker_utterances) == len(speaker_texts), "speaker_utterances len {} != speaker texts len {} for speaker {}".format(len(speaker_utterances), len(speaker_texts), speaker.name)
        collected_utterances.append(speaker_utterances)
        collected_transcripts.append(speaker_texts)
    print ("Collected {} utterances from {} speakers, avg time {:.2f} seconds per utterance.".format(
        num_utterances, len(collected_utterances), total_time / num_utterances
    ))
    return collected_utterances, collected_transcripts


def get_data_per_utterance(split_name="dev-clean"):
    # collected_audios: A list (len n_utterance_total) of torch.tensor
    # transcripts_per_utternace: A list (len n_utterance_total) of str
    audios, transcripts_per_speaker = get_data_per_speaker()
    collected_audios = []
    collected_transcripts = []
    speaker_ids = []
    for i, (a,t) in enumerate(zip(audios, transcripts_per_speaker)):
        collected_audios.extend(a)
        collected_transcripts.extend(t)
        speaker_ids.append(i)
    return collected_audios, collected_transcripts, speaker_ids 


def compute_mfcc(utterances, speaker_ids, fix_len=50):
    # fix_len needs to be set: clip the MFCCs of the longer audios and discard those of the shorter audios. Discard the corresponding speaker_id as well.

    # Return: 
    #   mfccs: (n_speaker, n_mfcc, T)
    #   new_speaker_ids: (n_speaker)
    mfccs = []
    new_speaker_ids = []
    for sid, utt in zip(speaker_ids, utterances):
        y = utt.numpy().T  # (1, T)
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)  # (1, n_mfcc, n_frame)
        if mfcc.shape[-1] >= fix_len:
            mfccs.append(mfcc[0, :, :fix_len])
            new_speaker_ids.append(sid)
    return np.array(mfccs), np.array(new_speaker_ids)


def my_train_test_split(mfccs, speaker_ids):
    # Inputs:
    #   mfccs: np.array(n_speaker, n_mfcc, T)
    #   speaker_ids: np.array(n_speaker)
    # Outputs:
    #   train_mfcc, test_mfcc: (n_speaker/2 * n_mfcc, T)
    #   train_sids, test_sids: np.array(n_speaker/2 * n_mfcc)
    n_speaker, n_mfcc, _ = mfccs.shape[0], mfccs.shape[1], mfccs.shape[2]
    unrolled_mfccs = mfccs.reshape((n_speaker * n_mfcc, -1))
    unrolled_sids = np.array(list(map(
        lambda c: [c] * n_mfcc, speaker_ids
    ))).reshape(-1)

    train_mfcc, test_mfcc, train_sids, test_sids = train_test_split(unrolled_mfccs, unrolled_sids, test_size=0.5, random_state=0, stratify=unrolled_sids)
    return train_mfcc, test_mfcc, train_sids, test_sids


######################
###  Main scripts
######################

def anonymize_cache_speech(section, setting):
    
    cache_path = Path(base_dir, "cache", section, f"{setting}.pkl")
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            checkpoint = pickle.load(f)
            anonymized_utterances = checkpoint["anonymized_utterances"]
            transcripts = checkpoint["transcripts"]
            speaker_ids = checkpoint["speaker_ids"]
        print("Loaded cache from {}".format(cache_path))
    else:
        print("Cache not found. Collecting new caches")

        anonymized_utterances = [] 
        speaker_id = []
        if setting == "baseline":
            anonymized_utterances, transcripts, speaker_ids = get_data_per_utterance()  # List (len n_utterance_total) of torch.tensor

        elif setting == "uniform_noise_per_utterance":
            noise_std = 0.01
            audio, transcripts, speaker_ids = get_data_per_utterance()  # List (len n_utterance_total) of torch.tensor
            anonymized_utterances = []
            for a in tqdm(audio):
                n = torch.normal(mean=torch.zeros_like(a), std=torch.tensor(noise_std))
                anonymized_utterances.append(a+n)

        elif setting == "adaptive_noise_per_utterance":
            scaling_factor = 0.1
            audio, transcripts, speaker_ids = get_data_per_utterance()
            anonymized_utterances = []
            for a in tqdm(audio):
                n = torch.normal(mean=torch.zeros_like(a), std=torch.tensor(scaling_factor) * torch.std(a))
                anonymized_utterances.append(a+n)

        elif setting == "multimodal_noise_per_utterance":
            n_components = 3
            scaling_factor = 0.1
            audio, transcripts, speaker_ids = get_data_per_utterance()
            anonymize_utterances = []
            for a in tqdm(audio):
                gmm = GaussianMixture(n_components=n_components)
                gmm.fit(a.numpy().reshape(-1, 1))
                a_ = a 
                for c in range(n_components):
                    a_ += torch.normal(
                        mean=-torch.ones_like(a) * gmm.means_[c, 0], 
                        std=torch.tensor(np.sqrt(gmm.covariances_[c][0,0])))
                anonymize_utterances.append(a_)
        
        elif setting == "normalize_frequency_per_speaker":
            raise NotImplementedError("TODO")
        elif setting == "normalize_frequency_per_utterance":
            raise NotImplementedError("TODO")
        else:
            raise ValueError(f"Setting {setting} not supported.")
        
        if not cache_path.parents[0].exists():
            cache_path.parents[0].mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({
                "anonymized_utterances": anonymized_utterances,
                "transcripts": transcripts,
                "speaker_ids": speaker_ids}, f)
        print("Anonymization done")
    return anonymized_utterances, transcripts, speaker_ids

def run_speaker_identification_evaluation(utterances, speaker_ids):
    # utterances: list (len N) of torch.tensor of shape (D_i)
    all_X, all_Y = compute_mfcc(utterances, speaker_ids)  # X is(n_speaker, n_mfcc, n_frame)
    train_X, test_X, train_Y, test_Y = my_train_test_split(all_X, all_Y)
    model = MLPClassifier()
    model.fit(train_X, train_Y)
    pred = model.predict(test_X)
    print("Accuracy {:.2f}".format(accuracy_score(test_Y, pred)))
    print("EER {:.4f}".format(eer(test_Y, pred)))

def run_asr_evaluation(utterances, transcripts, verbose=True):
    # utterances: list (len N) of torch.tensor of shape (D_i)
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="../data/pretrained_models/asr-crdnn-rnnlm-librispeech")

    total_edit_dist = 0
    total_num_words = 0
    num_printed = 0
    for signal, transcript in tqdm(zip(utterances, transcripts)):
        batch = AudioNormalizer()(signal, SAMPLE_RATE).unsqueeze(0)
        rel_length = torch.tensor([1.0])

        predicted_words, predicted_tokens = asr_model.transcribe_batch(batch, rel_length)
        N = len(transcript.split())
        edit_distance = N * wer(transcript, hypothesis=predicted_words[0])
        total_num_words += N 
        total_edit_dist += edit_distance 
        if verbose and num_printed < 10:  # Randomly sample 10 to print
            print("transcript: {}\npredicted: {}\ndist={}".format(transcript, predicted_words[0], edit_distance))
            num_printed += 1
    print ("Average WER: {:.4f}".format(total_edit_dist / total_num_words))


if __name__ == "__main__":
    torch.manual_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default="baseline")
    args = parser.parse_args()
    print(args)

    utterances, transcripts, speaker_ids = anonymize_cache_speech("dev-clean", args.setting)
    run_speaker_identification_evaluation(utterances, speaker_ids)
    run_asr_evaluation(utterances, transcripts)
    print("All done!")