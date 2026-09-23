import os
import shutil
import random
import torch
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from fast_bss_eval import si_sdr

def plot_spectrogram(audio_path, out_path):
    y, sr = librosa.load(audio_path, sr=16000)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    # EchoSet/speech mostly below 8kHz, but 4kHz or 8kHz is fine. 
    # We will plot 0-8kHz
    fig, ax = plt.subplots(figsize=(5, 3))
    librosa.display.specshow(D, sr=sr, y_axis='linear', x_axis='time', ax=ax, cmap='magma')
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

def main():
    test_root = "/mnt/c/Users/jerry/Desktop/Speech/dataset/SS/EchoSet_extracted/test/PuKPg4mmafe/0_0_lounge"
    folders = [f.path for f in os.scandir(test_root) if f.is_dir()]
    random.seed(42)
    selected = random.sample(folders, 5)
    
    docs_audio = "docs/audio"
    os.makedirs(docs_audio, exist_ok=True)
    
    # We will run inference.py as a subprocess to keep it simple, 
    # then load the outputs and align them.
    for i, f_path in enumerate(selected):
        ex_num = i + 4
        mix_path = os.path.join(f_path, "mix.wav")
        s1_path = os.path.join(f_path, "spk1_reverb.wav")
        s2_path = os.path.join(f_path, "spk2_reverb.wav")
        
        # Run inference
        out_dir = f"tmp_ex{ex_num}"
        cmd = f"python inference.py --config checkpoints/e266_config_historical.yaml --checkpoint checkpoints/seal-small-e266.tar --legacy-e266 --audio {mix_path} --output-dir {out_dir}"
        os.system(cmd)
        
        # Load audio to align
        mix, sr = torchaudio.load(mix_path)
        ref1, _ = torchaudio.load(s1_path)
        ref2, _ = torchaudio.load(s2_path)
        est1, _ = torchaudio.load(f"{out_dir}/source1.wav")
        est2, _ = torchaudio.load(f"{out_dir}/source2.wav")
        
        ref = torch.cat([ref1, ref2], dim=0)
        est = torch.cat([est1, est2], dim=0)
        
        # Compute SI-SDR for both permutations
        # perm1: est1->ref1, est2->ref2
        sdr1 = si_sdr(est1.unsqueeze(0), ref1.unsqueeze(0)).mean() + si_sdr(est2.unsqueeze(0), ref2.unsqueeze(0)).mean()
        # perm2: est1->ref2, est2->ref1
        sdr2 = si_sdr(est2.unsqueeze(0), ref1.unsqueeze(0)).mean() + si_sdr(est1.unsqueeze(0), ref2.unsqueeze(0)).mean()
        
        if sdr2 > sdr1:
            est1, est2 = est2, est1
            print(f"Ex {ex_num} swapped")
            
        # Save aligned wavs
        ex_prefix = os.path.join(docs_audio, f"ex{ex_num}")
        torchaudio.save(f"{ex_prefix}_mix.wav", mix, sr)
        torchaudio.save(f"{ex_prefix}_s1_clean.wav", ref1, sr)
        torchaudio.save(f"{ex_prefix}_s2_clean.wav", ref2, sr)
        torchaudio.save(f"{ex_prefix}_s1_seal.wav", est1, sr)
        torchaudio.save(f"{ex_prefix}_s2_seal.wav", est2, sr)
        
        # Generate spectrograms
        plot_spectrogram(f"{ex_prefix}_mix.wav", f"{ex_prefix}_mix.png")
        plot_spectrogram(f"{ex_prefix}_s1_clean.wav", f"{ex_prefix}_s1_clean.png")
        plot_spectrogram(f"{ex_prefix}_s2_clean.wav", f"{ex_prefix}_s2_clean.png")
        plot_spectrogram(f"{ex_prefix}_s1_seal.wav", f"{ex_prefix}_s1_seal.png")
        plot_spectrogram(f"{ex_prefix}_s2_seal.wav", f"{ex_prefix}_s2_seal.png")
        
        # cleanup
        shutil.rmtree(out_dir)

if __name__ == "__main__":
    main()
