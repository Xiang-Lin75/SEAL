import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa

def get_spec(audio, sr):
    D = librosa.stft(audio, n_fft=512, hop_length=256)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db

def plot_and_save(audio, sr, path):
    S_db = get_spec(audio, sr)
    plt.figure(figsize=(4, 1.5))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def align_and_process(ex_id, mix_path, s1_clean_path, s2_clean_path, o1_path, o2_path, out_dir):
    mix, sr = sf.read(mix_path)
    c1, _ = sf.read(s1_clean_path)
    c2, _ = sf.read(s2_clean_path)
    o1, _ = sf.read(o1_path)
    o2, _ = sf.read(o2_path)
    
    # Calculate MSE to find best alignment
    mse_11 = np.mean((c1 - o1)**2)
    mse_12 = np.mean((c1 - o2)**2)
    mse_21 = np.mean((c2 - o1)**2)
    mse_22 = np.mean((c2 - o2)**2)
    
    if (mse_11 + mse_22) > (mse_12 + mse_21):
        # Swap o1 and o2
        print(f"Swapping outputs for {ex_id}")
        o1, o2 = o2, o1
        
    # Save aligned outputs
    sf.write(os.path.join(out_dir, f"{ex_id}_s1_seal.wav"), o1, sr)
    sf.write(os.path.join(out_dir, f"{ex_id}_s2_seal.wav"), o2, sr)
    sf.write(os.path.join(out_dir, f"{ex_id}_mix.wav"), mix, sr)
    sf.write(os.path.join(out_dir, f"{ex_id}_s1_clean.wav"), c1, sr)
    sf.write(os.path.join(out_dir, f"{ex_id}_s2_clean.wav"), c2, sr)
    
    # Plot spectrograms
    plot_and_save(mix, sr, os.path.join(out_dir, f"{ex_id}_mix.png"))
    plot_and_save(c1, sr, os.path.join(out_dir, f"{ex_id}_s1_clean.png"))
    plot_and_save(c2, sr, os.path.join(out_dir, f"{ex_id}_s2_clean.png"))
    plot_and_save(o1, sr, os.path.join(out_dir, f"{ex_id}_s1_seal.png"))
    plot_and_save(o2, sr, os.path.join(out_dir, f"{ex_id}_s2_seal.png"))

if __name__ == '__main__':
    align_and_process(
        "ex3",
        "/mnt/c/Users/jerry/Desktop/Speech/dataset/SS/EchoSet_extracted/test/VLzqgDo317F/0_7_meetingroom_conferenceroom/1089_2094/mix.wav",
        "/mnt/c/Users/jerry/Desktop/Speech/dataset/SS/EchoSet_extracted/test/VLzqgDo317F/0_7_meetingroom_conferenceroom/1089_2094/spk1_reverb.wav",
        "/mnt/c/Users/jerry/Desktop/Speech/dataset/SS/EchoSet_extracted/test/VLzqgDo317F/0_7_meetingroom_conferenceroom/1089_2094/spk2_reverb.wav",
        "docs/audio/ex3/source1.wav",
        "docs/audio/ex3/source2.wav",
        "docs/audio"
    )
