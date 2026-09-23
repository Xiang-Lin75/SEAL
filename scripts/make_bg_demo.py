import os
import torch
import torchaudio
import subprocess
from omegaconf import OmegaConf

def load_model(device):
    # Same logic as inference.py
    from scripts.legacy_e266_adapter import construct_e266_model
    model, _, _, metadata = construct_e266_model(
        device=device,
        checkpoint_path="checkpoints/seal-small-e266.tar",
        config_path="checkpoints/e266_config_historical.yaml",
    )
    model.eval()
    return model

def main():
    docs_bg = "docs/bg-demo"
    os.makedirs(docs_bg, exist_ok=True)
    
    # Download sample1 mixture from TIGER if it doesn't exist
    mix_mp4 = f"{docs_bg}/mixture.mp4"
    if not os.path.exists(mix_mp4):
        os.system(f"curl -s -o {mix_mp4} https://cslikai.cn/TIGER/assets/dnr-demo/sample1/mixture.mp4")
    
    # Extract audio
    mix_wav = f"{docs_bg}/mix.wav"
    os.system(f"ffmpeg -y -i {mix_mp4} -ac 1 -ar 16000 {mix_wav} -loglevel quiet")
    
    # Run SEAL inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    
    mix, sr = torchaudio.load(mix_wav)
    mix = mix.to(device)
    
    with torch.inference_mode():
        output = model(mix)
        # output is (1, 2, L)
        aux = model._last_aux
        sink = aux["sink_waveform"] # (1, 1, L)
        
    # We combine both separated speakers into one "dialog" track
    dialog = output[0, 0] + output[0, 1]
    dialog = dialog.unsqueeze(0).cpu()
    background = sink[0].cpu()
    
    torchaudio.save(f"{docs_bg}/dialog.wav", dialog, sr)
    torchaudio.save(f"{docs_bg}/background.wav", background, sr)
    
    # Generate visualization videos
    def make_vid(wav, out):
        cmd = [
            "ffmpeg", "-y", "-i", wav,
            "-filter_complex", "[0:a]showspectrum=s=640x360:mode=combined:color=magma:slide=scroll[v]",
            "-map", "[v]", "-map", "0:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            out
        ]
        subprocess.run(cmd, check=True)
        
    make_vid(f"{docs_bg}/dialog.wav", f"{docs_bg}/dialog.mp4")
    make_vid(f"{docs_bg}/background.wav", f"{docs_bg}/background.mp4")
    
    # Cleanup wavs
    os.remove(f"{docs_bg}/mix.wav")
    os.remove(f"{docs_bg}/dialog.wav")
    os.remove(f"{docs_bg}/background.wav")

if __name__ == "__main__":
    main()
