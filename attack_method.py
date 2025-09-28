# attack_method.py

import torch
import torch.nn.functional as F
import librosa

from sem import small_energy_masking


class AdversarialAttacker:
    def __init__(self, speaker_encoder, n_fft=512, hop_length=200, win_length=400, n_mels=64, sr=16000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.speaker_encoder = speaker_encoder.to(self.device).eval()
        self.compute_gradient = loss_function(e, e_tilde)
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sr = sr
        
        self.mel_basis = torch.FloatTensor(
            librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels)
        ).to(self.device)
        self.window = torch.hann_window(self.win_length).to(self.device)
        

    def _preprocess(self, original_speech: torch.Tensor):
        stft = torch.stft(
            original_speech.to(self.device), n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window, return_complex=True,
            pad_mode='reflect', center=True, normalized=True
        )
        spec_power = torch.abs(stft) ** 2
        phase = torch.angle(stft)
        return spec_power, phase
    

    def _reconstruct(self, spec_adv: torch.Tensor, phase: torch.Tensor, original_length: int):
        spec_adv_mag = torch.sqrt(spec_adv)
        spec_complex = spec_adv_mag * torch.exp(1j * phase)
        
        waveform = torch.istft(
            spec_complex, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window, length=original_length,
            center=True, normalized=True
        )
        return waveform.detach().cpu()
    

    def mep(self, original_speech: torch.Tensor, epsilon: float = 0.0002) -> torch.Tensor:
        original_length = original_speech.size(-1)
        spec_power, phase = self._preprocess(original_speech)
        spec_power_masked = small_energy_masking(spec_power)

        with torch.no_grad():
            mel_spec = torch.matmul(self.mel_basis, spec_power)
            log_mel = torch.log(mel_spec + 1e-9).squeeze(0)
            e = F.normalize(self.speaker_encoder(log_mel), p=2, dim=1)

        spec_adv = spec_power.clone().detach().requires_grad_(True)
        
        current_mel = torch.matmul(self.mel_basis, spec_adv)
        current_log_mel = torch.log(current_mel + 1e-9).squeeze(0)
        e_tilde = F.normalize(self.speaker_encoder(current_log_mel), p=2, dim=1)
        
        grad, _ = self.compute_gradient(e, e_tilde)
        
        grad_upsampled = F.interpolate(
            grad.unsqueeze(0).unsqueeze(0), size=(spec_adv.size(1), spec_adv.size(2)),
            mode='bilinear', align_corners=False
        ).squeeze()
        
        grad = grad_upsampled.unsqueeze(0) * spec_power_masked
        perturbation = epsilon * grad.sign()
        
        with torch.no_grad():
            spec_adv = torch.clamp(spec_adv + perturbation, min=1e-6)
        
        return self._reconstruct(spec_adv, phase, original_length)
    

    def imep(self, original_speech: torch.Tensor, num_iterations: int = 20, epsilon: float = 0.0002, alpha: float = 0.00001) -> torch.Tensor:
        original_length = original_speech.size(-1)
        spec_power, phase = self._preprocess(original_speech)
        spec_power_masked = small_energy_masking(spec_power)
        
        with torch.no_grad():
            mel_spec = torch.matmul(self.mel_basis, spec_power)
            log_mel = torch.log(mel_spec + 1e-9).squeeze(0)
            e = F.normalize(self.speaker_encoder(log_mel), p=2, dim=1)

        spec_adv = spec_power.clone().detach()

        for _ in range(num_iterations):
            spec_adv.requires_grad = True
            
            current_mel = torch.matmul(self.mel_basis, spec_adv)
            current_log_mel = torch.log(current_mel + 1e-9).squeeze(0)
            e_tilde = F.normalize(self.speaker_encoder(current_log_mel), p=2, dim=1)

            grad, _ = self.compute_gradient(e, e_tilde)

            grad_upsampled = F.interpolate(
                grad.unsqueeze(0).unsqueeze(0), size=(spec_adv.size(1), spec_adv.size(2)),
                mode='bilinear', align_corners=False
            ).squeeze()

            grad = grad_upsampled.unsqueeze(0) * spec_power_masked
            perturbation = alpha * grad.sign()

            with torch.no_grad():
                spec_adv = spec_adv + perturbation
                spec_adv = torch.max(spec_power - epsilon, torch.min(spec_power + epsilon, spec_adv))
                spec_adv = torch.clamp(spec_adv, min=1e-6)

        return self._reconstruct(spec_adv, phase, original_length)


