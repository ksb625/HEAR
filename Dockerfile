# -------------------------------------------------------
# Base Image: PyTorch + CUDA + cuDNN
# -------------------------------------------------------
    FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

    # -------------------------------------------------------
    # System packages
    # -------------------------------------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        libasound2-dev \
        build-essential \
        git \
        wget \
        && rm -rf /var/lib/apt/lists/*
    
    # -------------------------------------------------------
    # Python dependencies
    # -------------------------------------------------------
    RUN pip install --upgrade pip
    
    # Install python packages
    RUN pip install \
        librosa==0.10.1 \
        soundfile>=0.12.1 \
        torchaudio==2.2.0 \
        numpy>=1.24.0 \
        scipy>=1.10.0 \
        pandas>=2.0.0 \
        scikit-learn>=1.3.0 \
        pystoi>=0.3.3 \
        pesq>=0.0.4 \
        torch-audiomentations>=0.11.0 \
        stable-baselines3>=2.0.0 \
        tensorboard>=2.13.0 \
        matplotlib>=3.7.0 \
        tqdm>=4.65.0 \
        jupyter>=1.0.0 \
        notebook>=6.5.0 \
        requests>=2.31.0 \
        wandb>=0.15.0
    
    # -------------------------------------------------------
    # Default command (override if needed)
    # -------------------------------------------------------
    CMD ["/bin/bash"]
    