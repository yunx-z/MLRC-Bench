#/bin/bash

# auto-gpt
# pip install -r Auto-GPT/requirements.txt

# crfm api
# pip install crfm-helm

# ML dependencies
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
uv pip install torch torchvision torchaudio
uv pip install -r requirements.txt
uv pip install typing-inspect==0.8.0 typing_extensions==4.5.0
uv pip install pydantic -U
uv pip install -U numpy
uv pip install --force-reinstall charset-normalizer==3.1.0
