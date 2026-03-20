
# Setup & Train on Server

```bash
# Clone & install
git clone https://github.com/lam1101999/ResNet.git ResNet && cd ResNet
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv sync

# HuggingFace login (accept terms at huggingface.co/datasets/ILSVRC/imagenet-1k)
uv run huggingface-cli login
token: <your-hf-token>

# Train ImageNet (run inside tmux to survive disconnect)
uv run python main.py \
    --model resnet18 resnet34 resnet50 resnet101 resnet152 plain18 plain34 vgg16 vgg16_bn vgg19 vgg19_bn \
    --dataset imagenet --epochs 200 --batch-size 256 --workers 8
```