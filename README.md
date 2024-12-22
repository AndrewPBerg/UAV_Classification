# UAV Classification

This repository contains a machine learning project for UAV (Unmanned Aerial Vehicle) classification using deep learning techniques. The project is containerized using Docker and supports experiment tracking with Weights & Biases.

# **NOTE**

The Datasets used in this project are not included in the repository due to their visibility -> We have decided not to open-source the datasets.

If you would like to use the codebase, please use this example directory to store your datasets. and update the `config.yaml` file to point to your datasets.

## Project Structure

```
├── src/                    # Source code directory
│   ├── main.py            # Main training script
│   ├── script.py          # Utility scripts
│   ├── sweeps.py          # Hyperparameter tuning
│   ├── config.yaml        # Configuration file
│   ├── helper/            # Helper functions
│   ├── notebooks/         # Jupyter notebooks
│   └── run_configs/       # Run configurations
├── .datasets/             # Dataset directory
├── docker-compose.yml     # Docker compose configuration
├── Dockerfile            # Docker build instructions
└── requirements.txt      # Python dependencies
```

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- CUDA-compatible GPU (recommended)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/UAV_Classification_repo.git
cd UAV_Classification_repo
```

2. (Optional) Copy the example environment file and configure your variables:
```bash
cp .env.example .env
```

3. Install dependencies (if running locally):
> Note: If you are running the container, you can skip this step.
```bash
pip install -r requirements.txt
```

4. Build and using Docker:

```bash
docker compose build app
```



## Environment Variables

Create a `.env` file in the root directory with the following variables:
> Note: see setup section for more details

- `WANDB_API_KEY`: Your Weights & Biases API key
- `BOT_TOKEN`: Telegram bot token for notifications (optional)
- `CHAT_ID`: Telegram chat ID for notifications (optional)

## Python Usage

1. Configure your experiment in `src/config.yaml` & `orchestrate.yaml`

2. Run training:
```bash
python src/main.py
```

## Docker Usage
1. Configure your experiment in `src/config.yaml` & `orchestrate.yaml`

2. Run training:
```bash
docker compose run app
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 