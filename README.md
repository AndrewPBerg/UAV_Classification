# Spectrogram Dataset 🔉
Images of the feature extracted samples of the Custom UAV dataset can be found at [UAV Classification Dataset](https://github.com/AndrewPBerg/UAV_Classification_Dataset/tree/main/dataset)

# Papers 📜

 1️⃣ 4,500 Seconds [Accepted, Preprint]: [Arxiv 2505.23782](https://arxiv.org/abs/2505.23782) </br>
 2️⃣ 15,500 Seconds [Under Review, Preprint]: Link TBA </br>
 3️⃣ The Unbearable Weight: TBD </br>

# UAV Classification 🛩️

Code repository for UAV (Unmanned Aerial Vehicle) classification using deep learning techniques. The project is containerized using Docker and supports experiment tracking with Weights & Biases.

# **NOTE** 📎

The Datasets used in this project are not included in the repository due to their visibility -> We have decided not to open-source the datasets.

If you would like to use the codebase, please use this example directory to store your datasets. and update the `config.yaml` file to point to your datasets.

## Prerequisites 🔮

- Docker
- Python 3.8+
- CUDA-compatible GPU (recommended)

## Setup 🏗️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/UAV_Classification_repo.git
cd UAV_Classification_repo
```

2. (Optional) Copy the example environment file and configure your variables:
```bash
cp .env.example .env
```

3. Build and using Docker:

```bash
docker compose build app
```

## Environment Variables 📨

Create a `.env` file in the root directory with the following variables (see [.env.example](https://github.com/AndrewPBerg/UAV_Classification/blob/master/.env.example)):
> Note: see setup section for more details

- `WANDB_API_KEY`: Your Weights & Biases API key
- `BOT_TOKEN`: Telegram bot token for notifications (optional)
- `CHAT_ID`: Telegram chat ID for notifications (optional)

## Usage 🐳
1. Configure your experiment in `src/config.yaml` & `orchestrate.yaml`

2. Run training:
```bash
docker compose run app
```
## License ⚖️

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
