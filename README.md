# Flappy Bird DQN Project

This project implements a Deep Q-Network (DQN) to play the game Flappy Bird. It was developed as part of a CS465 AI class.

## Project Structure

```
dqn/
├── runs/               # Directory for storing training runs (ignored by git)
├── runs2/              # (and so on for runs3 to runs7)
├── .DS_Store           # macOS specific file (ignored by git)
├── agent.py            # Contains the Agent class implementing the DQN algorithm
├── dqn.py              # Main script to train or test the DQN agent
├── experience_replay.py # Implements the experience replay buffer
├── flappybird1.pt      # Example pre-trained model (can be version controlled if desired)
├── hyperparameters.yml # Configuration file for hyperparameters
├── .gitignore          # Specifies intentionally untracked files that Git should ignore
└── README.md           # This file
```

## Files

*   **`agent.py`**: Defines the `Agent` class, which encapsulates the DQN model, target network, optimizer, and learning logic. It handles action selection (epsilon-greedy), storing experiences, and updating the Q-network.
*   **`dqn.py`**: The main executable script. It orchestrates the training process (initializing the agent, environment, and replay buffer, and running training loops) or loads a pre-trained model for gameplay.
*   **`experience_replay.py`**: Contains the `ReplayBuffer` class, used to store and sample past experiences (state, action, reward, next_state, done) for training the DQN, which helps in decorrelating experiences and improving learning stability.
*   **`flappybird1.pt`**: A PyTorch model file. This is likely a saved state of a trained DQN agent.
*   **`hyperparameters.yml`**: A YAML file to store and manage hyperparameters for the DQN agent, such as learning rate, discount factor, epsilon decay, buffer size, batch size, etc. This allows for easy modification and tracking of different experimental setups.

## Setup

To set up this project, you'll need Python and several libraries.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd dqn
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    A `requirements.txt` file is needed. Assuming standard libraries for such a project, it might include:
    ```
    # requirements.txt
    pygame
    torch
    numpy
    PyYAML
    # Add any other specific libraries used
    ```
    Install them using:
    ```bash
    pip install -r requirements.txt
    ```
    (I will create this `requirements.txt` file in a subsequent step).

## Usage

### Training

To train a new agent, you can typically run the main script. Ensure your `hyperparameters.yml` is configured as desired.

```bash
python dqn.py --train
```
*(Modify the command if your script uses different arguments for training)*

### Playing with a Pre-trained Agent

To run the game with a pre-trained agent (e.g., `flappybird1.pt`):

```bash
python dqn.py --play --model flappybird1.pt
```
*(Modify the command if your script uses different arguments for playing or specifying the model)*

## Results

Below is a visual representation of the agent's performance during training (e.g., rewards over episodes).

(Please move `runs7/flappybird7.png` to `assets/flappybird7.png` for this image to display correctly)
![Training Results](assets/flappybird7.png)

## Implementation Details

(This section would ideally be populated with details from your "Final paper.docx". Since I cannot read it directly, please consider summarizing the key aspects of your DQN implementation here, such as:
*   State representation
*   Network architecture
*   Reward function
*   Hyperparameter choices and their rationale
*   Any novel techniques or modifications to the standard DQN algorithm)

## Future Work / Improvements

*   (Example: Implement Double DQN or Dueling DQN for potentially better performance)
*   (Example: More extensive hyperparameter tuning)
*   (Example: Visualization of Q-values or feature maps) 