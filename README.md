# VRPO-Robust Reinforcement Learning from Human Feedback for Large Language Models Fine-Tuning



## Introduction


This repository is designed for training and evaluating the Variance-Reduced Preference Optimization (VRPO). It includes the following:

* Trainers code for both One-stage optimization and Two-stage optimization.

* Exaples for training this two models



## Quick Start
This code is heavily based on [trl](https://github.com/huggingface/trl), so you can easily follow the instructions provided by [trl](https://github.com/huggingface/trl) to run it.

### Installation

```bash
git clone https://github.com/huggingface/trl.git
cd trl/
pip install -e .[dev]
```
Then you can Clone this repository, and copy the contents of the trainer/ directory into the trl/train/ directory of the TRL repository:
```bash
git clone https://github.com/VRPO/VRPO.git
cp -r VRPO/trainer/* trl/train/
```

## Example of Data Format Accepted by VRPO
Our trainer recommed Explicit prompt preference dataset with preference data
```python
# Standard format
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green.","a_1": "blue when it’s sunny.", "a_2": "blue at daytime, orange in late afternoon and black at night.","chosen_preference": 0.99, "rejected_preference": 0.01,"a_1_preference": 0.3, "a_2_preference": 0.7}

# Conversational format
preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                      "chosen": [{"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "assistant", "content": "It is green."}]
                      "a_1": [{"role": "assistant", "content": "It is blue when it’s sunny."}],
                      "a_2": [{"role": "assistant", "content": "It is blue at daytime,orange in late afternoon and black at night."}],
                      "chosen_preference": 0.99,
                      "rejected_preference": 0.01,
                      "a_1_preference": 0.3,
                      "a_2_preference": 0.7}

```
