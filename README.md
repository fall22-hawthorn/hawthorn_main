# English proficiency checker

[Demo](https://huggingface.co/spaces/alex42t/EssayChecker) |
[Kaggle link](https://www.kaggle.com/c/feedback-prize-english-language-learning)

## About

To start with this project, take a look at the Kaggle [competition](https://www.kaggle.com/c/feedback-prize-english-language-learning).

## Files
* `main.md`: main notebook 
* `human_baseline.md`: notebook with analysis of scoring by our hands
* `features.py`: feature extractor class
* `demopage.md`: playground for the gradio demo
* `app.py`: gradio demo file for Hugging Face 🤗 Spaces

## Requirements
The project requires a number of additional packages to be installed which are listed in `requirements.txt`.
One of the options to do so it to run
`pip install -r /path/to/requirements.txt`

## Additional info
This project uses Jupytext to do git things with notebooks. Please do not upload/commit .ipynb files without great need. You can learn more about Jupytext in [this](https://www.youtube.com/watch?v=J5yW-NEJp5Q&t=3s) video.
