# English proficiency checker
[Slides](https://docs.google.com/presentation/d/1ZvPy25h1qGewxezwvOQ6vg1WUvwfPdKmsFG0RMoEQzQ) |
[Demo](https://huggingface.co/spaces/alex42t/EssayChecker) |
[Kaggle link](https://www.kaggle.com/c/feedback-prize-english-language-learning)

## About

In this project we aimed to create machine learning models to grade English essays written by 8th-12th grade English Language Learners (ELLs).
The resulting model can be used to alleviate the burden of grading on teachers, who are often underpaid and overworked, while giving nearly instant feedback to ELLs.

To start with this project, you take a look at the corresponding Kaggle [competition](https://www.kaggle.com/c/feedback-prize-english-language-learning).


## Files
* `main.md`: main notebook with models and EDA
* `human_baseline.md`: notebook with analysis of scoring by our hands
* `features.py`: feature extractor class
* `demopage.md`: playground for the gradio demo
* `app.py`: gradio demo file for Hugging Face 🤗 Spaces

## Requirements
The project requires a number of additional packages to be installed which are listed in `requirements.txt`.
One of the options to do so is to run
`pip install -r /path/to/requirements.txt`

## Additional info
This project uses Jupytext to do git things with notebooks. Please do not upload/commit .ipynb files without great need. You can learn more about Jupytext in [this](https://www.youtube.com/watch?v=J5yW-NEJp5Q&t=3s) video.
