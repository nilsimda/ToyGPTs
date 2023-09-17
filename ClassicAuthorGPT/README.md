# ClassicAuthorGPT
Some GPTs trained on the full works of classic authors, such as Shakespeare, Dickens and so on.

## Usage
To train your own GPT run:
```
# create training and validation set
python3 data/prepare_data.py data/shakesepeare.txt
# train the model
python3 train.py shakespeare
```
This saves a model in the `trained_models/` folder under the name `shakespeare_gpt.pth`. This model can then be loaded and used to generate text that immitates Shakespeare:
```
python3 generate.py shakespeare
```
