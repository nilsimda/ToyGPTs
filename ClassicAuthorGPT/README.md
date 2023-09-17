# ClassicAuthorGPT
Some GPTs trained on the full works of classic authors. Currently included are:
* William Shakespeare
* Charles Dickens
* Jane Austen

## Usage
### Generating Text
To generate some text from the pretrained models run:
```
python3 generate.py shakespeare
```
### Training your own ClassicAuthorGPT
To train you need to first create the training and validiation set from the full works of the author like so:
```
python3 data/prepare_data.py data/shakesepeare.txt
```
If you want to use an author currently not included, just add their full work in a file called `<lastname_author>.py`

Now we can train the model by running:
```
python3 train.py shakespeare
```
