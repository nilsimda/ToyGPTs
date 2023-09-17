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
python3 prepare_data.py shakespeare
```
If you want to use an author currently not included, just add their full work to the `data/` folder in a file called `<lastname_author>.txt`

Now we can train the model by running:
```
python3 train.py shakespeare
```
The `config.yml` file should be used to change the parameters of the training and model. These include the 
batch size, learning rate, number of decoder layers, number of training steps and a few more. 
