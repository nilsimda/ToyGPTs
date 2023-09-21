# CalculatorGPT
This Repo trains a GPT on basic arithmetic problems of 
the form `number1 (+|-|*|/) number2 =` with according labels.
It therefore (hopefully) outputs the correct results by predicting 
the next tokens sequentially. It is usable as a simple (but of course 
inefficient) calculator. 

## Usage
Install the ToyGPT Repo cd into CalculatorGPT and run:
```
python3 calculate.py 
```
This allows you to input a two number arithmetic problem like 
`10092 - 456` and the GPT will calculate the result.
### Training your own
Training your own Calculator is also possible with:
```
python3 prepare_data.py 
python3 train.py 
```
You can specify parameters of the model and training through the `config.yml` file. 