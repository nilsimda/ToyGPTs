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
### Training your own
Training your own Calculator is also possible with:
```
python3 train.py
```