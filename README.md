# Source Code and Dataset for Text Infilling

This repository contains the source code and dataset for [Text Infilling](). 
The implementation is based on [Texar](https://github.com/asyml/texar).




## Repository Structure

This repository contains two branches:

- `master` branch
  - This branch contain the code for conducting the experiments of  *Varying Mask Rates and #Blanks*. 
- `ShowCases` branch
  - This branch is used for *Preposition Infilling* and *Long Content Infilling*. 



## Install

Run the following lines to download the code and install Texar:
```bash
git clone https://github.com/VegB/Text_Infilling
cd Text_Infilling
pip install [--user] -e .    
```



## Experiment Instructions

After installation, you may follow the instructions to conduct experiments for Text Infilling:

- [Instructions for Varying Mask Rates and #Blanks](https://github.com/VegB/Text_Infilling/tree/master/text_infilling)
- [Instructions for Infilling Showcases](https://github.com/VegB/Text_Infilling/tree/ShowCases/text_infilling)



## Requirements

- Python 3
- Tensorflow >= 1.7.0
- Texar
