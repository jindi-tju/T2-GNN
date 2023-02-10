# T2-GNN
Source code for AAAI2023 "T2-GNN: Graph Neural Networks for Graphs with Incomplete Features and Structure via Teacher-Student Distillation"

## Dependencies
* Python3
* NumPy
* SciPy
* PyTorch
* TensorFlow.keras

## Example Usages
Before running the code, please unzip the *data.zip*.

* `python run.py --dataset texas --Ts 4.0 --topk 10 --lambd 0.8`

Please refer to the *args.py* for more parameters.

## Acknowledgements
The demo code is implemented based on [GCN-with-Hinton-Knowledge-Distillation](https://github.com/berlincho/GCN-with-Hinton-Knowledge-Distillation)

## Reference
If you make advantage of T2-GNN in your research, please cite the following in your manuscript:

Cuiying Huo, et al. "T2-GNN: Graph Neural Networks for Graphs with Incomplete Features and Structure via Teacher-Student Distillation." In AAAI. 2023.
