The goal of this project was to train my own 1-layer transformer on the algorithmic task of modular arithmetic, then interpret the learned algorithm of the model.

The mytrain script allows you to train your own model, which will save into a my_runs folder. The script uses CUDA, so make sure to use a compatible GPU or remote machine. If you wish to jump into the mech interp, the lens.py file uploads a checkpoint and each code block has a different function. For further information, I recommend checking out ARENA education where many of this was adapted from. Also the model is an adaptation of Karpathy's NanoGPT.

![image](embedding_evolution.gif)
