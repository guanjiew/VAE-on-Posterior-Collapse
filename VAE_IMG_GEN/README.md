### Betty's Notes 

To run and generate training curves/visualizations: start visdom server by running

```
python3 -m visdom.server
```

If you don't want to generate training curves/visualization, set the --viz_on argument to False

I'm currently parsing the commandline argument directly from ControlVAE_Image_generation.py, will comment it out later

Params to consider:

is_PID: if True, we're running the CVAE solver, if not, we're running naive VAE
beta: I set the default to 1, to make it a naive vae
train: True to train, false to test
max_iter: 1500 for now, we probably don't need this many, it looks like we converge at around 500

After changing the params you want to change, just run 

```
python3 ControlVAE_Image_Generation.py
```