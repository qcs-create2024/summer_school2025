# Obtaining the code

To obtain a local copy of all CREATE Summer School 2025 codes, run the following at the linux terminal in a folder where you want to store the materials:

```$ git clone https://github.com/qcs-create2024/summer_school2025.git```

# Running the code 

In order to run psNHSW.py, it is a good idea to install all dependencies to a python 3 virtual environment (virtualenv).

To do this, run the following at the terminal prompt:

<pre>$ cd summer_school2025
$ python3 -m venv .venv
</pre>

The above command create a blank virtual environment in a hidden dot-folder named '.venv'. To activate the new virtual environment, run:

```$ source .venv/bin/activate```

Next, load CUDA by running:

```$ module load cuda```

Navigate to the directory ```pseudospecNHSW```. Installing necessary modules can be achieved by now running:

``` (.venv) $ pip install -r requirements.txt```

*(Note that this step can take up to 15 minutes.)*

Finally, the code may be invoked via:

``` (.venv) $ python psNHSW.py```

If successful, this command should start writing output in the form of PNG files for each 'frame' of output.

If you wish to exit/deactivate the virtual environment, run: 

```$ deactivate```
