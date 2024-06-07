Steps to Reproduce VTAB Benchmark:

1. Install dependencies using requirements.txt. Python version is 3.11.5.
2. Create a folder called "vtab_weights". Store the model checkpoints in this folder. The folder should have the following structure at this point.
    "vtab_weights/in1000.pth.tar",
    "vtab_weights/la1000.pth.tar",
    "vtab_weights/oi1000.pth.tar",
    "vtab_weights/sd1000-i2i.tar",
    "vtab_weights/sd1000-t2i.tar",
    "vtab_weights/laionnet.pth.tar"
3. Run "testAllModelsDatasets.py". Results will be found in the "results" folder.
