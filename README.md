## Usage of this code:
This code assumes you have in the same repository, in folder "acllmdb_v1" the IMDb dataset, with structure:
- acllmdb_v1
  - acllmdb
    - test
      - pos
      - neg
    - train
      - pos
      - neg

and withing "pos" and "neg" folders text files with review texts.

By default, running the code runs all the neural network sizes, saving the intermediate data to speed up loading 
for further reruns. 

The ``sentiment.py`` file contains feature extraction functions, while ``neuralNet.py`` 
holds the actual neural network code.

To run the code, simply run ``python neuralNet.py``