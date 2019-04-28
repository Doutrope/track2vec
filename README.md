# track2vec : track recommendation based on word2vec
track2vec uses word2vec algorithm to compute recommendations for tracks. You can visit [there](https://www.overleaf.com/read/zwcjdrxpybjd) if you want a concise explanation of the algorithm and the recommendation problematic.
# Project organisation
* ```data```: contains data
* ```models```: stores trained models and outputs
* ```track2vec.py```: main algorithm
* ```utils.py```: data management and metric computation functions

track2vec works with ```python3.6``` and needs the following packages to run :
```
gensim, json, numpy, pandas, sklearn
```
# Get datasets
First make sure you have access ```gsutil``` installed and an access to the bucket ```gs://titan-source-dinar-raw``` then run the following command from project dir:
```bash
gsutil -m cp -r gs://titan-source-dinar-raw/ ./data
```
It will download 3 folders: ```playlists```, ```users``` and ```albums``` from which the algorithm will compute track similarities. Playlists contains deezer playlists, users contains info about the users. albums is not used.
# Filter data
In order to filter users data and free some space, start by running : 
```bash
python -c 'import utils; filter_usersdata_parallel'
``` 
from the project directory. It will filter users data and only keep userid as key and the user's country as value.
# Run training and save best model
```bash
python track2vec.py -c FR -p 0.1 -t 5000
``` 
Means train the algorithm with FR playlists, with a test proportion of 0.1 and 5000 playlists (train+test) for training.
Then the algorithm will keep the best model in order to re-train on the whole dataset.
The outputs are stored under ```models/flow``` and the scores are kept in the files ```models/result_date```
