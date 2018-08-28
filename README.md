# musical-neural-net
Train an <a href="https://arxiv.org/pdf/1708.02182.pdf">AWD-LSTM</a> to generate piano or violin/piano music
Project structure overview is <a href="http://christinemcleavey.com/music-generator-project-structure/">here</a>.<br>
Sample generations are <a href="http://christinemcleavey.com/human-or-ai/">here</a>.<br>

<h2>Requirements:</h2>
<ul>
<li>PyTorch version 0.3.0.post4</li>
<li><a href="https://github.com/fastai/fastai">FastAI</a>: install link in musical-neural-net home directory</li>
<li><a href="http://schristiancollins.com/generaluser.php">GeneralUser</a>: install in data folder (this is needed to translate midi files to mp3)</li>
</ul>

<h2>Data:</h2>
Some sample midi files are included in data/composers/midi. The complete midi dataset is available here: **TO DO**. <br>
Run:

```
python midi-to-encoding.py
```

to translate midi files to text files in the various notewise and chordwise options. The midi translation takes a long time,
and preprocessed data files are available here: **TO DO**

<h2>Training and Generation:</h2>
<ul>
<li>make_test_train.py - create the training and testing datasets (adjust notewise/chordwise, optionally create only a small sample size)</li>
<li>train.py - train an AWD-LSTM (adjust model parameters, dropout, and training regime)</li>
<li>generate.py - generate new samples (adjust generation size)</li>
</ul>
Each script has default settings which should be reasonable, but use --help to see the different options and parameters which can be modified.
<h2>Music Critic:</h2>
<h2>Composer Classifier:</h2>
