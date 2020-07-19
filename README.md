## project for SUR: machine learning and classification

Project consists of two face classifier and voice classifier. Both implemented in Python using TensorFlow and Keras.

### Training

Training data are 80x80 RGB pictures of target and nontarget faces in their separate folders for the face classifier.

Voice classifiers uses .wav files of target and nontarget speaker in their separate folders sampled at 16kHz.


Train each classifier like:

python train_face.py/train_voice.py TARGET_FOLDER NON_TARGET_FOLDER [KERAS_MODEL]

### Evaluation

Run evaluation like:

python eval_face.py/eval_voice.py KERAS_MODEL INPUT_FOLDER OUTPUT_FILE

OUTPUT_FILE looks like: file_name score decision

### Requirements

* NumPy
* TensorFlow
* Keras
* SciPy
* soundfile
* opencv-python

