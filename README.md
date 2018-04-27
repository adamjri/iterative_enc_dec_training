# iterative_enc_dec_training

## Dependencies
* Python 2.7
* Tensorflow
* Keras
* Matplotlib

## Description
Code for testing an iterative encoder/decoder scheme for the training of binary classification neural networks. The procedure can be seen in "iter_enc_dec_scheme.pdf"

* /classifier_model: Used for the encoder+classification portion of the training scheme
  * /models
    * custom_ops: dual sigmoid custom activation for performing ordinal tri-classification
	* initializers: Custom initializers for neural network model parameters using weight initialization procedures from: https://arxiv.org/abs/1704.08863
	* model_base: Abstract Keras model class with training procedure
	* simple_dense: Keras model implementation of basic multi-layer dense network
	* visualizers: Visualization function for plotting data during training
  * datasets: loads and parses datasets into training and testing samples
  * model_params: parameters used for building neural network model at runtime
  * run_training: runs a training session
* /decoder_model: Used for the decoder portion of the training scheme
  * /models: Same structure as classifier model
  * run_inference: loads a model and runs inference on a set of inputs
* /generative_model: Used for generating classification data using generative neural network models
  * /data
    * hyperplane_labeling: Apply binary labels to a dataset using a hyperplane in the input space of the generator
	* sampler_base: Abstract base class for a dataset generator
	* simple_1d_sampler: Generates samples in 1 dimension
	* simple_Nd_sampler: Generates samples in an N dimensional hyper-cube
  * /models: Same structure as classifier model
  * /plotting: Generates plots for visualizing a dataset
  * generate_dataset_from_generators: Used to generate a dataset based on classifying between different generators
  * generate_dataset_from_model: Generates a dataset given an input dataset and a generative model
  * generate_input_dataset: Generates a simple input dataset to be used for building datasets using generators
* /separator_correction: Used for applying corrections in between encoder and decoder portions of the scheme
* full_training_loop: Runs the whole procedure iteratively.
