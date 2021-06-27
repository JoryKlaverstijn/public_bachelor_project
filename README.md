Designing a Hopfield Neural Network Using Memristors
====================================================

Abstract
--------

In this paper it is explained how Nb-doped SrTiO3 memristors could be
integrated into classical Hopfield neural networks (HNN), and how a
working simulated model could be designed. The performance of this model
was tested and compared to the performance of a linearised version,
non-memristor version and a modern continuous HNN. The performances of
the models are measured on the MNIST data-set by classifying the
converged patterns with a secondary feed forward network. From the
experiment, it can be concluded that a memristor HNN model can have an
adequate accuracy. The accuracy of the (linearised) memristor HNN ranged
from 63.6% to 89.4%, dependent on the amount of noise added to the
images. It is shown that the linearised version of the memristor based
HNN performs slightly better and that the memristor based HNNs perform
worse than the continuous HNN model for higher magnitudes of noise.

Folders
-------

-   `mnist`: Various functions used for (pre-)processing the MNIST
    data-set, and returning random processed MNIST images. Also contains
    code for creating a FFN classifier for processed MNIST images.
-   `memristor_continuous`: Some experimental programs to test out the
    continuous HNN from "Hopfield Networks is all you need".
-   `memristor_nengo`: Functions used for visualising the Nengo HNN.
    Also contains the learning rules for the Nengo HNN.
-   `main_experiment`: All functions used for the main experiment:
    running, testing and evaluating the network.

Running the code
----------------

1.  Clone the entire repository for the library code
2.  Call the create_mnist_array("filename") function inside of `mnist.py` to create an ordered array of processed MNIST-images
3.  Rename the resulting .npy file "mnist_mOja_array.npy" and place it inside `main_experiment` folder
4.  Set desired parameters (noise, nr. of digits, which networks to run
    etc.) in `main.py`
5.  Run `main.py`

*
----------------
Some parts of the code are modified segments from the following repository:
https://github.com/Tioz90/Learning-to-approximate-functions-using-niobium-doped-strontium-titanate-memristors/tree/single_op
