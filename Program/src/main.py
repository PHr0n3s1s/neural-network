# ------------------------------------------------------------------------- #
# Created By        :   PHr0n3s1s.                                          #
# Creation Date     :   6/10/2023 (MM-DD-YYYY)                              #
# CPython version   :   '3.11.4'                                            #
# Licence           :   MIT                                                 #
# ------------------------------------------------------------------------- #
"""
    -   Optical Recognition of Handwritten Digits.
"""
# ------------------------------------------------------------------------- #

import machine_learning as ml
from sys import exit


def get_numeric_input(msg: str, min=float('-inf'), max=float('inf')) -> float:
    while True:
        try:
            _input = float( input(msg) )
            if (_input < min) or (_input > max):
                print(f'\t\x1b[31mInvalid Input. Not in range: {min} - {max}\x1b[0m', file=stderr)
                continue
            break
        except ValueError:
            # ANSI Color(red) Escape Codes: https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
            print('\t\x1b[31mInvalid input. Please enter a number.\x1b[0m', file=stderr)
        except KeyboardInterrupt:
            print('\n\t\x1b[31mPlease enter the appropriate option to quit the program.\x1b[0m', file=stderr)

    return _input


def main():
    ann = None  # Artifical Neural Network...

    # X : A batch of np.arrays containing the individual pixels on a 8*8 image,
    # y : An np.array containing the correct digit/label an image has:
    X_train, y_train = ml.load_ML_data('../Assets/training_data.dat')
    X_test, y_test = ml.load_ML_data('../Assets/testing_data.dat')

    size1 = 32; size2 = 12   # Default hidden-layer sizes...
    train_step = .5; adaptive=False

    print (
        '''\
        \r=========================================================================
        \r USER MENU:
        \r \033[96m\033[1mCLASSIFICATION OF THE OPTICAL RECOGNITION of HANDWRITTEN DIGITS DATASET\033[0m
        \r=========================================================================\
        '''
    )

    input_choices = \
    '''
    \rPlease choose an option:
    -> 1. Display the first 5 training images of the dataset (in ASCII art).
    -> 2. Set the size of the hidden layers (64-?-?-10).
    -> 3. Set the learning rate (0.001 - 1).
    -> 4. Train the ANN on labeled data, and display progress graph.
    -> 5. Classify the unlabeled data, output training report & confusion matrix.
    -> 6. Perform comparative experimentation with varying ANN parameters.
    -> 7. Exit.
    \rNote: !If no option is modified, the default values will be used!
    \rEnter your choice: '''

    while True:
        match int( get_numeric_input(input_choices, 1, 7) ):
            case 1:
                ml.display_images(X_train, y_train)

            case 2:
                size1 = int( get_numeric_input('\nEnter the size of the first hidden layer: ', min=1) )
                size2 = int( get_numeric_input('\nEnter the size of the second hidden layer: ', min=1) )

            case 3:
                train_step, adaptive = ml.get_learning_step()

            case 4:
                train_epochs = int(get_numeric_input (
                    msg = '\nEnter the training epochs: ',
                    min = 50
                ))

                ann = ml.NeuralNetwork(hidden1_size=size1, hidden2_size=size2)  # Initialize the Neural Network...
                errors, accuracy, weights = ann.train (
                    X=X_train,
                    y=y_train,
                    iterations=train_epochs,
                    learning_step=train_step,
                    adaptive_step=adaptive
                )

                ml.generate_graphs (
                    errors,
                    accuracy,
                    weights,
                    (len(ann.W1[0]), len(ann.W2[0])),
                    train_step,
                    adaptive
                )

            case 5:
                if ann is None:
                    # If there's a pre trained network file, it's going to use it,
                    # otherwise it assigns random weights & biases:
                    ann = ml.NeuralNetwork(hidden1_size=size1, hidden2_size=size2)
                ml.predictions_report (
                    predictions=ann.predict(X_test),
                    labels=y_test,
                    topology=(len(ann.W1[0]), len(ann.W2[0])),
                    learning_step=train_step,
                    adaptive=adaptive
                )

            case 6:
                ml.perform_experiments(X_train, y_train)

            case 7:
                break

    exit()


if __name__ == '__main__':
    main()

