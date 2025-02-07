import numpy as np
import util
import sys

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***
    tmp = LogisticRegression()
    tmp.fit(x_train, y_train)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])

        for i in range(self.max_iter):
            loss = self.loss_function(x, y)
            print("Before the {}-th fit iteration, the loss is {}".format(i, loss))
            
            if (loss < self.eps):
                print("During {}-th fit, the loss is less than the threshold, the training process is done.".format(i))
                break
            grad = self.grad(x, y)
            hessian = self.hessian(x)

# array([[2.50000000e-01, 8.78956314e-01, 2.31870598e+01],
#        [8.78956314e-01, 3.40183221e+00, 9.12140762e+01],
#        [2.31870598e+01, 9.12140762e+01, 5.38650439e+03]])
#             [np.float64(3.204336902546158), np.float64(-1.1529303446728003), np.float64(0.00915558861476609)]
            self.theta = self.theta - ( np.linalg.inv(hessian) @ grad).ravel()  # update the theta
            
            loss = self.loss_function(x, y)
            print("After the {}-th fit iteration, the loss is {}".format(i, loss))

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return np.apply_along_axis(self.regression, 1, x)
        # *** END CODE HERE ***

    def hessian(self, x):
        """Calculate the Hessian of the loss function.

        Args:
            x: Training example inputs. Shape (m, n).
            theta: Parameters of the model.

        Returns:
            Hessian of the loss function. Shape (n, n).
        """
        # *** START CODE HERE ***
        tmp_result = self.predict(x)
        tmp_result = tmp_result.reshape(-1, 1)

        data_size = x.shape[0]

        return 1.0/data_size *  np.dot(x.T, tmp_result * (1 - tmp_result) * x)
        # *** END CODE HERE ***

    def grad(self, x, y):
        """Calculate the gradient of the loss function.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            Gradient of the loss function. Shape (n,).
        """
        # *** START CODE HERE ***
        tmp_result = self.predict(x).reshape(-1, 1)
        y = y.reshape(-1, 1)
        data_size = x.shape[0]

        return 1.0/data_size * np.transpose(x) @ (tmp_result - y)
        # *** END CODE HERE ***
    def regression(self, x):
        theta = self.theta
        if theta is None:
            theta = np.zeros(x.shape[0])
        h_x = np.dot(x, theta)
        return 1/(1+np.exp(-h_x))
    
    def loss_function(self, x, y):
        tmp_result = self.predict(x)
        print ("The shape of tmp_result is {}".format(tmp_result.shape))
        
        tmp_result = tmp_result.reshape(-1, 1)
        y = y.reshape(-1, 1)

        print("The shape of tmp_result and y is {} and {}".format(np.log(tmp_result).shape, y.shape))
        data_size = x.shape[0]

        result = -1.0/data_size * np.sum(np.transpose(y) @ np.log(tmp_result) + np.transpose(1 - y) @ np.log(1 - tmp_result))
        print(np.transpose(y) @ np.log(tmp_result))
        return result

if __name__ == '__main__':
    # get the parameters for main function from arguments
    train_path = sys.argv[1]
    eval_path = sys.argv[2]
    pred_path = sys.argv[3]
    main(train_path, eval_path, pred_path)