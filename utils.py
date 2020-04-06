import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    true_positive = sum([x == 1 and y == 1 for x,y in zip(real_labels, predicted_labels)])
    false_positive = sum([x == 0 and y == 1 for x,y in zip(real_labels, predicted_labels)])
    false_negative = sum([x == 1 and y == 0 for x,y in zip(real_labels, predicted_labels)])
    true_negative = sum([x==0 and y==0 for x,y in zip(real_labels, predicted_labels)])
    if true_positive + false_positive == 0:
        return 0
    else:
        precision = true_positive / (true_positive + false_positive)
    if false_negative + true_positive == 0:
        return 0
    else:
        recall = true_positive / (false_negative + true_positive)
    if precision + recall == 0:
        return 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return f1

class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        abs_diff = [abs(x-y)**3 for x,y in zip(point1,point2)]
        distance = sum(abs_diff)**(1/3)
        return distance
       

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        diff_square = [(x-y)**2 for x,y in zip(point1,point2)]
        distance = sum(diff_square)**(0.5)
        return distance
       

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        inner_product = [(x * y) for x, y in zip(point1, point2)]
        distance = sum(inner_product)
        return distance
     

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        inner_product = [x*y for x,y in zip(point1,point2)]
        square1 = [x**2 for x in point1]
        square2 = [y**2 for y in point2]
        distance = 1-sum(inner_product)/(sum(square1)**(0.5)*sum(square2)**(0.5))
        return distance
    

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        inner_square = [(x-y)**2 for x,y in zip(point1,point2)]
        distance = -np.exp(-0.5*sum(inner_square))
        return distance
      


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        best_f1, best_k = 0, 1
        for functions_name, functions in distance_funcs.items():

            for i in range(1,30,2):
                if i > len(x_train):
                    break
                model = KNN(i, functions)
                model.train(x_train, y_train)
                valid_f1 = f1_score(y_val, model.predict(x_val))
                if valid_f1 == best_f1:
                    continue
                if valid_f1 > best_f1:
                   best_f1 = valid_f1
                   best_k = i
                   best_distance_function = functions_name
                   best_model = model
                   
        # You need to assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function = best_distance_function
        self.best_model = best_model
        

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        best_f1, best_k = 0, 1
        for scaling_name, scaling_class in scaling_classes.items():
            for functions_name, functions in distance_funcs.items():
                scaler = scaling_class()
                train_features_scaled = scaler(x_train)
                valid_features_scaled = scaler(x_val)


                for i in range(1,30,2):
                    if i > len(train_features_scaled):
                        break
                    model = KNN(i, functions)
                    model.train(train_features_scaled, y_train)
                    valid_f1 = f1_score(y_val, model.predict(valid_features_scaled))
                    if valid_f1 == best_f1:
                        continue
                    if valid_f1 > best_f1:
                       best_f1 = valid_f1
                       best_k = i
                       best_distance_function = functions_name
                       best_scaler = scaling_name
                       best_model = model
        
        # You need to assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function = best_distance_function
        self.best_scaler = best_scaler
        self.best_model = best_model
        


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normal = []
        for i in features:
            if all(elements == 0 for elements in i):
                normal.append(i)
            else:
                SD = sum([x*y for x,y in zip(i,i)])**(0.5)
                normal.append([x/SD for x in i])
        return normal
               


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.cmin = None
        self.cmax = None

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.array(features)
        if self.cmin is None or self.cmax is None:
            self.cmin = np.amin(features, axis=0)
            self.cmax = np.amax(features, axis=0)
            
        
        normalized = (features - self.cmin) / (self.cmax - self.cmin) 
        normalized = normalized.tolist()
        return normalized        