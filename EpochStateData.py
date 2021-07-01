
class EpochStateData(object):
    def __init__(self, current_validate_accuracy,current_train_accuracy, epoch, weights):
        self.validate_accuracy = current_validate_accuracy
        self.train_accuracy = current_train_accuracy
        self.epoch = epoch
        self.weights = self.deep_copy_list_of_np_arrays(weights)

    def __str__(self):
        return f"Epoch {self.epoch}\nTrain accuracy: {self.train_accuracy}% and Validate accuracy: {self.validate_accuracy}%"

    @staticmethod
    def deep_copy_list_of_np_arrays(l):
        res = []
        for arr in l:
            if arr is None:
                res.append(None)
            else:
                res.append(arr.copy())

        return res
