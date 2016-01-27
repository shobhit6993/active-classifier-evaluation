import classifier
import dataset
import operator

class ActiveFramework(object):
    """Main active learning framework class.
    
    Attributes:
        budget (float64): Budget available.
        clf (list): List of objects for the set of classifiers.
        ds (DatasetSim object): DatasetSim class object.
        min_loss_per_step (float64): Minimum loss per iteration of the infinite loop.
        num_classifiers (int): Number of classifiers.
        num_items (int): Number of items in the dataset.
    """
    def __init__(self, num_items, num_classifiers, budget, min_loss_per_step):
        self.num_items = num_items
        self.num_classifiers = num_classifiers
        self.ds = self.__setup_dataset(num_items)
        self.clf = self.__setup_classifiers(num_classifiers)
        self.budget = budget
        self.min_loss_per_step = min_loss_per_step

    def __setup_dataset(self, num_items):
        return dataset.DatasetSim(num_items, 'uniform')

    def __setup_classifiers(self, num_classifiers):
        """Setup and create a list of classifier objects.
        
        Args:
            num_classifiers (int): Number of classifiers.
        
        Returns:
            list: List of classifiers objects.
        """
        clf = []
        for _ in xrange(0,num_classifiers):
            clf.append(classifier.ExponentialClassifierSim(0.4,
                                                        'dense_high_scorers'))
        return clf

    def print_classifier_predictions(self):
        """Prints prediction of classifiers on the input data.
        """
        for c in self.clf:
            print c.predict(self.ds.target)

    def get_item_with_max_disagreement(self):
        """Returns the first item with max disagreement among classifiers.
        Works only for binary classifier.
        
        Returns:
            int: index of item with max disagreement.
        """
        freq = [0] * self.num_items
        for index, item in enumerate(self.ds.data):
            for _, c in enumerate(self.clf):
                freq[index] = freq[index] + \
                              c.predict([self.ds.target[index]])[0]

        index, value = max(enumerate(freq), key=operator.itemgetter(1))
        return index

    def request_gold_label(self, index):
        """Requests gold label for the specified item.
        Adds the (item, gold label) to dataset's gold_requested attribute.
        
        Args:
            index (int): Index of item whose gold label needs to be obtained.
        
        """
        self.ds.gold_requested[index] = self.ds.target[index]

    def compute_classifier_accuracy(self, c):
        """Calculates classifier's accuracy based on the available labels.
        (gold + crowd). Updates classifier's estimated_accuracy attribute.
        
        Args:
            c (Classifier object): classifier object whose accuracy needs to be computed.
        
        """
        correct = 0
        for i, g in self.ds.gold_requested.iteritems():
            if g != -1:
                l = c.predict([self.ds.target[i]])[0]
                if g == l:
                    correct = correct + 1

        c.estimated_accuracy = float(correct)/len(self.ds.gold_requested)

    def infinite_loop(self):
        """The main infinite loop for active learning framework.
        """
        for c in self.clf:
            self.compute_classifier_accuracy(c)
        

def main():
    active_fr = ActiveFramework(100,10, float('inf'), 3)
    print active_fr.ds.target
    print ""
    active_fr.print_classifier_predictions()
    index = active_fr.get_item_with_max_disagreement()  # TODO(Shobhit): What to do with this item?
    active_fr.request_gold_label(index)
    active_fr.infinite_loop()


if __name__ == "__main__":
   main()