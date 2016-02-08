import classifier
import dataset
import operator
import numpy
import matplotlib.pyplot as plt

class ActiveFramework(object):
    """Main active learning framework class.
    
    Attributes:
        budget (float): Budget available.
        clf (list): List of objects for the set of classifiers.
        curr_loss (float): Current loss
        ds (DatasetSim object): DatasetSim class object.
        min_loss_per_step (float): Min loss per iteration of the infinite loop.
        num_classifiers (int): Number of classifiers.
        num_items (int): Number of items in the dataset.
    """
    def __init__(self, num_items, num_classifiers, budget, num_seeds,
                 min_loss_per_step, gold_label_cost):
        self.num_items = num_items
        self.num_classifiers = num_classifiers
        self.ds = self.__setup_dataset(num_items)
        self.clf = self.__setup_classifiers(num_classifiers)
        self.min_loss_per_step = min_loss_per_step
        self.curr_loss = 0.0
        self.num_seeds = num_seeds
        self.budget = budget
        self.gold_label_cost = gold_label_cost

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

    def calculate_classfier_predictions(self):
        for c in self.clf:
            c.predict(self.ds.target)

    def print_classifier_predictions(self):
        """Prints predictions of classifiers on the input data.
        """
        for c in self.clf:
            print c.predicted_label

    def out_of_budget(self):
        #TODO(Shobhit): Add crowd cost
        return self.budget <= min(0, self.gold_label_cost-1)

    def get_item_with_max_disagreement(self):
        """Returns the first item with max disagreement among classifiers.
        disagreement is calculated by minimizing the sum (over all labels)
        of abs(freq - mean) value for each item, where freq = number of 
        classifiers which predicted a label l for item i, and 
        mean = num_classifiers/num_labels
        
        Returns:
            int: index of item with max disagreement.
        """
        freq = [[0 for i in xrange(0, 2)] for j in xrange(0, self.num_items)]

        for i in xrange(0, self.num_items):
            for c in self.clf:
                l = c.predicted_label[i]
                freq[i][l] = freq[i][l] + 1

        mean = self.num_classifiers/2.0
        min_error = float('inf')
        for i in xrange(0, self.num_items):
            error = 0.0
            for l in xrange(0,2):
                error = error + abs(freq[i][l] - mean)
            if error < min_error:
                min_error = error
                index = i

        return index

    def seed_gold_labels(self):
        for count in xrange(0, self.num_seeds):
            r = numpy.random.randint(0, len(self.ds.data))
            if r not in self.ds.gold_requested:
                self.ds.gold_requested[r] = self.ds.target[r]


    def request_gold_label(self, index):
        """Requests gold label for the specified item.
        Adds the (item, gold label) to dataset's gold_requested attribute.
        Updates budget.
        
        Args:
            index (int): Index of item whose gold label needs to be obtained.

        Returns:
            bool: True if sufficient budget available else False.
        """
        if self.budget >= self.gold_label_cost:
            self.ds.gold_requested[index] = self.ds.target[index]
            self.budget = self.budget - self.gold_label_cost
            return True
        else:
            return False
            

    def calculate_classifier_accuracy(self, c):
        """Calculates classifier's accuracy based on the available labels.
        (gold + crowd). Updates classifier's estimated_accuracy attribute.
        
        Args:
            c (Classifier object): classifier object whose accuracy 
                needs to be calculated.
        
        """
        correct = 0
        for i, g in self.ds.gold_requested.iteritems():
            if g != -1:
                if g == c.predicted_label[i]:
                    correct = correct + 1

        c.num_correct_predictions = correct
        c.estimated_accuracy = float(correct)/len(self.ds.gold_requested)
        # print c.estimated_accuracy

    def calculate_loss(self, accuracy):
        loss = 0.0
        for i in xrange(0, self.num_classifiers):
            for j in xrange(i + 1, self.num_classifiers):
                    loss = loss + abs(accuracy[i] - accuracy[j])
        return (-loss)

    def calculate_new_accuracy(self, c, i, g):
        n = len(self.ds.gold_requested)
        if g == c.predicted_label[i]:
            return float(c.num_correct_predictions + 1)/(n + 1)
        else:
            return float(c.num_correct_predictions)/(n + 1)

    def calculate_new_loss(self, i, g):
        new_accuracy = []
        for c in self.clf:
            new_accuracy.append(self.calculate_new_accuracy(c, i, g))
            # print new_accuracy[-1]

        return self.calculate_loss(new_accuracy)
        # return new_loss
    
    def plot_graph(self, x_axis, y_axis):
        plt.plot(x_axis, y_axis, 'ro-')
        plt.axis([0, self.num_items, 0, -30])
        plt.title("items = " + str(self.num_items) 
                + " with seeds = " + str(self.num_seeds))
        plt.show()

    def infinite_loop(self):
        """The main infinite loop for active learning framework.
        """
        x_axis = []
        y_axis = []
        for count in xrange(1,self.num_items - self.num_seeds):
            for c in self.clf:
                self.calculate_classifier_accuracy(c)

            self.curr_loss = self.calculate_loss(
                                    [c.estimated_accuracy for c in self.clf])

            print "count = %d" % count
            # print "curr_loss = %d" % self.curr_loss
            x_axis.append(count)
            y_axis.append(self.curr_loss)
            if self.out_of_budget():
                print "Out of budget! Terminating..."
                return

            loss_array = []
            for i, item in enumerate(self.ds.data):
                if i in self.ds.gold_requested:
                    continue
                
                expected_loss = 0.0
                for g in [0,1]:
                    loss = self.calculate_new_loss(i, g)
                    expected_loss = expected_loss + 0.5 * loss;
                    
                # print expected_loss
                loss_array.append((i, self.curr_loss - expected_loss))
            
            # print "---"
            index, _ = max(loss_array, key = operator.itemgetter(1))

            if self.request_gold_label(index) == False:
                print "Out of budget! Terminating..."
                return

        self.plot_graph(x_axis, y_axis)

def main():
    active_fr = ActiveFramework(10, 10, float('inf'), 0, 3, 0)
    # print active_fr.ds.target
    # print ""
    active_fr.calculate_classfier_predictions()
    # active_fr.print_classifier_predictions()
    active_fr.seed_gold_labels()
    index = active_fr.get_item_with_max_disagreement()
    
    if active_fr.request_gold_label(index) == False:
        print "Out of budget! Terminating..."
        return

    active_fr.infinite_loop()


if __name__ == "__main__":
   main()