import classifier
import dataset
import hierarchical
import operator
import numpy
import matplotlib.pyplot as plt
import pickle
import random


class ActiveFramework(object):
    """Main active learning framework class.

    Attributes:
        budget (float): Budget available.
        clf (list): List of objects for the set of classifiers.
        clf_assignments (list): The equivalence class to which each classifier
            is assigned.
        curr_loss (float): Current loss
        ds (DatasetSim object): DatasetSim class object.
        gold_label_cost (float): Cost to obtain a gold label
        min_loss_per_step (float): Min loss per iteration of the infinite loop.
        num_classifiers (int): Number of classifiers.
        num_egs (int): Number of examples in the dataset.
        num_equiv (int): Number of equivalence classes
        num_seeds (int): Number of seed examples for which gold label is
            already present.
    """

    def __init__(self, num_classifiers, num_equiv, num_egs, num_seeds,
                 min_loss_per_step, budget, gold_label_cost, num_mcmc_steps):
        self.num_classifiers = num_classifiers
        self.clf = self.__setup_classifiers(num_classifiers)
        self.num_equiv = num_equiv
        self.clf_assignments = [0] * num_classifiers
        self.accuracy_set = [c.accuracy for c in self.clf]

        self.num_egs = num_egs
        self.ds = self.__setup_dataset(num_egs)
        self.num_seeds = num_seeds

        self.min_loss_per_step = min_loss_per_step
        self.curr_loss = 0.0

        self.budget = budget
        self.gold_label_cost = gold_label_cost

        self.num_mcmc_steps = num_mcmc_steps

        self.eps = 0.001    # epsilon added to classifier with acc=0

    def __setup_dataset(self, num_egs):
        return dataset.DatasetSim(num_egs, 'uniform')

    def __setup_classifiers(self, num_classifiers):
        """Setup and create a list of classifier objects.

        Args:
            num_classifiers (int): Number of classifiers.

        Returns:
            list: List of classifiers objects.
        """
        clf = []
        for _ in xrange(0, num_classifiers):
            clf.append(
                classifier.ExponentialClassifierSim(0.4, 'dense_high_scorers'))
        return clf

    def calc_classifier_accuracies(self):
        for c in self.clf:
            c.predict(self.ds.target)

    def out_of_budget(self):
        return self.budget <= min(0, self.gold_label_cost - 1)

    # def get_eg_with_max_disagreement(self):
    #     """Returns the first example with max disagreement among classifiers.
    #     Disagreement is calculated by minimizing the sum (over all labels)
    #     of abs(freq - mean) value for each example, where freq[i][l] is the
    #     number of classifiers which predicted a label l for eg i, and
    #     mean = num_classifiers/num_labels

    #     Returns:
    #         int: index of eg with max disagreement.
    #     """
    #     freq = [[0 for i in xrange(0, 2)] for j in xrange(0, self.num_egs)]

    #     for i in xrange(0, self.num_egs):
    #         for c in self.clf:
    #             l = c.predicted_label[i]
    #             freq[i][l] = freq[i][l] + 1

    # Mean is calculated by dividing by 2 because there are two possible
    # labels in binary classification.
    #     mean = self.num_classifiers/2.0
    #     min_error = float('inf')
    #     for i in xrange(0, self.num_egs):
    #         error = 0.0
    #         for l in xrange(0,2):
    #             error = error + abs(freq[i][l] - mean)
    #         if error < min_error:
    #             min_error = error
    #             index = i

    #     return index

    def top_disagree_egs(self, num):
        freq = [[0 for i in xrange(0, 2)] for j in xrange(0, self.num_egs)]

        for i in xrange(0, self.num_egs):
            if i in self.ds.gold_requested:
                continue
            for c in self.clf:
                l = c.predicted_label[i]
                freq[i][l] = freq[i][l] + 1

        # Mean is calculated by dividing by 2 because there are two possible
        # labels in binary classification.
        mean = self.num_classifiers / 2.0
        error_tup = []
        for i in xrange(0, self.num_egs):
            if i in self.ds.gold_requested:
                continue

            error = 0.0
            for l in xrange(0, 2):
                error = error + abs(freq[i][l] - mean)
            error_tup.append(tuple((error, i)))

        error_tup.sort()
        return [i for err, i in error_tup][0:num]

    def seed_gold_labels(self):
        """Randomly selects num_seeds number of examples from the dataset
        and treats them as seed set by obtaining their gold labels.
        The function adds the seed examples along with their gold labels to
        the gold_requested attribute of the Dataset object. Note that seed
        gold labels do not incur any cost.

        Returns:
            TYPE: Description
        """
        arr = random.sample(range(0, len(self.ds.data)), self.num_seeds)
        for r in arr:
            self.ds.gold_requested[r] = self.ds.target[r]

    def request_gold_label(self, index):
        """Requests gold label for the specified example.
        Adds the (eg, gold label) to dataset's gold_requested attribute.
        Updates budget.

        Args:
            index (int): Index of eg whose gold label needs to be obtained.

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
        """Calculates classifier's accuracy based on the available labels
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
        c.estimated_accuracy = float(correct) / len(self.ds.gold_requested)

        # if accuracy = 0, add epsilon to it,
        # if accuracy = 0, subtract epsilon from it
        # because our model does not have 0 or 1 in its support
        if c.estimated_accuracy == 0.0:
            c.estimated_accuracy = self.eps
        elif c.estimated_accuracy == 1.0:
            c.estimated_accuracy = c.estimated_accuracy - self.eps

    # def calculate_loss(self, accuracy):
    #     loss = 0.0
    #     for i in xrange(0, self.num_classifiers):
    #         for j in xrange(i + 1, self.num_classifiers):
    #                 loss = loss + abs(accuracy[i] - accuracy[j])
    #     return (-loss)

    def calculate_loss(self, h):
        """Calculates the current loss which is equal to the sum of area under
        the curve for each equivalence classes's posterior predictive
        distribution.

        Args:
            h (class object): Object for Hierarchical class

        Returns:
            float: Current loss as the sum of areas under curves
        """
        loss = 0.0
        bin_width = 1.0 / h.num_bins
        for i in xrange(0, self.num_equiv):
            for j in xrange(i + 1, self.num_equiv):
                # Histogram array tuples
                # tuple(array[of freq counts], array[of bin boundaries])
                h_i = h.post_pred[i][0]
                h_j = h.post_pred[j][0]
                loss = loss + numpy.dot(h_i, h_j)

        return loss * bin_width

    def new_accuracy(self, c, i, g):
        n = len(self.ds.gold_requested)
        if g == c.predicted_label[i]:
            a = float(c.num_correct_predictions + 1) / (n + 1)
        else:
            a = float(c.num_correct_predictions) / (n + 1)

        # if accuracy = 0, add epsilon to it,
        # if accuracy = 0, subtract epsilon from it
        # because our model does not have 0 or 1 in its support
        if a == 0.0:
            a = self.eps
        elif a == 1.0:
            a = a - self.eps

        return a

    def estimate_new_accuracy(self, i, g):
        new_accuracy = []
        for c in self.clf:
            new_accuracy.append(self.new_accuracy(c, i, g))
        return new_accuracy

    # def calculate_new_loss(self, i, g):
    #     new_accuracy = []
    #     for c in self.clf:
    #         new_accuracy.append(self.calculate_new_accuracy(c, i, g))
    # print new_accuracy[-1]

    #     return self.calculate_loss(new_accuracy)
        # return new_loss

    def plot_graph(self, x_axis, y_axis):
        plt.plot(x_axis, y_axis, 'ro-')
        plt.axis([0, self.num_egs, 0, -30])
        plt.title("Number of egs = " + str(self.num_egs) +
                  " with seeds = " + str(self.num_seeds))
        plt.show()

    def acc_by_class(self):
        acc_by_equiv_class = [[] for _ in xrange(0, self.num_equiv)]

        # classifier i assigned in equivalence class j
        for i, j in enumerate(self.clf_assignments):
            acc_by_equiv_class[j].append(self.clf[i].accuracy)

        return acc_by_equiv_class

    # def acc_by_class_anticipatory(self, clf_assignments, accuracy):
    #     acc_by_equiv_class = [[] for _ in xrange(0, self.num_equiv)]

    # classifier i assigned in equivalence class j
    #     for i, j in enumerate(clf_assignments):
    #         acc_by_equiv_class[j].append(accuracy[i])

    #     return acc_by_equiv_class

    def initial_clf_assignments(self):
        """Assigns classifiers to equivalence classes based on their initial
        accuracy estimates. Sorts the classifiers by their accuracy and
        assigns each block to an equivalence class. Tries to assign equal
        number of classifiers to each equivalence class, except possibly
        one equivalence class.
        """
        # perm = range(0, self.num_classifiers)
        # numpy.random.shuffle(perm)
        temp = []
        for i, c in enumerate(self.clf):
            temp.append((c.accuracy, i))
        temp.sort()
        perm = [i for (_, i) in temp]

        equiv_class = 0
        count = 0
        for i in perm:
            self.clf_assignments[i] = equiv_class
            print self.clf[i].accuracy,
            count = count + 1
            if count == self.num_classifiers / self.num_equiv:
                equiv_class = equiv_class + 1
                count = 0
                print "\n"

    def update_clf_assignments(self, h):
        for i in xrange(0, self.num_classifiers):
            self.clf_assignments[i] = h.eqv[i].value

    def clf_reassignment(self, h):
        reassigned = False
        for i, c in enumerate(self.clf):
            max_likelihood = -1
            best_equiv_class = 0
            for j in xrange(0, self.num_equiv):
                # likelihood of c's accuracy in jth equiv class
                l = h.get_likelihood(c.accuracy, j)
                if l > max_likelihood:
                    max_likelihood = l
                    best_equiv_class = j

            if self.clf_assignments[i] != best_equiv_class:
                reassigned = True

            self.clf_assignments[i] = best_equiv_class
        return reassigned

    def clf_reassignment_anticipatory(self, h, clf_assignments, accuracy):
        reassigned = False
        for i in xrange(0, len(accuracy)):
            max_likelihood = -1
            best_equiv_class = 0
            for j in xrange(0, self.num_equiv):
                # likelihood of c's accuracy in jth equiv class
                l = h.get_likelihood(accuracy[i], j)
                if l > max_likelihood:
                    max_likelihood = l
                    best_equiv_class = j

            if clf_assignments[i] != best_equiv_class:
                reassigned = True

            clf_assignments[i] = best_equiv_class
        return reassigned

    def mcmc(self, accuracy_set):
        h = hierarchical.Hierarchical(self.num_equiv,
                                      self.num_classifiers, accuracy_set)
        h.mcmc_sampling()
        return h

        # Update model parameters using current accuracies.
        # Perform classifier re-assignment until convergence
        # h = None
        # iteration = 0
        # stop = False
        # while stop == False and iteration < self.num_mcmc_steps:
        #     acc_by_equiv_class = self.acc_by_class()
        #     h = hierarchical.Hierarchical(self.num_equiv,
        #             self.num_classifiers, acc_by_equiv_class)
        #     h.mcmc_sampling()
        #     stop = self.clf_reassignment(h)
        #     iteration = iteration + 1
        # print "..........Assignments stabilized after %d iterations" %
        # iteration
        # return h

    # def anticipatory_mcmc(self, accuracy):
    #     clf_assignments = self.clf_assignments

    #     h = None
    #     iteration = 0
    #     stop = False
    #     while stop == False and iteration < self.num_mcmc_steps:
    #         acc_by_equiv_class = \
    #             self.acc_by_class_anticipatory(clf_assignments, accuracy)
    #         h = hierarchical.Hierarchical(acc_by_equiv_class, self.num_equiv)
    #         h.mcmc_sampling()
    #         stop = self.clf_reassignment_anticipatory(h, clf_assignments,
    #                                                     accuracy)
    #         iteration = iteration + 1
    #     print "..........Assignments stabilized after %d iterations" %
    #     iteration
    #     return h

    def infinite_loop(self):
        """The main infinite loop for active learning framework.
        """
        # x_axis = []
        # y_axis = []
        for count in xrange(0, self.num_egs - self.num_seeds):
            print "*************************************"
            print "Iteration = %d" % (count + 1)

            # Calculate the accuracy of each classifier.
            for c in self.clf:
                self.calculate_classifier_accuracy(c)

            acc_set = []
            for c in self.clf:
                acc_set.append(c.accuracy)
            acc_set.sort()
            h_true = self.mcmc(acc_set)
            # return h_true

            # Calculate current loss
            self.curr_loss = self.calculate_loss(h_true)
            print "Current loss = %d" % self.curr_loss

            # x_axis.append(count)
            # y_axis.append(self.curr_loss)
            if self.out_of_budget():
                print "Out of budget! Terminating..."
                return h_true

            loss_array = []
            top_disagree = self.top_disagree_egs(10)
            for i in top_disagree:

                print "\n---Trying example %d" % i
                expected_loss = 0.0
                for g in [0, 1]:
                    print "______With label %d" % g
                    new_accuracy = self.estimate_new_accuracy(i, g)
                    # h = self.anticipatory_mcmc(new_accuracy)
                    h = self.mcmc(new_accuracy)
                    loss = self.calculate_loss(h)
                    expected_loss = expected_loss + 0.5 * loss

                print "______Expected loss = %f" % expected_loss
                loss_array.append((i, self.curr_loss - expected_loss))

            index, loss_redn = max(loss_array, key=operator.itemgetter(1))

            if loss_redn <= 0:
                print "Cannot reduce loss further. Terminating..."
                return h_true
            else:
                print "Best unlabelled example = %d, loss reduction = %f" \
                    % (index, loss_redn)

            if self.request_gold_label(index) is False:
                print "Out of budget! Terminating..."
                return h_true

        return h_true
        # self.plot_graph(x_axis, y_axis)

    def print_classifier_accuracies(self):
        """Prints classifiers' accuracies in sorted order
        """
        sorted_accuracies = [c.accuracy for c in self.clf]
        sorted_accuracies.sort()
        print sorted_accuracies

    def print_classifier_predictions(self):
        """Prints predictions of classifiers on the input data.
        """
        for c in self.clf:
            print c.predicted_label


def main():
    active_fr = ActiveFramework(num_classifiers=50,
                                num_equiv=2,
                                num_egs=500,
                                num_seeds=499,
                                min_loss_per_step=0,
                                budget=float('inf'),
                                gold_label_cost=10,
                                num_mcmc_steps=20)
    active_fr.print_classifier_accuracies()
    active_fr.calc_classifier_accuracies()
    # active_fr.print_classifier_predictions()
    active_fr.seed_gold_labels()
    index = active_fr.top_disagree_egs(1)[0]
    if active_fr.request_gold_label(index) is False:
        print "Out of budget! Terminating..."
        return

    # active_fr.initial_clf_assignments()
    h_true = active_fr.infinite_loop()
    active_fr.update_clf_assignments(h_true)

    print active_fr.acc_by_class()
    h_true.plot()
    # dump(ActiveFramework)


def dump(a):
    with open('active.pkl', 'wb') as output:
        pickle.dump(a, output, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
