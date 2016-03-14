import numpy

class DatasetSim(object):
    """Class for a simulated dataset.
    
    Attributes:
        data (List[int]): List of examples in the dataset.
        gold_requested (dict): Map of Index of item -> obtained gold label. 
        mode (str): Simulation mode. 'uniform' for uniformly generated
            target labels.
        num_egs (int): Number of examples in the simulated dataset.
        target (List[int]): Target labels.
    """
    def __init__(self, num_egs, mode):
        self.num_egs = num_egs
        self.mode = mode
        self.data = [0] * num_egs     #TODO(Shobhit): Create Example class
        self.target = self.__generate_target_labels(mode)
        self.gold_requested = {}
        # self.crowd_requested = 

    def __generate_target_labels(self, mode):
        """Synthetically generates target labels.
        
        Args:
            mode (str): Mode of generation. 'uniform' for uniformly generated
                target labels.
        
        Returns:
            List[int]: List of synthetically generated target labels.
        """
        if mode == 'uniform':
            return self.__uniform_target_labels()

    def __uniform_target_labels(self):
        """Synthetically generated target labels by sampling from
        a uniform(0,1) distribution.
        
        Returns:
            List[int]: List of synthetically generated target labels.
        """
        gold_label = []
        for i in xrange(0, self.num_egs):
            r = numpy.random.uniform(0,1)
            gold_label.append(1 if r >= 0.5 else 0)

        return gold_label

