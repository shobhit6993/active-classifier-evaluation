import pymc
import numpy
from pymc.Matplot import plot as mcplot
from bisect import bisect_right
from matplotlib import pyplot as plt


class Hierarchical(object):
    """Class for the Hierarchical model constructed to model the distribution
    of classifier accuracies.

    Attributes:
        accuracy (TYPE): Description
        alpha (list): List of RV parameterizing Beta dist. of each equiv. class
        beta (list): List of RV parameterizing Beta dist. of each equiv. class
        mcmc (list): Description
        model (list): Description
        mu_alpha (PYMC.Stochastic): mu param of lognormal prior for alpha
        mu_beta (PYMC.Stochastic): mu param of lognormal prior for beta
        num_equiv (int): Number of equivalence classes.
        obs (list of list): List of List of observations (classifier accuracy),
            one List for each equivalence class.
        sigma (PYMC.Stochastic): Description

    Deleted Attributes:
        num_classifiers (int): Total number of classifiers.
    """

    def __init__(self, num_equiv, num_classifiers, observations):
        self.post_pred = []
        self.num_bins = 100

        self.num_equiv = num_equiv
        self.num_classifiers = num_classifiers
        self.observations = observations

        # Hyperprior for mean of Normal
        self.mu_star = []
        self.__setup_mu_star()

        # Hyperprior for sd of Normal
        self.sigma_star = []
        self.__setup_sigma_star()

        # Prior for mu of LogitNormal
        self.mu = []
        self.__setup_mu()

        # Prior for sigma of LogitNormal
        self.sigma = []
        self.__setup_sigma()

        # Hyperprior for parameters of Categorical
        self.theta = []
        self.__setup_theta()

        # Equivalence class of each classifier
        self.eqv = []
        self.__setup_eqv()

        # Observed accuracies, modelled as LogitNormal
        self.obs = []
        self.__setup_obs()

        # Posterior predictive for accuracies
        self.predictive = []
        self.__setup_predictive()

        self.model_dict = {}
        self.model = None
        self.__build_model()

        self.mcmc = None

        self.num_iter = 100
        self.num_burn = 50
        self.thin = 2

    def __setup_mu_star(self):
        # self.mu_star = pymc.Normal('mu_star', mu = 0,
        #                             tau = 10**(-4), value = 0.0)
        self.mu_star = pymc.Uniform('mu_star', lower=-4,
                                    upper=4, value=0.0)

    def __setup_sigma_star(self):
        # Jeffer'y prior for sigma_star
        # self.log_sigma_star = pymc.Uniform('log_sigma_star',
        #                             lower = -3.0,
        #                             upper = 1.0,
        #                             value = -1.0)

        # self.sigma_star = pymc.Lambda('sigma_star',
        # lambda s = self.log_sigma_star: numpy.exp(s))

        self.sigma_star = pymc.Uniform('sigma_star',
                                       lower=0.0,
                                       upper=2.0,
                                       value=1.0)

    def __setup_mu(self):
        """Populates the self.mu list with RVs corresponding to mu param
        of the Logit-Normal distribution, one for each equivalence class.
        """
        # self.mu = pymc.Container([pymc.Uniform('mu_%s' % j,
        #                                     lower = -4,
        #                                     upper = 4,
        #                                     value = 0)
        #                         for j in xrange(0, self.num_equiv)])
        self.mu = pymc.Container([pymc.Normal('mu_%s' % j,
                                              mu=self.mu_star,
                                              tau=1.0 / (self.sigma_star**2),
                                              value=0.6)
                                  for j in xrange(0, self.num_equiv)])

    def __setup_sigma(self):
        """Populates the self.sigma list with RVs corresponding to sigma param
        of the Logit-normal distribution, one for each equivalence class.
        """
        self.sigma = pymc.Container([pymc.Uniform('sigma_%s' % j,
                                                  lower=0.01,
                                                  upper=0.3,
                                                  value=0.15)
                                     for j in xrange(0, self.num_equiv)])
        # for j in xrange(0, self.num_equiv):
        # The range is selected based on discussion in report
        #     self.sigma.append(pymc.Uniform('sigma_%s' % j,
        #                                     lower = 0.05,
        #                                     upper = 0.3,
        #                                     value = 0.15))

    def __setup_theta(self):
        # init_val = [1.0/self.num_equiv]*(self.num_equiv - 1)
        self.theta = [[0.5] * self.num_equiv] * self.num_classifiers
        # dirichlet = \
        #     pymc.Container([pymc.Dirichlet('dir_%s' % i,
        #                                    theta=[0.1] * self.num_equiv)
        #                     for i in xrange(0, self.num_classifiers)])

        # self.theta = pymc.Container([pymc.CompletedDirichlet('theta_%s' % i,
        #                                                      dirichlet[i])
        # for i in xrange(0, self.num_classifiers)])

    def __setup_eqv(self):
        """Populates the self.eqv list for each classifier by assigning it
        a categorical distribution.
        """
        # per_class = self.num_classifiers / self.num_equiv
        self.eqv = pymc.Container(
            [pymc.Categorical('categ_%s' % i,
                              p=self.theta[i],
                              value=numpy.random.randint(0, self.num_equiv))
             # value=min(i / per_class, self.num_equiv - 1))
             for i in xrange(0, self.num_classifiers)])

    def __setup_obs(self):
        self.obs = pymc.Container(
            [pymc.Normal('obs_%s' % i,
                         mu=pymc.Lambda('omu_%s' % i,
                                        lambda cls=self.eqv[i]: self.mu[cls]),
                         tau=pymc.Lambda('otau_%s' % i,
                                         lambda cls=self.eqv[i]:
                                         1.0 / (self.sigma[cls]**2)),
                         value=self.logit(acc),
                         observed=True)
             for i, acc in enumerate(self.observations)])

    def __setup_predictive(self):
        for j in xrange(0, self.num_equiv):
            self.predictive.append(
                pymc.Lambda('predictive_%s' % j,
                            lambda p=pymc.Normal('log_pred_%s' % j,
                                                 mu=self.mu[j],
                                                 tau=1.0/self.sigma[j]**2):
                            self.inv_logit(p)))

    def __build_model(self):
        self.model_dict['mu_star'] = self.mu_star
        self.model_dict['sigma_star'] = self.sigma_star

        for i in xrange(0, self.num_classifiers):
            self.model_dict['obs_%s' % i] = self.obs[i]
            # self.model_dict['theta_%s' % i] = self.theta[i]
            self.model_dict['categ_%s' % i] = self.eqv[i]

        for j in xrange(0, self.num_equiv):
            self.model_dict['predictive_%s' % j] = self.predictive[j]
            self.model_dict['mu_%s' % j] = self.mu[j]
            self.model_dict['sigma_%s' % j] = self.sigma[j]

        self.model = pymc.Model(self.model_dict)

    def inv_logit(self, x):
        expo = numpy.exp(x)
        return expo / (1.0 + expo)

    def logit(self, x):
        return numpy.log(x / (1.0 - x))

    def __use_custom_step_methods(self):
        for i in xrange(0, self.num_classifiers):
            self.mcmc.use_step_method(
                pymc.DiscreteMetropolis, self.eqv[i],
                proposal_distribution='Prior')

        for i in xrange(0, self.num_equiv):
            self.mcmc.use_step_method(
                pymc.Metropolis, stochastic=self.mu[i], proposal_sd=0.5)
            self.mcmc.use_step_method(
                pymc.Metropolis, stochastic=self.sigma[i], proposal_sd=0.05)

        self.mcmc.use_step_method(
            pymc.Metropolis, stochastic=self.mu_star, proposal_sd=4)
        self.mcmc.use_step_method(
            pymc.Metropolis, stochastic=self.sigma_star, proposal_sd=0.25)

    def mcmc_sampling(self):
        # map_estimate = pymc.MAP(self.model)
        # map_estimate.fit(method='fmin_l_bfgs_b')

        self.mcmc = pymc.MCMC(self.model)
        self.__use_custom_step_methods()
        self.mcmc.sample(iter=self.num_iter,
                         burn=self.num_burn,
                         thin=self.thin,
                         progress_bar=True)

        # print self.mcmc.step_method_dict[self.eqv[0]][0].ratio
        # print self.mcmc.step_method_dict[self.eqv[1]][0].ratio

        self.calc_posterior_pred()

    def calc_posterior_pred(self):
        width = 1.0 / self.num_bins
        self.post_pred = []
        for j in xrange(0, self.num_equiv):
            pred = self.mcmc.trace('predictive_%s' % j)[:]
            # 1 + 2*width because the upper bound in not included in range
            # the bins have left boundary included and right boundary excluded
            h = numpy.histogram(
                pred, bins=numpy.arange(0, 1 + 2 * width, width))
            self.post_pred.append(h)

    def get_likelihood(self, x, j):
        # Bisect rights returns the pos at which a new element x should be
        # inserted in a sorted list. In case x is already present, it returns
        # the position to the right of any existing entries. The right option
        # is needed because the ranges in histogram bins are left inclusive
        # but not right inclusive.The bin boundaries' array is sorted.
        # We'll use it to find the bin corresponding to the input x,
        # and return the bin frequency as the likelihood.
        index = bisect_right(self.post_pred[j][1], x)

        if index > 0:
            index = index - 1
            # Because index i returned above means the bin is given by
            # index [i-1, i). The freq corresponding to this bin is at
            # index i-1 in the freq array. Note that if index=0, then
            # we should not subtract 1.

        return self.post_pred[j][0][index]

    def plot(self):
        self.combined_plot()
        self.auto_plots()
        # self.individual_plots()

    def individual_plots(self):
        for j in xrange(0, self.num_equiv):
            a = self.mcmc.trace('predictive_%s' % j)[:]
            print "predictive__%s" % j
            print numpy.mean(a)
            fig, ax = plt.subplots(1, 1)
            ax.set_autoscaley_on(True)
            plt.hist(a, histtype='stepfilled', bins=40, alpha=0.85,
                     label="pred_%s" % j, color="#A60628", normed=True)
            plt.legend(loc="upper right")
            fig.savefig("./fig/pred_%s" % j + ".png")

        for j in xrange(0, self.num_equiv):
            a = self.mcmc.trace('mu_%s' % j)[:]
            print "mu_%s" % j
            print numpy.mean(a)
            fig, ax = plt.subplots(1, 1)
            ax.set_autoscaley_on(True)
            plt.hist(a, histtype='stepfilled', bins=40, alpha=0.85,
                     label="mu_%s" % j, color="#A60628", normed=True)
            plt.legend(loc="upper right")
            fig.savefig("./fig/mu_%s" % j + ".png")

        for j in xrange(0, self.num_equiv):
            a = self.mcmc.trace('sigma_%s' % j)[:]
            print "sigma_%s" % j
            print numpy.mean(a)
            fig, ax = plt.subplots(1, 1)
            ax.set_autoscaley_on(True)
            plt.hist(a, histtype='stepfilled', bins=40, alpha=0.85,
                     label="sigma_%s" % j, color="#A60628", normed=True)
            plt.legend(loc="upper right")
            fig.savefig("./fig/sigma_%s" % j + ".png")

        a = self.mcmc.trace('mu_star')[:]
        print "mu_star"
        print numpy.mean(a)
        fig, ax = plt.subplots(1, 1)
        ax.set_autoscaley_on(True)
        plt.hist(a, histtype='stepfilled', bins=40, alpha=0.85,
                 label="mu_star_", color="#A60628", normed=True)
        plt.legend(loc="upper right")
        fig.savefig("./mu_star_.png")

        a = self.mcmc.trace('sigma_star')[:]
        print "sigma_star"
        print numpy.mean(a)
        fig, ax = plt.subplots(1, 1)
        ax.set_autoscaley_on(True)
        plt.hist(a, histtype='stepfilled', bins=40, alpha=0.85,
                 label="sigma_star", color="#A60628", normed=True)
        plt.legend(loc="upper right")
        fig.savefig("./sigma_star.png")

    def auto_plots(self):
        mcplot(self.model.predictive_0, common_scale=False)
        mcplot(self.model.predictive_1, common_scale=False)
        # mcplot(self.model.predictive_2, common_scale=False)
        # mcplot(self.model.predictive_3, common_scale=False)
        # mcplot(self.model.predictive_4, common_scale=False)

        mcplot(self.model.mu_0, common_scale=False)
        mcplot(self.model.mu_1, common_scale=False)
        # mcplot(self.model.mu_2, common_scale=False)
        # mcplot(self.model.mu_3, common_scale=False)
        # mcplot(self.model.mu_4, common_scale=False)

        mcplot(self.model.sigma_0, common_scale=False)
        mcplot(self.model.sigma_1, common_scale=False)
        # mcplot(self.model.sigma_2, common_scale=False)
        # mcplot(self.model.sigma_3, common_scale=False)
        # mcplot(self.model.sigma_4, common_scale=False)

        # mcplot(self.model.theta_0, common_scale=False)
        # mcplot(self.model.theta_1, common_scale=False)
        # mcplot(self.model.theta_2, common_scale=False)
        # mcplot(self.model.theta_3, common_scale=False)
        # mcplot(self.model.theta_4, common_scale=False)

        mcplot(self.model.categ_0, common_scale=False)
        mcplot(self.model.categ_1, common_scale=False)
        mcplot(self.model.categ_45, common_scale=False)
        mcplot(self.model.categ_46, common_scale=False)
        # mcplot(self.model.categ_2, common_scale=False)
        # mcplot(self.model.categ_3, common_scale=False)
        # mcplot(self.model.categ_4, common_scale=False)

        # mcplot(self.model.mu_star, common_scale=False)
        # mcplot(self.model.sigma_star, common_scale=False)

    def combined_plot(self):
        color_list = ["blue", "brown", "green", "red", "black",
                      "orange", "aqua", "yellow", "darkorchid",
                      "darkolivegreen"]
        fig, ax = plt.subplots(1, 1)
        for j in xrange(0, self.num_equiv):
            a = self.mcmc.trace('predictive_%s' % j)[:]
            plt.xlim([0, 1])
            plt.hist(a, histtype='step', bins=100, alpha=1.0, lw=2,
                     label="equiv_class_%s" % j, color=color_list[j],
                     normed=True)

        plt.hist(self.observations, histtype='stepfilled', bins=100,
                 alpha=0.2, label="truth", color="#A60628", normed=True)
        ax.set_autoscaley_on(True)
        plt.legend(loc="best", framealpha=0.3)
        fig.savefig("./fig/combined.png")

        # fig, ax = plt.subplots(1, 1)
        # plt.xlim([0, 1])
        # fig.savefig("./fig/combined2.png")
