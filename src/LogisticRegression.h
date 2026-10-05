#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <cstddef>
#include <vector>

// Binary logistic regression fitted with batch gradient descent.
// Features are standardised (zero mean, unit variance) using training-set
// statistics, so coefficients are comparable: each is the effect of a
// one-standard-deviation increase in that feature.
class LogisticRegression {
public:
    LogisticRegression(double learningRate = 0.1,
                       int maxIterations = 2000,
                       double tolerance = 1e-6);

    void fit(const std::vector<std::vector<double>>& features,
             const std::vector<int>& labels);

    // Probability of the positive class (label 1).
    double predictProbability(const std::vector<double>& features) const;

    const std::vector<double>& weights() const { return weights_; }
    const std::vector<double>& means() const { return means_; }
    const std::vector<double>& stds() const { return stds_; }
    double bias() const { return bias_; }
    int iterationsRun() const { return iterationsRun_; }

private:
    double learningRate_;
    int maxIterations_;
    double tolerance_;

    std::vector<double> weights_;
    std::vector<double> means_;
    std::vector<double> stds_;
    double bias_ = 0.0;
    int iterationsRun_ = 0;

    static double sigmoid(double z);
};

#endif
