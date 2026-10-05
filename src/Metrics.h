#ifndef METRICS_H
#define METRICS_H

#include <vector>

struct Evaluation {
    double auc = 0.0;      // ROC AUC (NaN if only one class is present)
    double accuracy = 0.0; // Fraction correct at the given threshold
    int truePositive = 0;
    int falsePositive = 0;
    int trueNegative = 0;
    int falseNegative = 0;
};

class Metrics {
public:
    // ROC AUC via the Mann-Whitney U statistic, with average ranks for ties.
    static double auc(const std::vector<double>& scores,
                      const std::vector<int>& labels);

    static Evaluation evaluate(const std::vector<double>& probabilities,
                             const std::vector<int>& labels,
                             double threshold = 0.5);
};

#endif
