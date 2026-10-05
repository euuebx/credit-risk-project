#include "Metrics.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

double Metrics::auc(const std::vector<double>& scores, const std::vector<int>& labels) {
    if (scores.size() != labels.size()) {
        throw std::invalid_argument("Scores and labels must have the same length.");
    }

    int positives = 0;
    int negatives = 0;
    for (const int label : labels) {
        if (label == 1) ++positives;
        else if (label == 0) ++negatives;
        else throw std::invalid_argument("Labels must be 0 or 1.");
    }
    if (positives == 0 || negatives == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Rank scores ascending, averaging ranks within groups of tied scores.
    std::vector<std::size_t> order(scores.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return scores[a] < scores[b]; });

    double rankSumPositive = 0.0;
    std::size_t i = 0;
    while (i < order.size()) {
        std::size_t j = i;
        while (j < order.size() && scores[order[j]] == scores[order[i]]) ++j;
        const double averageRank = (static_cast<double>(i) + static_cast<double>(j) + 1.0) / 2.0;
        for (std::size_t k = i; k < j; ++k) {
            if (labels[order[k]] == 1) rankSumPositive += averageRank;
        }
        i = j;
    }

    return (rankSumPositive - static_cast<double>(positives) * (positives + 1) / 2.0) /
           (static_cast<double>(positives) * static_cast<double>(negatives));
}

Evaluation Metrics::evaluate(const std::vector<double>& probabilities,
                             const std::vector<int>& labels,
                             double threshold) {
    if (probabilities.size() != labels.size()) {
        throw std::invalid_argument("Probabilities and labels must have the same length.");
    }

    Evaluation result;
    result.auc = auc(probabilities, labels);

    int correct = 0;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const bool predictedPositive = probabilities[i] >= threshold;
        const bool actualPositive = labels[i] == 1;
        if (predictedPositive && actualPositive) {
            ++result.truePositive;
            ++correct;
        } else if (predictedPositive && !actualPositive) {
            ++result.falsePositive;
        } else if (!predictedPositive && actualPositive) {
            ++result.falseNegative;
        } else {
            ++result.trueNegative;
            ++correct;
        }
    }

    result.accuracy = labels.empty()
                          ? 0.0
                          : static_cast<double>(correct) / static_cast<double>(labels.size());
    return result;
}
