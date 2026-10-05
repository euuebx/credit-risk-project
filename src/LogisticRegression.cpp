#include "LogisticRegression.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

LogisticRegression::LogisticRegression(double learningRate, int maxIterations,
                                       double tolerance)
    : learningRate_(learningRate),
      maxIterations_(maxIterations),
      tolerance_(tolerance) {
    if (learningRate_ <= 0 || maxIterations_ <= 0 || tolerance_ <= 0) {
        throw std::invalid_argument("Learning rate, iteration count and tolerance must be positive.");
    }
}

void LogisticRegression::fit(const std::vector<std::vector<double>>& features,
                             const std::vector<int>& labels) {
    if (features.empty() || features.size() != labels.size()) {
        throw std::invalid_argument("Features and labels must be non-empty and of equal length.");
    }

    const std::size_t rows = features.size();
    const std::size_t dims = features[0].size();
    if (dims == 0) throw std::invalid_argument("Feature vectors must not be empty.");

    // Standardisation statistics from the training set.
    means_.assign(dims, 0.0);
    stds_.assign(dims, 0.0);
    for (const auto& row : features) {
        if (row.size() != dims) {
            throw std::invalid_argument("All feature vectors must have the same length.");
        }
        for (std::size_t j = 0; j < dims; ++j) means_[j] += row[j];
    }
    for (std::size_t j = 0; j < dims; ++j) means_[j] /= static_cast<double>(rows);
    for (const auto& row : features) {
        for (std::size_t j = 0; j < dims; ++j) {
            const double diff = row[j] - means_[j];
            stds_[j] += diff * diff;
        }
    }
    for (std::size_t j = 0; j < dims; ++j) {
        stds_[j] = std::max(std::sqrt(stds_[j] / static_cast<double>(rows)), 1e-12);
    }

    // Precompute standardised features once.
    std::vector<std::vector<double>> x(rows, std::vector<double>(dims));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < dims; ++j) {
            x[i][j] = (features[i][j] - means_[j]) / stds_[j];
        }
    }

    weights_.assign(dims, 0.0);
    bias_ = 0.0;

    // Batch gradient descent on the logistic loss.
    for (int iteration = 0; iteration < maxIterations_; ++iteration) {
        std::vector<double> gradient(dims, 0.0);
        double biasGradient = 0.0;

        for (std::size_t i = 0; i < rows; ++i) {
            double z = bias_;
            for (std::size_t j = 0; j < dims; ++j) z += weights_[j] * x[i][j];
            const double error = sigmoid(z) - static_cast<double>(labels[i]);
            biasGradient += error;
            for (std::size_t j = 0; j < dims; ++j) gradient[j] += error * x[i][j];
        }

        const double scale = learningRate_ / static_cast<double>(rows);
        double maxStep = std::fabs(scale * biasGradient);
        bias_ -= scale * biasGradient;
        for (std::size_t j = 0; j < dims; ++j) {
            const double step = scale * gradient[j];
            weights_[j] -= step;
            maxStep = std::max(maxStep, std::fabs(step));
        }

        ++iterationsRun_;
        if (maxStep < tolerance_) break;
    }
}

double LogisticRegression::predictProbability(const std::vector<double>& features) const {
    if (features.size() != weights_.size()) {
        throw std::invalid_argument("Feature vector length does not match the fitted model.");
    }
    double z = bias_;
    for (std::size_t j = 0; j < weights_.size(); ++j) {
        z += weights_[j] * (features[j] - means_[j]) / stds_[j];
    }
    return sigmoid(z);
}

double LogisticRegression::sigmoid(double z) {
    if (z >= 0) return 1.0 / (1.0 + std::exp(-z));
    const double e = std::exp(z);
    return e / (1.0 + e);
}
