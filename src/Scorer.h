#ifndef SCORER_H
#define SCORER_H

#include "Customer.h"

#include <string>

struct ScoreResult {
    double score = 0.0;            // 0..1, higher = riskier
    double debtToIncome = 0.0;     // clamped to 0..1
    double creditUtilization = 0.0;  // clamped to 0..1
    double latePaymentRate = 0.0;  // clamped to 0..1
    std::string category;          // LOW, MEDIUM or HIGH
};

// Hand-weighted baseline heuristic. Weights are normalised to sum to 1.
class Scorer {
public:
    Scorer(double debtWeight = 0.40,
           double utilizationWeight = 0.35,
           double latePaymentWeight = 0.25);

    ScoreResult score(const Customer& customer) const;

private:
    double debtWeight_;
    double utilizationWeight_;
    double latePaymentWeight_;

    static double clamp(double value, double lo, double hi);
    static std::string categoryFor(double score);
};

#endif
