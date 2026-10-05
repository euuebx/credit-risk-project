#include "Scorer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

Scorer::Scorer(double debtWeight, double utilizationWeight, double latePaymentWeight)
    : debtWeight_(debtWeight),
      utilizationWeight_(utilizationWeight),
      latePaymentWeight_(latePaymentWeight) {
    const double total = debtWeight_ + utilizationWeight_ + latePaymentWeight_;
    if (debtWeight_ < 0 || utilizationWeight_ < 0 || latePaymentWeight_ < 0 ||
        total <= 0) {
        throw std::invalid_argument("Scoring weights must be non-negative and not all zero.");
    }
    debtWeight_ /= total;
    utilizationWeight_ /= total;
    latePaymentWeight_ /= total;
}

ScoreResult Scorer::score(const Customer& customer) const {
    // Missing income or credit limit counts as maximum risk (ratio = 1.0).
    const double debtToIncome = customer.annualIncome > 0.0
                                    ? customer.totalDebt / customer.annualIncome
                                    : 1.0;
    const double creditUtilization = customer.creditLimit > 0.0
                                         ? customer.creditUsed / customer.creditLimit
                                         : 1.0;
    const double latePaymentRate = static_cast<double>(customer.latePayments) / 12.0;

    const double weighted = debtWeight_ * clamp(debtToIncome, 0.0, 1.0) +
                            utilizationWeight_ * clamp(creditUtilization, 0.0, 1.0) +
                            latePaymentWeight_ * clamp(latePaymentRate, 0.0, 1.0);

    // Saturating transform: risk rises steeply at first, then flattens towards 1.
    const double risk = clamp(1.0 - std::exp(-1.5 * weighted), 0.0, 1.0);

    ScoreResult result;
    result.score = risk;
    result.debtToIncome = clamp(debtToIncome, 0.0, 1.0);
    result.creditUtilization = clamp(creditUtilization, 0.0, 1.0);
    result.latePaymentRate = clamp(latePaymentRate, 0.0, 1.0);
    result.category = categoryFor(risk);
    return result;
}

double Scorer::clamp(double value, double lo, double hi) {
    return std::max(lo, std::min(value, hi));
}

std::string Scorer::categoryFor(double score) {
    if (score < 0.33) return "LOW";
    if (score < 0.66) return "MEDIUM";
    return "HIGH";
}
