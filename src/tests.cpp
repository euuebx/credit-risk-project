#include "CSVReader.h"
#include "GermanData.h"
#include "LogisticRegression.h"
#include "Metrics.h"
#include "Scorer.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const std::string& message) {
    if (condition) {
        std::cout << "  PASS  " << message << "\n";
    } else {
        std::cout << "  FAIL  " << message << "\n";
        ++failures;
    }
}

static bool approx(double actual, double expected, double tolerance = 1e-3) {
    return std::fabs(actual - expected) <= tolerance;
}

static void testScorer() {
    std::cout << "Scorer\n";

    Customer customer;
    customer.id = "C001";
    customer.name = "James Murphy";
    customer.annualIncome = 72000.0;
    customer.totalDebt = 18000.0;
    customer.creditLimit = 30000.0;
    customer.creditUsed = 7500.0;
    customer.latePayments = 0;

    const Scorer scorer;
    ScoreResult result = scorer.score(customer);
    check(approx(result.score, 0.2452), "C001 risk score is 0.2452");
    check(result.category == "LOW", "C001 is LOW risk");
    check(approx(result.debtToIncome, 0.25), "debt-to-income is 0.25");
    check(approx(result.creditUtilization, 0.25), "credit utilization is 0.25");

    Customer emma = customer;
    emma.id = "C004";
    emma.name = "Emma Walsh";
    emma.annualIncome = 35000.0;
    emma.totalDebt = 28000.0;
    emma.creditLimit = 12000.0;
    emma.creditUsed = 11500.0;
    emma.latePayments = 8;
    result = scorer.score(emma);
    check(approx(result.score, 0.7086), "C004 risk score is 0.7086");
    check(result.category == "HIGH", "C004 is HIGH risk");

    Customer sarah = customer;
    sarah.id = "C002";
    sarah.name = "Sarah O'Brien";
    sarah.annualIncome = 48000.0;
    sarah.totalDebt = 22000.0;
    sarah.creditLimit = 18000.0;
    sarah.creditUsed = 14500.0;
    sarah.latePayments = 3;
    result = scorer.score(sarah);
    check(approx(result.score, 0.5470), "C002 risk score is 0.5470");
    check(result.category == "MEDIUM", "C002 is MEDIUM risk");

    Customer noIncome = customer;
    noIncome.annualIncome = 0.0;
    result = scorer.score(noIncome);
    check(result.debtToIncome == 1.0, "zero income counts as maximum debt-to-income");

    // Weights are normalised: 2/2/1 behaves like 0.4/0.4/0.2.
    const Scorer normalised(2.0, 2.0, 1.0);
    Customer maxed = customer;
    maxed.totalDebt = maxed.annualIncome;
    maxed.creditUsed = maxed.creditLimit;
    maxed.latePayments = 12;
    result = normalised.score(maxed);
    check(approx(result.score, 1.0 - std::exp(-1.5)), "maxed-out customer scores 1-exp(-1.5)");
}

static void testCSVReader() {
    std::cout << "CSVReader\n";

    {
        std::ofstream file("test_customers_tmp.csv");
        file << "id,name,annual_income,total_debt,credit_limit,credit_used,late_payments\n"
                "C001,\"Murphy, James\",72000,18000,30000,7500,0\n"
                "C002,Jane Doe,50000,10000,20000,5000,\n"
                "C003   Bob   60000   5000   25000   10000   1\n";
    }
    const std::vector<Customer> customers = CSVReader::read("test_customers_tmp.csv");
    check(customers.size() == 3, "reads three rows (CSV, quoted, missing field, whitespace)");
    check(customers[0].name == "Murphy, James", "quoted field containing a comma parses correctly");
    check(customers[1].latePayments == 0, "empty field is treated as 0");
    check(customers[2].name == "Bob", "whitespace-separated row is supported");
    check(approx(customers[2].annualIncome, 60000.0), "whitespace row numeric field parses");
    std::remove("test_customers_tmp.csv");

    {
        std::ofstream file("test_customers_tmp.csv");
        file << "id,name,annual_income,total_debt,credit_limit,credit_used,late_payments\n"
                "C001,Bob,not_a_number,0,0,0,0\n";
    }
    bool threw = false;
    try {
        CSVReader::read("test_customers_tmp.csv");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "invalid number throws");
    std::remove("test_customers_tmp.csv");
}

static void testMetrics() {
    std::cout << "Metrics\n";

    check(approx(Metrics::auc({0.1, 0.4, 0.35, 0.8}, {0, 0, 1, 1}), 0.75),
          "AUC of hand-checked example is 0.75");
    check(approx(Metrics::auc({0.1, 0.2, 0.8, 0.9}, {0, 0, 1, 1}), 1.0),
          "perfect ranking has AUC 1.0");
    check(approx(Metrics::auc({0.5, 0.5, 0.5, 0.5}, {0, 0, 1, 1}), 0.5),
          "all-tied scores have AUC 0.5");

    const Evaluation evaluation = Metrics::evaluate({0.6, 0.4, 0.8, 0.2}, {1, 0, 1, 0});
    check(evaluation.truePositive == 2 && evaluation.trueNegative == 2,
          "confusion matrix counts correctly");
    check(approx(evaluation.accuracy, 1.0), "accuracy is 1.0 for a perfect threshold");
}

static void testLogisticRegression() {
    std::cout << "LogisticRegression\n";

    std::vector<std::vector<double>> features;
    std::vector<int> labels;
    for (int i = 0; i < 100; ++i) {
        const double x = -1.0 + 2.0 * i / 99.0;
        const double y = -1.0 + 2.0 * ((i * 7) % 100) / 99.0;
        features.push_back({x, y});
        labels.push_back(x + y > 0 ? 1 : 0);
    }

    LogisticRegression model(0.1, 2000, 1e-6);
    model.fit(features, labels);

    std::vector<double> probabilities;
    for (const auto& row : features) {
        probabilities.push_back(model.predictProbability(row));
    }
    const Evaluation evaluation = Metrics::evaluate(probabilities, labels);

    check(model.weights().size() == 2, "model fits one weight per feature");
    check(evaluation.auc >= 0.98, "AUC >= 0.98 on separable data");
    check(evaluation.accuracy >= 0.94, "accuracy >= 0.94 on separable data");
}

static void testGermanData() {
    std::cout << "GermanData\n";

    std::ifstream probe("data/german_credit.csv");
    if (!probe) {
        std::cout << "  SKIP  data/german_credit.csv not found; run from the project root\n";
        return;
    }

    const LabeledDataset dataset = GermanData::read("data/german_credit.csv");
    check(dataset.features.size() == 1000, "German credit dataset has 1000 rows");
    check(dataset.featureNames.size() == 7, "dataset has 7 numeric features");
    check(dataset.features.size() == dataset.labels.size(), "one label per row");

    int highRisk = 0;
    for (const int label : dataset.labels) highRisk += label;
    check(highRisk > 0 && highRisk < static_cast<int>(dataset.labels.size()),
          "dataset contains both classes");

    // Smoke test: the model should reach a reasonable in-sample AUC
    // on the real data (numeric features only).
    LogisticRegression model(0.1, 2000, 1e-6);
    model.fit(dataset.features, dataset.labels);
    std::vector<double> probabilities;
    for (const auto& row : dataset.features) {
        probabilities.push_back(model.predictProbability(row));
    }
    const double auc = Metrics::auc(probabilities, dataset.labels);
    check(auc >= 0.6, "logistic regression reaches AUC >= 0.6 on the full dataset " +
                          std::string("(got ") + std::to_string(auc) + ")");
}

int main() {
    std::cout << "Running tests...\n\n";
    testScorer();
    testCSVReader();
    testMetrics();
    testLogisticRegression();
    testGermanData();

    std::cout << "\n";
    if (failures == 0) {
        std::cout << "All tests passed.\n";
        return 0;
    }
    std::cout << failures << " test(s) failed.\n";
    return 1;
}
