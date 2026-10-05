#include "CSVReader.h"
#include "GermanData.h"
#include "LogisticRegression.h"
#include "Metrics.h"
#include "Scorer.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string input;  // required for heuristic mode
    std::string output = "reports";
    std::string summary;
    std::string train;
    double testSize = 0.2;
    unsigned seed = 42;
    bool noReports = false;
    bool quiet = false;
};

struct CategoryCounts {
    std::size_t low = 0;
    std::size_t medium = 0;
    std::size_t high = 0;
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n\n"
                 "Modes:\n"
                 "  (default)        Score customers with the hand-weighted heuristic\n"
                 "  --train <file>   Train and evaluate a logistic regression on labeled data\n\n"
                 "Options:\n"
                 "  --input <path>    Customer CSV (required for heuristic mode)\n"
                 "  --out <dir>       Output directory for reports (default: reports)\n"
                 "  --summary <file>  Optional JSON summary output\n"
                 "  --test-size <f>   Test fraction for --train, in (0, 1) (default: 0.2)\n"
                 "  --seed <n>        RNG seed for the train/test split (default: 42)\n"
                 "  --no-reports      Skip per-customer report files (large inputs)\n"
                 "  --quiet           Suppress per-customer console output\n"
                 "  --help            Show this help\n";
}

static Options parseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            usage(argv[0]);
            std::exit(0);
        }

        if (argument == "--input" || argument == "--out" ||
            argument == "--summary" || argument == "--train") {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + argument);
            const std::string value = argv[++i];
            if (value.empty() || value.rfind("--", 0) == 0) {
                throw std::runtime_error("Invalid value for " + argument);
            }
            if (argument == "--input") options.input = value;
            else if (argument == "--out") options.output = value;
            else if (argument == "--summary") options.summary = value;
            else options.train = value;
        } else if (argument == "--test-size") {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for --test-size");
            options.testSize = std::stod(argv[++i]);
            if (options.testSize <= 0.0 || options.testSize >= 1.0) {
                throw std::runtime_error("--test-size must be between 0 and 1.");
            }
        } else if (argument == "--seed") {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for --seed");
            options.seed = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (argument == "--no-reports") {
            options.noReports = true;
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }
    return options;
}

// ---------------------------------------------------------------------------
// Heuristic mode
// ---------------------------------------------------------------------------

static std::string sanitizeFilename(const std::string& value) {
    std::string result = value;
    for (char& c : result) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') {
            c = '_';
        }
    }
    return result.empty() ? "customer" : result;
}

static std::string escapeJson(const std::string& value) {
    std::ostringstream out;
    for (const char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c)) << std::dec;
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

static CategoryCounts countCategories(const std::vector<ScoreResult>& results) {
    CategoryCounts counts;
    for (const ScoreResult& result : results) {
        if (result.category == "LOW") ++counts.low;
        else if (result.category == "MEDIUM") ++counts.medium;
        else ++counts.high;
    }
    return counts;
}

static void writeReport(const fs::path& directory, const Customer& customer,
                          const ScoreResult& result) {
    const fs::path path = directory / (sanitizeFilename(customer.id) + ".txt");
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Could not create customer report: " + path.string());

    file << "CREDIT RISK REPORT\n"
            "==================\n\n"
            "Customer ID: " << customer.id << "\n"
            "Name: " << customer.name << "\n\n"
            "Financial Information\n"
            "---------------------\n"
            << std::fixed << std::setprecision(2)
            << "Annual income:       " << customer.annualIncome << "\n"
            << "Total debt:          " << customer.totalDebt << "\n"
            << "Credit limit:        " << customer.creditLimit << "\n"
            << "Credit used:         " << customer.creditUsed << "\n"
            << "Late payments:       " << customer.latePayments << "\n\n"
            "Normalized Features\n"
            "-------------------\n"
            << std::setprecision(4)
            << "Debt-to-income:      " << result.debtToIncome << "\n"
            << "Credit utilization:  " << result.creditUtilization << "\n"
            << "Late-payment rate:   " << result.latePaymentRate << "\n\n"
            "Risk Assessment\n"
            "---------------\n"
            << "Risk score:          " << result.score << "\n"
            << "Category:            " << result.category << "\n";
}

static void writeSummaryJson(const std::string& path,
                               const std::vector<Customer>& customers,
                               const std::vector<ScoreResult>& results) {
    const CategoryCounts counts = countCategories(results);
    double total = 0.0;
    for (const ScoreResult& result : results) total += result.score;

    std::ofstream file(path);
    if (!file) throw std::runtime_error("Could not create summary file: " + path);

    file << "{\n"
         << "  \"customer_count\": " << customers.size() << ",\n"
         << "  \"average_risk_score\": " << std::fixed << std::setprecision(4)
         << (customers.empty() ? 0.0 : total / static_cast<double>(customers.size())) << ",\n"
         << "  \"categories\": {\n"
         << "    \"LOW\": " << counts.low << ",\n"
         << "    \"MEDIUM\": " << counts.medium << ",\n"
         << "    \"HIGH\": " << counts.high << "\n"
         << "  },\n"
         << "  \"customers\": [\n";

    for (std::size_t i = 0; i < customers.size(); ++i) {
        file << "    {\n"
             << "      \"id\": \"" << escapeJson(customers[i].id) << "\",\n"
             << "      \"name\": \"" << escapeJson(customers[i].name) << "\",\n"
             << "      \"risk_score\": " << std::fixed << std::setprecision(4)
             << results[i].score << ",\n"
             << "      \"category\": \"" << results[i].category << "\",\n"
             << "      \"debt_to_income\": " << results[i].debtToIncome << "\n"
             << "      \"credit_utilization\": " << results[i].creditUtilization << "\n"
             << "      \"late_payment_rate\": " << results[i].latePaymentRate << "\n"
             << "    }" << (i + 1 < customers.size() ? "," : "") << "\n";
    }

    file << "  ]\n"
         << "}\n";
}

static void runHeuristicMode(const Options& options) {
    if (options.input.empty()) {
        throw std::runtime_error("Heuristic mode requires --input <path> "
                                 "(a customer CSV in the 7-field format).");
    }
    const std::vector<Customer> customers = CSVReader::read(options.input);
    fs::create_directories(options.output);

    const Scorer scorer;
    std::vector<ScoreResult> results;
    results.reserve(customers.size());

    std::cout << "Credit Risk Scoring Engine\n"
                 "==========================\n"
                 "Input:  " << options.input << "\n"
                 "Output: " << options.output << "\n\n";

    double totalScore = 0.0;
    for (const Customer& customer : customers) {
        const ScoreResult result = scorer.score(customer);
        results.push_back(result);
        if (!options.noReports) writeReport(options.output, customer, result);
        totalScore += result.score;
        if (!options.quiet) {
            std::cout << std::fixed << std::setprecision(3)
                      << customer.id << " | " << customer.name
                      << " | score=" << result.score << " | " << result.category << "\n";
        }
    }

    if (!options.summary.empty()) {
        writeSummaryJson(options.summary, customers, results);
        std::cout << "\nJSON summary: " << options.summary << "\n";
    }

    const CategoryCounts counts = countCategories(results);
    std::cout << "\nSummary\n"
                 "-------\n"
                 "Customers processed: " << customers.size() << "\n"
                 "Low risk:            " << counts.low << "\n"
                 "Medium risk:         " << counts.medium << "\n"
                 "High risk:           " << counts.high << "\n"
                 "Average risk score:  " << std::fixed << std::setprecision(4)
                 << totalScore / static_cast<double>(customers.size()) << "\n";
}

// ---------------------------------------------------------------------------
// Train/evaluate mode
// ---------------------------------------------------------------------------

static void splitTrainTest(const std::vector<std::vector<double>>& features,
                             const std::vector<int>& labels,
                             double testFraction,
                             unsigned seed,
                             std::vector<std::vector<double>>& trainFeatures,
                             std::vector<std::vector<double>>& testFeatures,
                             std::vector<int>& trainLabels,
                             std::vector<int>& testLabels) {
    std::vector<std::size_t> indices(features.size());
    for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = i;

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    const std::size_t testCount =
        static_cast<std::size_t>(static_cast<double>(features.size()) * testFraction);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (i < testCount) {
            testFeatures.push_back(features[indices[i]]);
            testLabels.push_back(labels[indices[i]]);
        } else {
            trainFeatures.push_back(features[indices[i]]);
            trainLabels.push_back(labels[indices[i]]);
        }
    }
}

// The original heuristic: 40/35/25 on its three ratio features.
// Used whenever a dataset starts with the heuristic's features
// (debt_to_income, credit_utilization, late_payment_rate): the
// heuristic only ever sees these three, whatever else the fitted
// model uses. The saturating transform is monotone, so the raw
// weighted sum ranks customers identically (AUC unchanged).
static std::vector<double> heuristicBaselineScores(
    const std::vector<std::vector<double>>& features) {
    std::vector<double> scores;
    scores.reserve(features.size());
    for (const auto& row : features) {
        scores.push_back(0.40 * row[0] + 0.35 * row[1] + 0.25 * row[2]);
    }
    return scores;
}

// Hand-picked weights on min-max normalised features, for datasets
// whose features are not ratios (German Credit data).
static std::vector<double> handPickedBaselineScores(
    const std::vector<std::vector<double>>& features,
    const std::vector<std::vector<double>>& trainFeatures) {
    const std::vector<double> weights = {0.20, 0.30, 0.15, 0.05, 0.10, 0.10, 0.10};

    const std::size_t dims = trainFeatures[0].size();
    if (dims != weights.size()) {
        throw std::runtime_error("Hand-picked baseline weights are defined for " +
                                 std::to_string(weights.size()) + " features.");
    }

    std::vector<double> mins(dims, std::numeric_limits<double>::infinity());
    std::vector<double> maxs(dims, -std::numeric_limits<double>::infinity());
    for (const auto& row : trainFeatures) {
        for (std::size_t j = 0; j < dims; ++j) {
            mins[j] = std::min(mins[j], row[j]);
            maxs[j] = std::max(maxs[j], row[j]);
        }
    }

    std::vector<double> scores;
    scores.reserve(features.size());
    for (const auto& row : features) {
        double score = 0.0;
        for (std::size_t j = 0; j < dims; ++j) {
            const double range = maxs[j] - mins[j];
            const double normalised = range > 1e-12 ? (row[j] - mins[j]) / range : 0.0;
            score += weights[j] * normalised;
        }
        scores.push_back(score);
    }
    return scores;
}

static void writeTrainSummaryJson(const std::string& path, const Options& options,
                                    const LabeledDataset& dataset,
                                    std::size_t trainRows, std::size_t testRows,
                                    const LogisticRegression& model,
                                    const Evaluation& fitted,
                                    const std::string& baselineMethod,
                                    double baselineAuc) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Could not create summary file: " + path);

    file << "{\n"
         << "  \"dataset\": \"" << escapeJson(options.train) << "\",\n"
         << "  \"rows\": " << dataset.features.size() << ",\n"
         << "  \"train_rows\": " << trainRows << ",\n"
         << "  \"test_rows\": " << testRows << ",\n"
         << "  \"seed\": " << options.seed << ",\n"
         << "  \"logistic_regression\": {\n"
         << "    \"iterations\": " << model.iterationsRun() << ",\n"
         << "    \"auc\": " << std::fixed << std::setprecision(4) << fitted.auc << ",\n"
         << "    \"accuracy\": " << fitted.accuracy << ",\n"
         << "    \"true_positive\": " << fitted.truePositive << ",\n"
         << "    \"false_positive\": " << fitted.falsePositive << ",\n"
         << "    \"true_negative\": " << fitted.trueNegative << ",\n"
         << "    \"false_negative\": " << fitted.falseNegative << ",\n"
         << "    \"coefficients\": {\n";

    for (std::size_t j = 0; j < dataset.featureNames.size(); ++j) {
        file << "      \"" << escapeJson(dataset.featureNames[j]) << "\": "
             << std::fixed << std::setprecision(4) << model.weights()[j]
             << (j + 1 < dataset.featureNames.size() ? "," : "") << "\n";
    }

    file << "    }\n"
         << "  },\n"
         << "  \"baseline\": {\n"
         << "    \"method\": \"" << escapeJson(baselineMethod) << "\",\n"
         << "    \"auc\": " << std::fixed << std::setprecision(4) << baselineAuc << "\n"
         << "  }\n"
         << "}\n";
}

static void runTrainMode(const Options& options) {
    const LabeledDataset dataset = GermanData::read(options.train);

    int positives = 0;
    for (const int label : dataset.labels) positives += label;
    if (positives == 0 || positives == static_cast<int>(dataset.labels.size())) {
        throw std::runtime_error("Training data must contain both classes (0 and 1).");
    }

    std::vector<std::vector<double>> trainFeatures, testFeatures;
    std::vector<int> trainLabels, testLabels;
    splitTrainTest(dataset.features, dataset.labels, options.testSize, options.seed,
                   trainFeatures, testFeatures, trainLabels, testLabels);

    int testPositives = 0;
    for (const int label : testLabels) testPositives += label;
    if (testLabels.empty() || testPositives == 0 ||
        testPositives == static_cast<int>(testLabels.size())) {
        throw std::runtime_error("Test split does not contain both classes; "
                                 "use a smaller --test-size.");
    }

    std::cout << "Credit Risk Model Training\n"
                 "==========================\n"
                 "Dataset:          " << options.train << "\n"
                 "Rows:             " << dataset.features.size() << "\n"
                 "Features:         " << dataset.featureNames.size() << "\n"
                 "Train/test split: " << trainFeatures.size() << " / " << testFeatures.size()
                 << " (test fraction " << options.testSize << ", seed " << options.seed << ")\n\n";

    // Fitted model: weights learned from the labels.
    LogisticRegression model(0.1, 2000, 1e-6);
    model.fit(trainFeatures, trainLabels);

    std::vector<double> probabilities;
    probabilities.reserve(testFeatures.size());
    for (const auto& row : testFeatures) {
        probabilities.push_back(model.predictProbability(row));
    }
    const Evaluation fitted = Metrics::evaluate(probabilities, testLabels);

    // Baseline: the hand-weighted heuristic. On Lending Club data it is
    // the original 40/35/25 formula on its three ratios; elsewhere it is
    // hand-picked weights on min-max normalised features.
    const bool hasHeuristicFeatures =
        dataset.featureNames.size() >= 3 &&
        dataset.featureNames[0] == "debt_to_income" &&
        dataset.featureNames[1] == "credit_utilization" &&
        dataset.featureNames[2] == "late_payment_rate";
    const std::string baselineMethod = hasHeuristicFeatures
        ? "40/35/25 heuristic on its 3 features"
        : "hand-picked weights on min-max normalised features";
    const std::vector<double> baseline = hasHeuristicFeatures
        ? heuristicBaselineScores(testFeatures)
        : handPickedBaselineScores(testFeatures, trainFeatures);
    const double baselineAuc = Metrics::auc(baseline, testLabels);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Logistic regression (batch gradient descent, learning rate 0.1)\n"
              << "  Iterations:   " << model.iterationsRun() << "\n"
              << "  AUC:          " << fitted.auc << "\n"
              << "  Accuracy:     " << fitted.accuracy << "\n"
              << "  Confusion:    TP=" << fitted.truePositive
              << " FP=" << fitted.falsePositive
              << " TN=" << fitted.trueNegative
              << " FN=" << fitted.falseNegative << "\n\n"
              << "Learned coefficients (effect per standard deviation):\n";
    for (std::size_t j = 0; j < dataset.featureNames.size(); ++j) {
        std::cout << "  " << std::left << std::setw(20) << dataset.featureNames[j]
                  << (model.weights()[j] >= 0 ? "+" : "") << model.weights()[j] << "\n";
    }

    std::cout << "\nBaseline (" << baselineMethod << ")\n"
              << "  AUC:          " << baselineAuc << "\n\n"
              << "Fitted model improves AUC by " << (fitted.auc - baselineAuc)
              << " over the baseline.\n";

    if (!options.summary.empty()) {
        writeTrainSummaryJson(options.summary, options, dataset,
                              trainFeatures.size(), testFeatures.size(),
                              model, fitted, baselineMethod, baselineAuc);
        std::cout << "\nJSON summary: " << options.summary << "\n";
    }
}

int main(int argc, char** argv) {
    try {
        const Options options = parseArgs(argc, argv);
        if (!options.train.empty()) {
            runTrainMode(options);
        } else {
            runHeuristicMode(options);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
