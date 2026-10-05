#ifndef GERMAN_DATA_H
#define GERMAN_DATA_H

#include <cstddef>
#include <string>
#include <vector>

// Labeled dataset for training and evaluation.
// labels[i] is the risk label (1 = high risk / defaulted, 0 = low risk)
// for feature row features[i].
struct LabeledDataset {
    std::vector<std::vector<double>> features;  // [row][feature]
    std::vector<int> labels;                    // 0 or 1, one per row
    std::vector<std::string> featureNames;
};

class GermanData {
public:
    // Reads a labeled CSV: header row of feature names plus a final label
    // column, then one row per record. Numeric fields only.
    static LabeledDataset read(const std::string& path);
};

#endif
