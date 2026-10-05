#include "GermanData.h"
#include "CSVReader.h"

#include <fstream>
#include <stdexcept>
#include <utility>

LabeledDataset GermanData::read(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Could not open input file: " + path);

    LabeledDataset dataset;
    std::string line;
    std::size_t lineNumber = 0;

    if (!std::getline(file, line)) {
        throw std::runtime_error("Input file is empty: " + path);
    }
    ++lineNumber;

    const std::vector<std::string> header = CSVReader::splitRow(line);
    if (header.size() < 2) {
        throw std::runtime_error("Header must contain at least one feature and a label column.");
    }
    dataset.featureNames.assign(header.begin(), header.end() - 1);
    const std::size_t expectedFields = header.size();

    while (std::getline(file, line)) {
        ++lineNumber;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        const std::vector<std::string> fields = CSVReader::splitRow(line);
        if (fields.size() != expectedFields) {
            throw std::runtime_error("Expected " + std::to_string(expectedFields) +
                                     " fields on line " + std::to_string(lineNumber) +
                                     ", got " + std::to_string(fields.size()) + ".");
        }

        std::vector<double> row;
        row.reserve(dataset.featureNames.size());
        for (std::size_t i = 0; i + 1 < fields.size(); ++i) {
            try {
                row.push_back(std::stod(fields[i]));
            } catch (...) {
                throw std::runtime_error("Invalid number for '" + dataset.featureNames[i] +
                                         "' on line " + std::to_string(lineNumber) +
                                         ": '" + fields[i] + "'");
            }
        }

        int label = -1;
        try {
            label = std::stoi(fields.back());
        } catch (...) {
            throw std::runtime_error("Invalid label on line " +
                                     std::to_string(lineNumber) + ": '" + fields.back() + "'");
        }
        if (label != 0 && label != 1) {
            throw std::runtime_error("Label must be 0 or 1 on line " +
                                     std::to_string(lineNumber) + ": '" + fields.back() + "'");
        }

        dataset.features.push_back(std::move(row));
        dataset.labels.push_back(label);
    }

    if (dataset.features.empty()) {
        throw std::runtime_error("Input file contains no data rows.");
    }
    return dataset;
}
