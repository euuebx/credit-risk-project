#include "CSVReader.h"

#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

std::vector<Customer> CSVReader::read(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Could not open input file: " + path);

    std::vector<Customer> customers;
    std::string line;
    std::size_t lineNumber = 0;
    bool firstRow = true;

    while (std::getline(file, line)) {
        ++lineNumber;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (trim(line).empty()) continue;

        // Rows without commas fall back to whitespace separation.
        const std::vector<std::string> fields =
            line.find(',') != std::string::npos ? splitRow(line) : splitWhitespace(line);

        if (firstRow) {
            firstRow = false;
            const std::string header = lower(trim(fields[0]));
            if (header == "id" || header == "customer_id") continue;  // header row
        }

        std::vector<std::string> padded = fields;
        if (padded.size() < 7) {
            for (std::size_t i = padded.size(); i < 7; ++i) {
                std::cerr << "Warning: missing field on line " << lineNumber
                          << ", treating as empty.\n";
            }
            padded.resize(7);
        }
        if (padded.size() > 7) {
            throw std::runtime_error("Too many fields on line " + std::to_string(lineNumber) +
                                     ". Expected 7 fields.");
        }

        Customer customer;
        customer.id = trim(padded[0]);
        customer.name = trim(padded[1]);
        if (customer.id.empty()) {
            throw std::runtime_error("Missing customer ID on line " + std::to_string(lineNumber));
        }
        if (customer.name.empty()) customer.name = "Unknown";

        customer.annualIncome = parseDouble(padded[2], "annual income", lineNumber);
        customer.totalDebt = parseDouble(padded[3], "total debt", lineNumber);
        customer.creditLimit = parseDouble(padded[4], "credit limit", lineNumber);
        customer.creditUsed = parseDouble(padded[5], "credit used", lineNumber);
        customer.latePayments = parseInt(padded[6], "late payments", lineNumber);
        customers.push_back(customer);
    }

    if (customers.empty()) throw std::runtime_error("Input file contains no customer records.");
    return customers;
}

std::vector<std::string> CSVReader::splitRow(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                current += '"';
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (c == ',' && !quoted) {
            fields.push_back(trim(current));
            current.clear();
        } else {
            current += c;
        }
    }

    if (quoted) throw std::runtime_error("Unclosed quote in CSV row.");
    fields.push_back(trim(current));
    return fields;
}

std::vector<std::string> CSVReader::splitWhitespace(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream in(line);
    std::string field;
    while (in >> field) fields.push_back(field);
    return fields;
}

std::string CSVReader::trim(const std::string& value) {
    std::size_t begin = 0, end = value.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
    return value.substr(begin, end - begin);
}

std::string CSVReader::lower(const std::string& value) {
    std::string result = value;
    for (char& c : result) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return result;
}

double CSVReader::parseDouble(const std::string& value, const std::string& field,
                              std::size_t line) {
    const std::string text = trim(value);
    if (text.empty()) {
        std::cerr << "Warning: empty " << field << " on line " << line << ", treating as 0.\n";
        return 0.0;
    }
    try {
        std::size_t parsed = 0;
        const double number = std::stod(text, &parsed);
        if (parsed != text.size() || number < 0) throw std::invalid_argument("out of range");
        return number;
    } catch (...) {
        throw std::runtime_error("Invalid " + field + " on line " + std::to_string(line) +
                                 ": '" + text + "'");
    }
}

int CSVReader::parseInt(const std::string& value, const std::string& field, std::size_t line) {
    const std::string text = trim(value);
    if (text.empty()) {
        std::cerr << "Warning: empty " << field << " on line " << line << ", treating as 0.\n";
        return 0;
    }
    try {
        std::size_t parsed = 0;
        const long number = std::stol(text, &parsed);
        if (parsed != text.size() || number < 0 || number > 1000000) {
            throw std::invalid_argument("out of range");
        }
        return static_cast<int>(number);
    } catch (...) {
        throw std::runtime_error("Invalid " + field + " on line " + std::to_string(line) +
                                 ": '" + text + "'");
    }
}
