#ifndef CSVREADER_H
#define CSVREADER_H

#include "Customer.h"

#include <cstddef>
#include <string>
#include <vector>

class CSVReader {
public:
    // Reads the unlabeled customer CSV used by the heuristic scorer.
    static std::vector<Customer> read(const std::string& path);

    // Parses one CSV row, honouring double-quoted fields ("" escapes a quote).
    // Fields are trimmed. Throws on an unclosed quote.
    static std::vector<std::string> splitRow(const std::string& line);

private:
    static std::vector<std::string> splitWhitespace(const std::string& line);
    static std::string trim(const std::string& value);
    static std::string lower(const std::string& value);
    static double parseDouble(const std::string& value, const std::string& field, std::size_t line);
    static int parseInt(const std::string& value, const std::string& field, std::size_t line);
};

#endif
