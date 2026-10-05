#ifndef CUSTOMER_H
#define CUSTOMER_H

#include <string>

// One unlabeled customer record from the heuristic-mode CSV.
struct Customer {
    std::string id;
    std::string name;
    double annualIncome = 0.0;
    double totalDebt = 0.0;
    double creditLimit = 0.0;
    double creditUsed = 0.0;
    int latePayments = 0;
};

#endif
