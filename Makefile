CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -O2

LIB = src/CSVReader.cpp src/Scorer.cpp src/GermanData.cpp src/LogisticRegression.cpp src/Metrics.cpp
HEADERS = src/Customer.h src/CSVReader.h src/Scorer.h src/GermanData.h src/LogisticRegression.h src/Metrics.h

.PHONY: all test clean

all: bin/credit_risk

bin/credit_risk: src/main.cpp $(LIB) $(HEADERS)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/main.cpp $(LIB) -o bin/credit_risk

test: bin/tests
	./bin/tests

bin/tests: src/tests.cpp $(LIB) $(HEADERS)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) src/tests.cpp $(LIB) -o bin/tests

clean:
	rm -rf bin reports
