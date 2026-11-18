CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -pthread -Iinclude
SRC = src/main.cpp
TEST_SRC = tests/test_cpp.cpp
BIN_DIR = bin
TARGET = $(BIN_DIR)/nebula-cpp
TEST_TARGET = $(BIN_DIR)/test-cpp

.PHONY: all clean test demo run

all: $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(TEST_TARGET): $(TEST_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)

demo: $(TARGET)
	./$(TARGET) demo

clean:
	rm -rf $(BIN_DIR)
