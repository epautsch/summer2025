#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

int main() {
  std::ifstream outputFile("output.txt");
  if (!outputFile.is_open()) {
    std::cerr << "Error opening output.txt" << std::endl;
    return 1;
  }

  std::vector<float> results;
  float value;
  while (outputFile >> value) {
    results.push_back(value);
  }

  outputFile.close();

  bool passed = true;
  for (float result : results) {
    if (std::abs(result - 1024.0) > 0.001) {
      passed = false;
      std::cerr << "Verification failed: Expected 1024.0, got " << result << std::endl;
      break;
    }
  }

  if (passed) {
    std::cout << "Verification passed!" << std::endl;
  } else {
    std::cout << "Verification failed!" << std::endl;
  }

  return 0;
}
