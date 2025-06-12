#include <iostream>
#include <vector>

using namespace std;

vector<vector<int>> multiplyMatrices(const vector<vector<int>>& matrix1, const vector<vector<int>>& matrix2) {
    int rows1 = matrix1.size();
    int cols1 = matrix1[0].size();
    int rows2 = matrix2.size();
    int cols2 = matrix2[0].size();

    if (cols1 != rows2) {
        cerr << "Error: Incompatible matrix dimensions for multiplication.\n";
        return {}; // Return an empty matrix to indicate an error
    }

    vector<vector<int>> result(rows1, vector<int>(cols2, 0));

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

int main() {
    // Example matrices
    vector<vector<int>> matrix1 = {{1, 2}, {3, 4}};
    vector<vector<int>> matrix2 = {{5, 6}, {7, 8}};

    // Multiply the matrices
    vector<vector<int>> result = multiplyMatrices(matrix1, matrix2);

    // Print the result
    if (!result.empty()) {
        for (const auto& row : result) {
            for (int val : row) {
                cout << val << " ";
            }
            cout << endl;
        }
    }

    return 0;
}
